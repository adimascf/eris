"""
Module to handle query (contigs) and target (features) IO.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Union, IO, Iterator, Optional
from pathlib import Path
from re import compile as re_compile
from gzip import open as gzopen
from bz2 import open as bzopen
from lzma import open as lzopen
from sys import stdout

from Bio.Seq import Seq
from Bio.SeqIO.FastaIO import FastaIterator
from Bio.SeqRecord import SeqRecord

from mappy import Aligner

from pyfgs import FaaWriter, FnaWriter, Gff3Writer

from eris.alignment import Cigar
from eris.graph import Edge
from eris.interval import Strand
from eris.constants import FeatureType


# Base classes ---------------------------------------------------------------------------------------------------------
class _Handle:
    def __init__(self, handle: IO):
        self._handle = handle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._handle.close()

# Readers --------------------------------------------------------------------------------------------------------------
class GfaReader(_Handle):
    """
    Reader for Graphical Fragment Assembly (GFA) files.

    Parses Segment (S) and Link (L) lines into SeqRecord and Edge objects.
    """

    @staticmethod
    def _parse_segment(parts: list[str]):
        name = parts[0]
        seq = parts[1]
        tags = {}
        for item in parts[2:]:
            tag, typ, val = item.split(':', maxsplit=2)
            if typ == 'f':
                val = float(val)
            elif typ == 'i':
                val = int(val)
            tags[tag] = val
        return SeqRecord(seq=Seq(seq), id=name, name=name, annotations=tags)

    @staticmethod
    def _parse_link(parts: list[str]):
        u = parts[0]
        u_strand = Strand(parts[1])
        v = parts[2]
        v_strand = Strand(parts[3])
        cigar = Cigar(parts[4])
        overlap = next((n for op, n, _, _, _ in cigar if op == 'M'), 0)
        return Edge(u, u_strand, v, v_strand, overlap)

    @classmethod
    def _parse_line(cls, line):
        if line.startswith('S\t'):
            return cls._parse_segment(line[2:].rstrip().split('\t'))
        elif line.startswith('L\t'):
            return cls._parse_link(line[2:].rstrip().split('\t'))
        else:
            return None

    def __next__(self):
        while True:  # Will naturally raise StopIteration when the handle is exhausted
            if (parsed := self._parse_line(next(self._handle))) is not None:
                return parsed

    def __iter__(self):
        for line in self._handle:
            if (parsed := self._parse_line(line)) is not None:
                yield parsed


# Writers --------------------------------------------------------------------------------------------------------------
class GfaWriter(_Handle):
    """Writer for GFA format files."""

    def write(self, item: Union[Edge, SeqRecord]) -> int:
        """Writes an Edge (Link) or SeqRecord (Segment) to the file."""
        if isinstance(item, Edge):
            return self._handle.write(f"L\t{item.u}\t{item.u_strand}\t{item.v}\t{item.u_strand}\t*\n")
        elif isinstance(item, SeqRecord):
            return self._handle.write(f"S\t{item.id}\t{item.seq}\n")
        raise TypeError(f"Unsupported type: {type(item)}")


# Classes --------------------------------------------------------------------------------------------------------------
class TargetDatabase:
    """
    Manages a database of nucleotide target sequences (e.g. MGEs, ARGs).

    Wraps a minimap2 (mappy) index for efficient searching.

    Example:
        >>> db = TargetDatabase("targets.fasta")
        >>> aligner = db.aligner
    """
    __slots__ = ('path', 'feature_type', 'indexing_threads', '_aligner')

    def __init__(self, path: Union[str, Path], feature_type: FeatureType = FeatureType.CDS, indexing_threads: int = 3):
        """
        Initialize the TargetDatabase.

        Args:
            path: Path to the FASTA or MMI file.
            feature_type: Classification for these targets.
            indexing_threads: Threads to use for on-the-fly indexing.
        """
        self.path = Path(path)
        self.feature_type = feature_type
        self.indexing_threads = indexing_threads  # Only needed if not already indexed
        self._aligner = None

        if not self.path.exists():
            raise FileNotFoundError(f"Target database not found: {self.path}")

    @property
    def aligner(self) -> Aligner:
        """Lazy-loaded mappy.Aligner instance."""
        if self._aligner is None:
            self._aligner = Aligner(fn_idx_in=str(self.path), n_threads=self.indexing_threads)
            if not self._aligner:
                raise ValueError(f"Minimap2 failed to load database: {self.path}")
        return self._aligner


@dataclass(slots=True, frozen=True)
class GenomeAssembly:
    """
    Container for a genome assembly, including contigs and their graph topology.

    Handles FASTA and GFA formats, with support for transparent decompression.

    Example:
        >>> assembly = GenomeAssembly.from_file("assembly.gfa.gz")
        >>> for contig_id, seq in assembly:
        >>>     print(contig_id, len(seq))
    """
    _SEQUENCE_FILE_REGEX = re_compile(
        r'\.('
        r'(?P<fasta>f(asta|a|na|fn|as|aa))|'
        r'(?P<gfa>gfa)|'
        r')\.?(?P<compression>(gz|bz2|xz))?$'
    )
    _OPENERS = {'gz': gzopen, 'bz2': bzopen, 'xz': lzopen}
    id: str
    contigs: dict[str, Seq]
    edges: list[Edge]
    contig_depths: dict[str, float]
    contig_lengths: dict[str, int]

    def __len__(self):
        """Total number of base pairs in the assembly."""
        return sum(len(i) for i in self.contigs.values())

    def __iter__(self) -> Iterator[tuple[str, Seq]]:
        """Iterate over contig IDs and sequences."""
        return iter(self.contigs.items())

    def __str__(self):
        return self.id

    def __getitem__(self, item: str) -> 'Seq':
        """Access a contig sequence by its ID."""
        return self.contigs[item]

    @classmethod
    def from_file(cls, file: Union[str, Path]):
        """
        Load an assembly from a FASTA or GFA file.

        Args:
            file: Path to the file. Supports .gz, .bz2, and .xz compression.
        """
        file = Path(file) # type: Path
        if not (m := cls._SEQUENCE_FILE_REGEX.search(file.name)):
            raise NotImplementedError(f'Unsupported format: {file}')
        reader = FastaIterator if m.group('fasta') else GfaReader
        with cls._OPENERS.get(m.group('compression'), open)(file, mode='rt') as handle:
            return cls.from_stream(handle, reader, file.name.rstrip(m.group()))

    @classmethod
    def from_stream(cls, handle: IO[str], reader, id_: str = None):
        """Load an assembly from an open file stream using the specified reader."""
        contigs, edges, depths, lengths = {}, [], {}, {}
        for record in reader(handle):
            if isinstance(record, SeqRecord):
                contigs[record.id] = record.seq
                depths[record.id] = record.annotations.get('DP', record.annotations.get('depth', 1.0))
                lengths[record.id] =  len(record)
            elif isinstance(record, Edge):
                edges.append(record)
        return cls(id_ or handle.name, contigs, edges, depths, lengths)


@dataclass(slots=True, frozen=True)
class ReportRow:
    """
    Represents a single structural variant or passenger gene record in the eris TSV report.

    Attributes:
        locus_id: Unique identifier for the assembled structural variant locus.
        target: The mobile genetic element(s) or query sequences found in this locus.
        gene_id: The identifier of the contextual passenger or flanking gene.
        context: The spatial relationship (e.g., INSIDE, UPSTREAM) of the gene to the target.
        dist_bp: Distance in base pairs between the gene and the target element.
        topo_hops: Number of graph nodes traversed to find this gene (0 if on the same contig).
        orientation: Strand orientation of the gene relative to the target (same or opposite).
        effect: Biological impact of the insertion on the gene (e.g., TRUNCATED, NONE).
        fractional_depth: The relative read depth of the variant path compared to the source contig, indicating sub-clonal abundance.
    """
    locus_id: str
    target: str
    gene_id: str
    context: str
    dist_bp: int
    topo_hops: int
    orientation: str
    effect: str
    fractional_depth: float
    estimated_copies: int

    @classmethod
    def header(cls) -> str:
        """Returns the TSV header string dynamically generated from the dataclass fields."""
        return "\t".join(cls.__annotations__.keys()) + "\n"

    def to_tsv(self) -> str:
        """Formats the row data into a tab-separated string."""
        return f"{self.locus_id}\t{self.target}\t{self.gene_id}\t{self.context}\t{self.dist_bp}\t{self.topo_hops}\t{self.orientation}\t{self.effect}\t{self.fractional_depth}\t{self.estimated_copies}\n"


class OutputManager:
    """
    Manages all file outputs and stream lifecycles for the eris pipeline.

    Handles writing of the TSV report, GFF3 annotations, protein FASTA,
    and locus nucleotide sequences.

    Example:
        >>> with OutputManager("my_sample") as out:
        >>>     out.write_locus_relations(locus)
    """

    def __init__(self, prefix: Optional[str], write_gff: bool = True, write_faa: bool = True):
        """
        Initialize the OutputManager.

        Args:
            prefix: Filename prefix for all output files. If None, TSV is sent to stdout.
            write_gff: Whether to output an assembly-wide GFF3 file.
            write_faa: Whether to output an assembly-wide protein FASTA file.
        """
        self.prefix = prefix
        self.write_gff = write_gff
        self.write_faa = write_faa
        self._stack = ExitStack()

    def __enter__(self):
        # 1. Setup TSV Report (Stdout or File)
        if self.prefix:
            self.tsv_handle = self._stack.enter_context(open(f"{self.prefix}_report.tsv", "w"))
        else:
            self.tsv_handle = stdout

        # Write TSV Header dynamically from the dataclass
        self.tsv_handle.write(ReportRow.header())

        # 2. Setup Optional PyFGS Writers
        self.gff_writer = None
        self.faa_writer = None

        if self.prefix and self.write_gff:
            self.gff_writer = self._stack.enter_context(Gff3Writer(f"{self.prefix}_assembly.gff"))

        if self.prefix and self.write_faa:
            self.faa_writer = self._stack.enter_context(FaaWriter(f"{self.prefix}_proteins.faa"))

        # 3. Setup Locus FASTA writer
        self.locus_fasta = None
        if self.prefix:
            self.locus_fasta = self._stack.enter_context(open(f"{self.prefix}_loci.fasta", "w"))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)

    # --- Helper methods ---

    def write_locus_relations(self, locus: 'Locus'):
        """Writes all contextual relationships (passengers, flanks) for a locus to the TSV report."""
        for relation in locus.passengers + locus.upstream_flanks + locus.downstream_flanks:
            self._write_tsv_row(locus, relation)

    def _write_tsv_row(self, locus: 'Locus', relation: 'FeatureRelation'):
        """Constructs a ReportRow dataclass and writes it to the TSV handle."""

        copies = getattr(locus, 'estimated_copies', 1)
        row = ReportRow(
            locus_id=locus.id,
            target=",".join(t.id for t in locus.targets),
            gene_id=relation.feature.id,
            context=relation.spatial.name,
            dist_bp=relation.distance_bp,
            topo_hops=relation.topological_dist,
            orientation=relation.orientation.value,
            effect=relation.effect.name,
            fractional_depth=locus.fractional_depth,
            estimated_copies=copies
        )
        self.tsv_handle.write(row.to_tsv())

    def write_global_genes(self, contig_id: str, sequence: Union[str, bytes], genes: list['Gene']):
        """Writes assembly-wide gene calls to the GFF3 and FAA files."""
        if self.gff_writer:
            self.gff_writer.write_record(genes, contig_id, sequence)
        if self.faa_writer:
            self.faa_writer.write_record(genes, contig_id)

    def write_locus_fasta(self, locus_id: str, sequence: str):
        """Writes a single locus nucleotide sequence to the loci FASTA file."""
        if self.locus_fasta:
            self.locus_fasta.write(f">{locus_id}\n{sequence}\n")