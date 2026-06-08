from dataclasses import dataclass, field
from typing import Iterable, Optional, Any
from threading import local as thread_local
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import numpy as np

from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation, SeqFeature, CompoundLocation

from mappy import ThreadBuffer

from pyfgs import GeneFinder, Model, Gene, Mutation

from eris.io import TargetDatabase, GenomeAssembly
from eris.graph import TopologyEngine
from eris.alignment import AlignmentBatch, AlignmentRecord
from eris.interval import IntervalBatch, Context, Strand, Interval
from eris.constants import FeatureType, Orientation, Effect


# DataClasses ----------------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class LocationSegment:
    """A single continuous segment of a genomic location (e.g. an exon)."""
    contig: str
    start: int
    end: int
    strand: Strand

    def to_biopython(self) -> SimpleLocation:
        """Convert to a Biopython SimpleLocation object."""
        return SimpleLocation(self.start, self.end, self.strand)


@dataclass(slots=True)
class GenomicFeature:
    """
    Represents a biological feature (e.g. a gene, MGE) which may span multiple segments.

    Example:
        >>> seg = LocationSegment("ctg1", 100, 500, Strand.FORWARD)
        >>> feat = GenomicFeature("geneA", FeatureType.CDS, [seg])
        >>> feat.is_multi_contig
        False
    """
    id: str
    type: FeatureType
    segments: list[LocationSegment]
    qualifiers: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_multi_contig(self) -> bool:
        """True if the feature spans more than one contig."""
        return len(set(seg.contig for seg in self.segments)) > 1

    def to_biopython(self) -> SeqFeature:
        """Convert to a Biopython SeqFeature object."""
        if len(self.segments) == 1:
            loc = self.segments[0].to_biopython()
        else:
            loc = CompoundLocation([s.to_biopython() for s in self.segments])
        return SeqFeature(location=loc, type=self.type.value, id=self.id, qualifiers=self.qualifiers)

    @property
    def bounding_start(self) -> int:
        """The absolute minimum start coordinate of all segments."""
        return self.segments[0].start

    @property
    def bounding_end(self) -> int:
        """The absolute maximum end coordinate of all segments."""
        return self.segments[-1].end

    @property
    def bounding_strand(self) -> Strand:
        """The strand of the primary (first) segment."""
        return self.segments[0].strand


@dataclass(slots=True)
class FeatureRelation:
    """
    Describes the topological relationship between a target and a contextual feature.

    Attributes:
        feature: The contextual (passenger or flanking) GenomicFeature.
        spatial: The spatial Context (e.g. UPSTREAM).
        distance_bp: Physical distance in base pairs.
        topological_dist: Distance in graph hops.
        orientation: Relative orientation (SAME or OPPOSITE).
        effect: Combined Effect flags (e.g. TRUNCATED).
    """
    feature: GenomicFeature
    spatial: Context
    distance_bp: int
    topological_dist: int
    orientation: Orientation
    effect: Effect = Effect.NONE

    def get_relative_position(self, locus_start: int, locus_end: int) -> str:
        """Returns a string describing the relative position to a locus boundary."""
        g_start = self.feature.bounding_start
        g_end = self.feature.bounding_end
        g_strand = self.feature.bounding_strand

        if locus_end <= g_start:
            return "5_prime_flank" if g_strand == Strand.FORWARD else "3_prime_flank"
        elif locus_start >= g_end:
            return "3_prime_flank" if g_strand == Strand.FORWARD else "5_prime_flank"
        return "overlapping"


@dataclass(slots=True)
class Locus:
    """
    A single assembled genomic region containing a target of interest and its context.

    The locus may be a simple interval on one contig or a complex 'stitched' path
    across multiple contigs in an assembly graph.
    """
    id: str
    contig: str
    start: int
    end: int
    targets: list[GenomicFeature]
    passengers: list['FeatureRelation']
    upstream_flanks: list['FeatureRelation']
    downstream_flanks: list['FeatureRelation']
    fractional_depth: float = 1.0  # Tracks the sub-clonal abundance

    def extract_sequence(self, genome: 'GenomeAssembly') -> str:
        """
        Extracts the full nucleotide sequence of the locus from the assembly.

        Handles multi-contig loci by stitching segments together in graph order.
        """
        if "|" not in self.contig:
            # Single-contig locus: Exact coordinate slicing
            # (Biopython Seq objects handle the slicing natively)
            return str(genome[self.contig][self.start:self.end])

        # Multi-contig stitched locus:
        # Note: To get the nucleotide-perfect stitched string without duplicating
        # the graph overlaps, we fetch the full length of the traversed contigs.
        seq_parts = []
        for ctg in self.contig.split('|'):
            seq_parts.append(str(genome[ctg]))

        return "".join(seq_parts)

# Classes --------------------------------------------------------------------------------------------------------------
class LocusBuilder:
    """
    Orchestrates the assembly of loci from raw alignments and genomic context.

    It uses the TopologyEngine to resolve graph-spanning alignments and then
    identifies flanking and passenger genes for each identified locus.
    """
    __slots__ = ('topology_engine', 'genome', 'target_feature_type', 'max_feature_hops',
                 'locus_tolerance', 'features', 'genes')

    def __init__(self, topology_engine: 'TopologyEngine', genome: 'GenomeAssembly',
                 target_feature_type: FeatureType = FeatureType.CDS,
                 max_feature_hops: int = 3, locus_tolerance: int = 0,
                 features: dict[str, list[GenomicFeature]] = None,
                 genes: dict[str, list[Gene]] = None):
        """
        Initialize the LocusBuilder.

        Args:
            topology_engine: Engine for graph traversal.
            genome: The full assembly and metadata.
            target_feature_type: Classification for primary alignment targets.
            max_feature_hops: Max contextual genes to look for in each direction.
            locus_tolerance: bp tolerance for merging adjacent targets.
            features: Dictionary of GenomicFeatures per contig.
            genes: Dictionary of PyFGS Gene objects per contig.
        """
        self.topology_engine = topology_engine
        self.genome = genome
        self.target_feature_type = target_feature_type
        self.max_feature_hops = max_feature_hops
        self.locus_tolerance = locus_tolerance
        self.features = features or {}
        self.genes = genes or {}

    def assemble(self, alignments: dict, min_coverage: float = 0.30) -> Iterable['Locus']:
        """
        The main entry point for generating loci.

        Stitches graph-spanning alignments and processes local alignments
        to produce a sequence of Locus objects.
        """

        # Now expects a list of paths (lists of AlignmentRecords) instead of pairs
        cleaned_alignments, resolved_paths = self.topology_engine.resolve_split_alignments(alignments)

        # Filter and yield STITCHED paths
        for path in resolved_paths:
            # Calculate total coverage of the stitched path against the IS query
            coverage = (path[-1].q_end - path[0].q_start) / path[0].q_length
            if coverage >= min_coverage:
                yield self._stitch(path)

        # Filter and yield LOCAL (unstitched) alignments
        for contig_id, batch in cleaned_alignments.items():
            cov_mask = ((batch.q_ends - batch.q_starts) / batch.q_lengths) >= min_coverage
            intact_batch = batch.filter(cov_mask)

            if len(intact_batch) > 0:
                contig_gene_intervals = self.topology_engine.features.get(contig_id, IntervalBatch.empty())
                for locus in self._build_local(contig_id, intact_batch, contig_gene_intervals):
                    yield locus

    def _resolve_relation(self, contig: str, idx: int, interval_batch: 'IntervalBatch',
                          spatial: Context, dist: int, topo: int, target_strand: Strand,
                          target_bounds: tuple[int, int]) -> FeatureRelation:
        """Analyzes and creates a FeatureRelation for a specific gene-target pair."""

        orig_idx = interval_batch.original_indices[idx]  # type: int
        feature = self.features[contig][orig_idx]  # type: GenomicFeature
        raw_pyfgs_gene = self.genes[contig][orig_idx]  # type: Gene
        gene_strand = Strand(interval_batch.strands[idx])

        # Determine strict relational spatial context natively using coordinate limits for local genes
        if topo == 0:
            target_interval = Interval(target_bounds[0], target_bounds[1], target_strand)
            gene_interval = Interval(interval_batch.starts[idx], interval_batch.ends[idx], gene_strand)
            spatial = target_interval.relate(gene_interval)

        effect = Effect.NONE

        if spatial in (Context.OVERLAPPING, Context.OVERLAPPING_START, Context.OVERLAPPING_END):
            effect = Effect.DISRUPTED

        # If the gene was biologically broken by the insertion
        if raw_pyfgs_gene.insertions or raw_pyfgs_gene.deletions:
            # Dynamically fetch the sequence from the genome for the mutation checker
            for mut in raw_pyfgs_gene.mutations(bytes(self.genome[contig])):  # type: Mutation
                dist_to_start = abs(mut.pos - target_bounds[0])
                dist_to_end = abs(mut.pos - target_bounds[1])

                if dist_to_start <= 5 or dist_to_end <= 5:
                    effect = Effect.TRUNCATED if effect == Effect.NONE else effect | Effect.TRUNCATED
                    break

        if target_strand == Strand.UNSTRANDED or gene_strand == Strand.UNSTRANDED:
            orientation = Orientation.NONE
        elif target_strand == gene_strand:
            orientation = Orientation.SAME
        else:
            orientation = Orientation.OPPOSITE

        return FeatureRelation(
            feature=feature, spatial=spatial, distance_bp=dist,
            topological_dist=topo, orientation=orientation, effect=effect
        )

    def _extract_flanks(self, contig: str, boundary: int, walk_direction: int,
                        context: Context, intervals: Optional['IntervalBatch'], dest_list: list,
                        target_strand: Strand, target_bounds: tuple[int, int]):
        """Walks the contig (and graph) from a boundary to find flanking features."""
        rem_hops = self.max_feature_hops
        exit_strand = Strand.REVERSE if walk_direction == -1 else Strand.FORWARD

        # 1. Local Search
        if intervals:
            if walk_direction == -1:
                idx = np.searchsorted(intervals.ends, boundary, side='right')
                for i in reversed(range(max(0, idx - self.max_feature_hops), idx)):
                    d = max(0, boundary - intervals.ends[i])
                    f = self._resolve_relation(contig, i, intervals, context, int(d), 0, target_strand, target_bounds)
                    dest_list.append(f)
                    rem_hops -= 1
            else:
                idx = np.searchsorted(intervals.starts, boundary, side='left')
                for i in range(idx, min(len(intervals), idx + self.max_feature_hops)):
                    d = max(0, intervals.starts[i] - boundary)
                    f = self._resolve_relation(contig, i, intervals, context, int(d), 0, target_strand, target_bounds)
                    dest_list.append(f)
                    rem_hops -= 1

        # 2. Graph Spillover
        if rem_hops > 0:
            for s_ctg, node_depth, batch in self.topology_engine.traverse(contig, exit_strand, rem_hops):
                for i in range(len(batch)):
                    if walk_direction == -1:
                        d = max(0, boundary + batch.starts[i] - self.genome.contig_lengths[contig])
                    else:
                        d = max(0, batch.starts[i] - boundary)
                    dest_list.append(
                        self._resolve_relation(s_ctg, i, batch, context, int(d), node_depth, target_strand, target_bounds)
                    )

    def _build_local(self, contig: str, alignment_batch: 'AlignmentBatch', gene_intervals: 'IntervalBatch') -> list[
        'Locus']:
        """Identifies loci within a single contig."""
        loci = []
        aln_intervals = alignment_batch.to_intervals()
        macro_intervals = aln_intervals.merge(tolerance=self.locus_tolerance)

        for i in range(len(macro_intervals)):
            macro = macro_intervals[i]
            target_indices = aln_intervals.query(macro.start, macro.end)
            if not (targets := [alignment_batch.get_record(idx) for idx in target_indices]):
                continue

            target_features = [
                GenomicFeature(t.q_name, self.target_feature_type,
                               [LocationSegment(contig, t.t_start, t.t_end, Strand(t.strand))])
                for t in targets
            ]

            # copy number estimation
            if getattr(self.topology_engine, 'mode', 'variant') == 'collapse':
                contig_depth = self.genome.contig_depths.get(contig, 1.0)
                metric_val = float(max(1, round(contig_depth / self.topology_engine.median_depth)))
            else:
                metric_val = 1.0
            locus = Locus(
                id=f"locus_{uuid4().hex[:8]}", contig=contig, start=macro.start, end=macro.end,
                targets=target_features, passengers=[], upstream_flanks=[], downstream_flanks=[],
                fractional_depth=metric_val
            )

            # Determine macro target context bounds
            primary_strand = Strand(targets[0].strand)
            macro_bounds = (macro.start, macro.end)

            internal_indices = gene_intervals.query(macro.start, macro.end)
            for idx in internal_indices:
                locus.passengers.append(
                    self._resolve_relation(contig, idx, gene_intervals, Context.INSIDE, 0, 0, primary_strand,
                                           macro_bounds))

            if primary_strand == Strand.REVERSE:
                u_dir, u_bound = 1, macro.end
                d_dir, d_bound = -1, macro.start
            else:
                u_dir, u_bound = -1, macro.start
                d_dir, d_bound = 1, macro.end

            self._extract_flanks(contig, u_bound, u_dir, Context.UPSTREAM, gene_intervals, locus.upstream_flanks,
                                 primary_strand, macro_bounds)
            self._extract_flanks(contig, d_bound, d_dir, Context.DOWNSTREAM, gene_intervals, locus.downstream_flanks,
                                 primary_strand, macro_bounds)

            loci.append(locus)

        return loci

    def _stitch(self, fragments: list['AlignmentRecord']) -> 'Locus':
        """Stitches multiple fragments into a single multi-contig locus."""
        first = fragments[0]
        last = fragments[-1]

        if getattr(self.topology_engine, 'mode', 'variant') == 'collapse':
            max_path_depth = max(self.genome.contig_depths.get(f.t_name, 1.0) for f in fragments)
            metric_val = float(max(1, round(max_path_depth / self.topology_engine.median_depth)))
        else:
            # Calculate the fractional flow of the stitched path
            source_depth = self.genome.contig_depths.get(first.t_name, 1.0)
            bottleneck_depth = min(self.genome.contig_depths.get(f.t_name, 1.0) for f in fragments)
            metric_val = round(bottleneck_depth / source_depth, 3) if source_depth > 0 else 1.0

        # 1. Build the multi-segment target feature across ALL fragments
        segments = [LocationSegment(f.t_name, f.t_start, f.t_end, Strand(f.strand)) for f in fragments]
        target_feature = GenomicFeature(
            id=f"{first.q_name}_stitched",
            type=self.target_feature_type,
            segments=segments
        )

        locus = Locus(
            id=f"locus_split_{uuid4().hex[:8]}",
            contig="|".join(f.t_name for f in fragments),
            start=first.t_start,
            end=last.t_end,
            targets=[target_feature], passengers=[],
            upstream_flanks=[], downstream_flanks=[],
            fractional_depth=metric_val  # Holds either frac_depth or copy_number; NEW: Assign it to the Locus
        )

        # 2. UPSTREAM FLANKS (Strictly from the first fragment)
        u_dir = -1 if first.strand == 1 else 1
        u_bound = first.t_start if first.strand == 1 else first.t_end
        u_ints = self.topology_engine.features.get(first.t_name)
        self._extract_flanks(
            first.t_name, u_bound, u_dir, Context.UPSTREAM, u_ints, locus.upstream_flanks,
            Strand(first.strand), (first.t_start, first.t_end)
        )

        # 3. DOWNSTREAM FLANKS (Strictly from the last fragment)
        v_dir = 1 if last.strand == 1 else -1
        v_bound = last.t_end if last.strand == 1 else last.t_start
        v_ints = self.topology_engine.features.get(last.t_name)
        self._extract_flanks(
            last.t_name, v_bound, v_dir, Context.DOWNSTREAM, v_ints, locus.downstream_flanks,
            Strand(last.strand), (last.t_start, last.t_end)
        )

        # 4. INSIDE PASSENGERS (Sweep across ALL fragments, including unaligned synthetic bubbles)
        for frag in fragments:
            f_ints = self.topology_engine.features.get(frag.t_name)
            if f_ints:
                internal_indices = f_ints.query(frag.t_start, frag.t_end)
                for idx in internal_indices:
                    locus.passengers.append(
                        self._resolve_relation(
                            frag.t_name, idx, f_ints, Context.INSIDE, 0, 0,
                            Strand(frag.strand), (frag.t_start, frag.t_end)
                        )
                    )

        return locus


class Pipeline:
    """
    High-level eris pipeline manager.

    Orchestrates the entire workflow: contig processing, target mapping,
    gene calling, graph building, and locus assembly. Uses a thread pool
    for parallel contig processing.
    """
    __slots__ = ('target_db', '_gene_finder', 'max_feature_hops', 'locus_tolerance', '_executor', 'mode', 'min_coverage')
    _THREAD_LOCAL = thread_local()

    def __init__(self, target_db: 'TargetDatabase', max_feature_hops: int = 3, locus_tolerance: int = 0,
                 max_workers: Optional[int] = None, mode: str = 'variant', min_coverage: float = 0.30):
        """
        Initialize the Pipeline.

        Args:
            target_db: Database of mapping targets (e.g. antibiotic resistance genes).
            max_feature_hops: Contextual search depth.
            locus_tolerance: Merging tolerance for adjacent hits.
            max_workers: Number of threads for parallel processing.
        """
        self.target_db = target_db
        self.max_feature_hops = max_feature_hops
        self.locus_tolerance = locus_tolerance
        self.mode = mode
        self.min_coverage = min_coverage
        self._gene_finder = GeneFinder(model=Model.Complete, whole_genome=False)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown(cancel_futures=True, wait=False)
        self._gene_finder = None

    def _process_contig(self, contig: tuple[str, Seq]) -> tuple[
        str, Optional[IntervalBatch], Optional[AlignmentBatch], list[GenomicFeature], list[Gene]]:
        """Worker function for processing a single contig (Mapping + Gene Calling)."""

        if not hasattr(self._THREAD_LOCAL, "buf"):
            self._THREAD_LOCAL.buf = ThreadBuffer()

        gene_batch, aln_batch, features, genes = None, None, [], []
        contig_id, contig_seq = contig

        if alns := list(self.target_db.aligner.map(str(contig_seq), buf=self._THREAD_LOCAL.buf)):
            aln_batch = AlignmentBatch.from_mappy(contig_id, len(contig_seq), alns).swap_sides().cull_overlaps()

        if genes := self._gene_finder.find_genes(bytes(contig_seq)):
            num_genes = len(genes)
            starts = np.empty(num_genes, dtype=np.int32)
            ends = np.empty(num_genes, dtype=np.int32)
            strands = np.empty(num_genes, dtype=np.int8)

            for i, g in enumerate(genes):
                starts[i] = g.start
                ends[i] = g.end
                strands[i] = g.strand
                features.append(GenomicFeature(
                    id=f"{contig_id}_{g.start}_{g.end}",
                    type=FeatureType.CDS,
                    segments=[LocationSegment(contig_id, g.start, g.end, Strand(g.strand))]
                ))

            gene_batch = IntervalBatch(starts=starts, ends=ends, strands=strands,
                                       original_indices=np.arange(num_genes, dtype=np.int32)).sort()

        return contig_id, gene_batch, aln_batch, features, genes

    def __call__(self, genome: 'GenomeAssembly', out: Optional[Any] = None) -> Iterable['Locus']:
        """Runs the full pipeline on a genome assembly."""
        alignments, gene_intervals, gene_features, gene_cds = {}, {}, {}, {}

        for contig_id, g_batch, a_batch, features, genes in self._executor.map(self._process_contig, genome):
            if g_batch:
                gene_intervals[contig_id] = g_batch
                gene_features[contig_id] = features
                gene_cds[contig_id] = genes
                if out:
                    out.write_global_genes(contig_id, bytes(genome[contig_id]), genes)
            if a_batch:
                alignments[contig_id] = a_batch

        # for batch in alignments.values():
        #     for n, _ in enumerate(batch.q_names):
        #         print(batch.get_record(n))

        # Calculate global median depth for the collapse mode
        if genome.contig_depths:
            median_depth = max(0.001, float(np.median(list(genome.contig_depths.values()))))
        else:
            median_depth = 1.0

        topology_engine = TopologyEngine(
            edges=genome.edges,
            contig_lengths=genome.contig_lengths,
            contig_depths=genome.contig_depths,
            features=gene_intervals,
            mode=self.mode,
            median_depth=median_depth)

        builder = LocusBuilder(
            topology_engine=topology_engine,
            genome=genome,
            target_feature_type=self.target_db.feature_type,
            max_feature_hops=self.max_feature_hops,
            locus_tolerance=self.locus_tolerance,
            features=gene_features,
            genes=gene_cds
        )

        yield from builder.assemble(alignments, min_coverage=self.min_coverage)