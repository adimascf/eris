import re
from pathlib import Path
from types import SimpleNamespace

from Bio import SeqIO
from Bio.Seq import Seq
from pyfgs import FaaWriter, FnaWriter, GeneFinder, Gff3Writer, Model

from eris.constants import Strand
from eris.copy_number import CopyNumber
from eris.io import GENE_SOURCE, OutputManager, ReportRow, format_contig_path, format_gff_line, format_identities, gene_id
from eris.pipeline import GenomicFeature, LocationSegment, Locus, Pipeline


def test_report_header_puts_target_identity_after_target():
    assert ReportRow.header().split("\t")[:3] == ["locus_id", "target", "target_identity"]


def test_report_header_puts_contig_path_after_target_identity_in_both_modes():
    for mode in ('variant', 'collapse'):
        assert ReportRow.header(mode).split("\t")[2:4] == ["target_identity", "contig_path"]


def test_identities_are_listed_in_target_order_to_four_decimals():
    locus = SimpleNamespace(targets=[SimpleNamespace(identity=0.97314), SimpleNamespace(identity=1.0)])
    assert format_identities(locus) == "0.9731,1.0000"


def _locus(id_, *segments):
    target = GenomicFeature("IS1", None, [LocationSegment(*s) for s in segments], identity=1.0)
    contig = "|".join(s[0] for s in segments)
    return Locus(id_, contig, segments[0][1], segments[-1][2], [target], [], [], [])


def test_contig_path_lists_each_contig_with_the_strand_it_is_read_on():
    stitched = _locus("locus_split_1", ("ctgA", 900, 1000, Strand.FORWARD), ("ctgB", 0, 500, Strand.REVERSE),
                      ("ctgC", 0, 500, Strand.FORWARD))
    assert format_contig_path(stitched) == "ctgA+,ctgB-,ctgC+"


def test_contig_path_of_a_local_locus_is_its_contig_on_the_target_strand():
    assert format_contig_path(_locus("locus_1", ("ctg5", 100, 1100, Strand.REVERSE))) == "ctg5-"


def test_report_and_locus_fasta_carry_the_contig_path(tmp_path):
    locus = _locus("locus_split_1", ("ctgA", 900, 1000, Strand.FORWARD), ("ctgB", 0, 500, Strand.REVERSE))
    prefix = str(tmp_path / "out")
    with OutputManager(prefix, write_gff=False, write_faa=False, write_fna=False, mode='collapse') as out:
        out.write_locus_relations(locus)
        out.write_locus_fasta(locus, "ACGT")

    header, row = (tmp_path / "out_report.tsv").read_text().splitlines()
    assert dict(zip(header.split("\t"), row.split("\t")))["contig_path"] == "ctgA+,ctgB-"
    assert (tmp_path / "out_loci.fasta").read_text() == ">locus_split_1 path=ctgA+,ctgB-\nACGT\n"


def _report(tmp_path, locus, mode):
    with OutputManager(str(tmp_path / "out"), write_gff=False, write_faa=False, write_fna=False, mode=mode) as out:
        out.write_locus_relations(locus)
    header, row = (tmp_path / "out_report.tsv").read_text().splitlines()
    return dict(zip(header.split("\t"), row.split("\t")))


def test_collapse_report_gives_both_copy_numbers_and_the_depth_tiers(tmp_path):
    # IS1 on the real-read EC958 assembly: 3 chromosome copies and 2 pEC958 copies, link count 4
    locus = _locus("locus_1", ("hub", 0, 768, Strand.FORWARD))
    locus.copy_number = CopyNumber(5, 4, ((3, 38.04), (2, 77.0)))
    locus.fractional_depth = 5.0

    row = _report(tmp_path, locus, 'collapse')

    assert {k: row[k] for k in list(row)[-4:]} == {
        "copy_number": "5", "copy_number_edges": "4", "copy_tiers": "3@38.0x,2@77.0x", "copy_fallback": "false"}


def test_collapse_report_flags_a_copy_number_from_the_genome_baseline(tmp_path):
    locus = _locus("locus_1", ("hub", 0, 1300, Strand.FORWARD))
    locus.copy_number = CopyNumber(4, 0, ((4, 38.9),), fallback=True)

    assert _report(tmp_path, locus, 'collapse')["copy_fallback"] == "true"


def test_variant_report_keeps_fractional_depth_as_its_last_column(tmp_path):
    locus = _locus("locus_1", ("ctgA", 0, 768, Strand.FORWARD))
    locus.fractional_depth = 0.5

    row = _report(tmp_path, locus, 'variant')

    assert list(row)[-1] == "fractional_depth" and row["fractional_depth"] == "0.5"


def _assembly(text, reader):
    from io import StringIO
    from eris.io import GenomeAssembly
    return GenomeAssembly.from_stream(StringIO(text), reader, "test")


def test_fasta_header_tag_marks_a_contig_circular_and_closes_it_with_a_self_link():
    from Bio.SeqIO.FastaIO import FastaIterator
    from eris.graph import Edge
    genome = _assembly(">chrom length=5000 circular=true\nACGT\n>frag\nACGT\n", FastaIterator)

    assert genome.circular == {"chrom"}
    assert genome.edges == [Edge("chrom", Strand.FORWARD, "chrom", Strand.FORWARD, 0)]


def test_an_isolated_gfa_self_link_marks_a_contig_circular():
    from eris.io import GfaReader
    # Written on the reverse strand, as SPAdes does: the same link as 'plasmid+ -> plasmid+'
    genome = _assembly("S\tplasmid\tACGTACGT\nL\tplasmid\t-\tplasmid\t-\t3M\n", GfaReader)

    assert genome.circular == {"plasmid"}


def test_a_gfa_self_link_alongside_other_links_is_a_collapsed_repeat_not_a_circular_contig():
    from eris.io import GfaReader
    genome = _assembly("S\trep\tACGT\nS\tctgA\tACGT\nL\trep\t+\trep\t+\t0M\nL\tctgA\t+\trep\t+\t0M\n", GfaReader)

    assert genome.circular == frozenset()


def test_a_hairpin_link_does_not_mark_a_contig_circular():
    from eris.io import GfaReader
    genome = _assembly("S\tctgA\tACGT\nL\tctgA\t+\tctgA\t-\t0M\n", GfaReader)

    assert genome.circular == frozenset()


# Depth tags -----------------------------------------------------------------------------------------------------------
def test_gfa_dp_tags_from_flye_and_unicycler_are_read_as_depth():
    from eris.io import GfaReader
    genome = _assembly("S\tctgA\tACGT\tdp:i:30\nS\tctgB\tACGT\tdp:f:12.5\n", GfaReader)

    assert genome.has_depth and genome.contig_depths == {"ctgA": 30, "ctgB": 12.5}


def test_gfa_uppercase_dp_and_depth_tags_are_read_as_depth():
    from eris.io import GfaReader
    genome = _assembly("S\tctgA\tACGT\tDP:f:30.0\nS\tctgB\tACGT\tdepth:f:12.5\n", GfaReader)

    assert genome.contig_depths == {"ctgA": 30.0, "ctgB": 12.5}


def test_kmer_count_is_not_read_as_depth():
    from eris.io import GfaReader
    genome = _assembly("S\tctgA\tACGT\tKC:i:108\n", GfaReader)

    assert not genome.has_depth and genome.contig_depths == {}


def test_depth_tags_on_only_some_contigs_leave_the_assembly_without_depth():
    from eris.io import GfaReader
    genome = _assembly("S\tctgA\tACGT\tdp:i:30\nS\tctgB\tACGT\n", GfaReader)

    assert not genome.has_depth and genome.contig_depths == {}


def test_fasta_input_has_no_depth_and_no_assembly_graph():
    from Bio.SeqIO.FastaIO import FastaIterator
    genome = _assembly(">ctgA\nACGT\n", FastaIterator)

    assert not genome.has_depth and not genome.has_graph


def test_variant_report_gives_na_for_a_fractional_depth_that_was_not_measured(tmp_path):
    locus = _locus("locus_1", ("ctgA", 0, 768, Strand.FORWARD))
    locus.fractional_depth = None

    assert _report(tmp_path, locus, 'variant')["fractional_depth"] == "NA"


def test_collapse_report_gives_na_for_a_copy_number_that_was_not_measured(tmp_path):
    locus = _locus("locus_1", ("hub", 0, 768, Strand.FORWARD))
    locus.copy_number = CopyNumber(None, 2, hub=frozenset({"hub"}))
    locus.fractional_depth = None

    row = _report(tmp_path, locus, 'collapse')

    assert {k: row[k] for k in list(row)[-4:]} == {
        "copy_number": "NA", "copy_number_edges": "2", "copy_tiers": "NA", "copy_fallback": "NA"}


# Gene files -----------------------------------------------------------------------------------------------------------
GENE_FIXTURE = Path(__file__).parent / "fixtures" / "ec958_genes.fasta"


def _called_genes():
    """Each fixture contig with its pyfgs gene calls, as Pipeline._process_contig makes them."""
    finder = GeneFinder(model=Model.Complete, whole_genome=False)
    contigs = []
    for record in SeqIO.parse(GENE_FIXTURE, "fasta"):
        sequence = bytes(str(record.seq), "ascii")
        contigs.append((record.id, sequence, finder.find_genes(sequence)))
    return contigs


def _write_eris(tmp_path, contigs):
    prefix = str(tmp_path / "eris")
    with OutputManager(prefix) as out:
        for contig_id, sequence, genes in contigs:
            out.write_global_genes(contig_id, sequence, genes)
    return prefix


def _write_pyfgs(tmp_path, contigs):
    prefix = str(tmp_path / "pyfgs")
    with Gff3Writer(f"{prefix}.gff") as gff, FaaWriter(f"{prefix}.faa") as faa, FnaWriter(f"{prefix}.fna") as fna:
        for contig_id, sequence, genes in contigs:
            gff.write_record(genes, contig_id, sequence)
            faa.write_record(genes, contig_id)
            fna.write_record(genes, contig_id)
    return prefix


def _fgs_to_gene_ids(contigs) -> dict[str, str]:
    """pyfgs's FGS_n name for each gene, mapped to the gene ID eris gives it."""
    return {f"{contig_id}_FGS_{n}": gene_id(contig_id, g)
            for contig_id, _, genes in contigs for n, g in enumerate(genes, 1)}


def _renamed(path: str, names: dict[str, str]) -> list[str]:
    """The lines of a pyfgs file with every FGS_n name replaced by its gene ID."""
    pattern = re.compile(r"[^\s=>]+_FGS_\d+")
    return [pattern.sub(lambda m: names[m.group(0)], line) for line in Path(path).read_text().splitlines()]


def test_gene_id_is_contig_start_end_in_one_based_closed_coordinates():
    gene = SimpleNamespace(start=3334642, end=3335068)  # 0-based half-open, as pyfgs calls it
    assert gene_id("HG941718.1", gene) == "HG941718.1_3334643_3335068"


def test_fixture_has_a_frameshifted_gene_and_genes_on_both_strands():
    contigs = _called_genes()
    genes = [(g, s) for _, s, gs in contigs for g in gs]
    assert any(g.mutations(s) for g, s in genes)
    assert {g.strand for g, _ in genes} == {1, -1}


def test_gene_files_equal_pyfgs_output_once_fgs_names_become_gene_ids(tmp_path):
    contigs = _called_genes()
    eris, pyfgs = _write_eris(tmp_path, contigs), _write_pyfgs(tmp_path, contigs)
    names = _fgs_to_gene_ids(contigs)
    assert Path(f"{eris}_assembly.gff").read_text().splitlines() == _renamed(f"{pyfgs}.gff", names)
    assert Path(f"{eris}_proteins.faa").read_text().splitlines() == _renamed(f"{pyfgs}.faa", names)
    assert Path(f"{eris}_cds.fna").read_text().splitlines() == _renamed(f"{pyfgs}.fna", names)


def test_report_gene_id_matches_the_gff_and_fasta_ids(tmp_path):
    contigs = _called_genes()
    prefix = _write_eris(tmp_path, contigs)
    gff_ids = re.findall(r"ID=([^;]+);", Path(f"{prefix}_assembly.gff").read_text())
    faa_ids = re.findall(r"^>(\S+)", Path(f"{prefix}_proteins.faa").read_text(), re.M)
    fna_ids = re.findall(r"^>(\S+)", Path(f"{prefix}_cds.fna").read_text(), re.M)
    # The report names a gene by the id of its GenomicFeature, built in Pipeline._process_contig
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.target_db = SimpleNamespace(aligner=SimpleNamespace(map=lambda *a, **k: []))
    pipeline._gene_finder = GeneFinder(model=Model.Complete, whole_genome=False)
    report_ids = []
    for contig_id, sequence, _ in contigs:
        _, _, _, features, _ = pipeline._process_contig((contig_id, Seq(sequence.decode())))
        report_ids += [f.id for f in features]
    assert report_ids == gff_ids == faa_ids == fna_ids


def test_gene_ids_are_unique_within_a_contig():
    for contig_id, _, genes in _called_genes():
        ids = [gene_id(contig_id, g) for g in genes]
        assert len(ids) == len(set(ids))


def test_gff_notes_every_frameshift_of_a_pseudogene_as_pyfgs_does():
    # pyfgs's Gff3Writer names 'ins' an insertion and joins one note per frameshift with commas
    mutations = [SimpleNamespace(mut_type="ins", pos=120, codon_idx=7), SimpleNamespace(mut_type="del", pos=300, codon_idx=67)]
    gene = SimpleNamespace(start=99, end=600, strand=-1, score=1.304, mutations=lambda sequence: mutations)
    assert format_gff_line("ctg1", b"", gene) == (
        f"ctg1\t{GENE_SOURCE}\tpseudogene\t100\t600\t1.30\t-\t0\tID=ctg1_100_600;"
        f"inference=ab initio prediction:{GENE_SOURCE};pseudogene=unknown;"
        "Note=Frameshift insertion at pos 120 (codon 7),Frameshift deletion at pos 300 (codon 67)\n")
