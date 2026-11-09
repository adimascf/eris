from types import SimpleNamespace

from eris.constants import Strand
from eris.copy_number import CopyNumber
from eris.io import OutputManager, ReportRow, format_contig_path, format_identities
from eris.pipeline import GenomicFeature, LocationSegment, Locus


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
