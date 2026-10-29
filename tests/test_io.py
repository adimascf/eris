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


def test_collapse_report_gives_the_copy_number_range_and_its_depth_tiers(tmp_path):
    # IS1 on the real-read EC958 assembly: 3 chromosome copies and 2 pEC958 copies, link count 4
    locus = _locus("locus_1", ("hub", 0, 768, Strand.FORWARD))
    locus.copy_number = CopyNumber(5, 4, 5, ((3, 38.04), (2, 77.0)))
    locus.fractional_depth = 5.0

    row = _report(tmp_path, locus, 'collapse')

    assert {k: row[k] for k in list(row)[-5:]} == {
        "copy_number": "5", "copy_number_min": "4", "copy_number_max": "5", "copy_tiers": "3@38.0x,2@77.0x",
        "copy_fallback": "false"}


def test_collapse_report_flags_a_copy_number_from_the_genome_baseline(tmp_path):
    locus = _locus("locus_1", ("hub", 0, 1300, Strand.FORWARD))
    locus.copy_number = CopyNumber(4, 4, 4, ((4, 38.9),), fallback=True)

    assert _report(tmp_path, locus, 'collapse')["copy_fallback"] == "true"


def test_variant_report_keeps_fractional_depth_as_its_last_column(tmp_path):
    locus = _locus("locus_1", ("ctgA", 0, 768, Strand.FORWARD))
    locus.fractional_depth = 0.5

    row = _report(tmp_path, locus, 'variant')

    assert list(row)[-1] == "fractional_depth" and row["fractional_depth"] == "0.5"
