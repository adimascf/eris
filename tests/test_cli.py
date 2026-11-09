import argparse

import pytest

from eris.cli import fraction


@pytest.mark.parametrize("value, expected", [("0", 0.0), ("0.9", 0.9), ("1", 1.0)])
def test_fraction_accepts_values_from_zero_to_one(value, expected):
    assert fraction(value) == expected


@pytest.mark.parametrize("value", ["90", "-0.1", "1.01", "high"])
def test_fraction_rejects_values_outside_zero_to_one(value):
    with pytest.raises(argparse.ArgumentTypeError):
        fraction(value)


# Start-up summary and warnings ----------------------------------------------------------------------------------------
from io import StringIO

from Bio.Seq import Seq

from eris.cli import Log, describe_input
from eris.constants import Strand
from eris.graph import Edge
from eris.io import GenomeAssembly

F = Strand.FORWARD


def _genome(contigs, edges=(), depths=None, circular=(), has_graph=True):
    lengths = {c: 100 for c in contigs}
    depths = {c: 10.0 for c in contigs} if depths is None else depths
    return GenomeAssembly("test", {c: Seq("A" * 100) for c in contigs}, list(edges), depths, lengths,
                          frozenset(circular), has_graph)


def test_an_assembly_of_only_circular_contigs_is_named_a_complete_genome():
    genome = _genome(["chrom", "plasmid"], [Edge("chrom", F, "chrom", F), Edge("plasmid", F, "plasmid", F)],
                     circular={"chrom", "plasmid"})

    summary, warnings = describe_input(genome)

    assert summary == "2 contigs, 2 links, 2 circular contigs, depth available: complete genome"
    assert warnings == []


def test_a_graph_with_non_circular_contigs_is_named_a_fragmented_assembly():
    genome = _genome(["ctgA", "ctgB", "plasmid"], [Edge("ctgA", F, "ctgB", F), Edge("plasmid", F, "plasmid", F)],
                     circular={"plasmid"})

    assert describe_input(genome)[0] == "3 contigs, 2 links, 1 circular contig, depth available: fragmented assembly"


def test_fasta_input_without_circular_tags_gets_counts_only_and_a_warning_that_stitching_is_off():
    summary, warnings = describe_input(_genome(["ctgA", "ctgB"], depths={}, has_graph=False))

    assert summary == "2 contigs, 0 links, 0 circular contigs, no depth"
    assert any("stitching and cross-contig flank search are off" in w for w in warnings)


def test_a_complete_genome_from_fasta_is_not_warned_about_stitching():
    genome = _genome(["chrom"], [Edge("chrom", F, "chrom", F)], depths={}, circular={"chrom"}, has_graph=False)

    summary, warnings = describe_input(genome)

    assert summary == "1 contig, 0 links, 1 circular contig, no depth: complete genome"
    assert not any("stitching" in w for w in warnings)


def test_missing_depth_is_warned_about():
    _, warnings = describe_input(_genome(["ctgA"], depths={}))

    assert any("NA" in w for w in warnings)


def test_warnings_are_written_even_when_quiet():
    stream = StringIO()
    log = Log(quiet=True, stream=stream)

    log.msg("progress\n")
    log.warn("careful")

    assert stream.getvalue() == "⚠️  careful\n"


def _run_quiet(tmp_path, monkeypatch, genome_name, genome_text):
    from eris.cli import main
    (tmp_path / genome_name).write_text(genome_text)
    monkeypatch.setattr("sys.argv", ["eris", "-i", str(tmp_path / genome_name), "-d", str(tmp_path / "targets.fasta"),
                                     "-o", str(tmp_path / "out"), "-q"])
    main()


def _contig_with_target(tmp_path):
    """A 4.8 kb contig with an 800 bp target in its middle; the target is written to targets.fasta."""
    import random
    rng = random.Random(1)
    target = "".join(rng.choice("ACGT") for _ in range(800))
    (tmp_path / "targets.fasta").write_text(f">IS1\n{target}\n")
    return "".join(rng.choice("ACGT") for _ in range(2000)) + target + "".join(rng.choice("ACGT") for _ in range(2000))


def test_depth_tags_on_only_some_gfa_contigs_give_na_depth_and_a_warning(tmp_path, monkeypatch, capsys):
    contig = _contig_with_target(tmp_path)
    _run_quiet(tmp_path, monkeypatch, "genome.gfa", f"S\tctgA\t{contig}\tdp:i:30\nS\tctgB\tACGTACGT\n")

    err = capsys.readouterr().err
    assert "reported as NA" in err and "stitching" not in err
    header, *rows = (tmp_path / "out_report.tsv").read_text().splitlines()
    assert rows and {dict(zip(header.split("\t"), row.split("\t")))["fractional_depth"] for row in rows} == {"NA"}


def test_eris_warns_about_fasta_input_even_with_quiet(tmp_path, monkeypatch, capsys):
    contig = _contig_with_target(tmp_path)
    _run_quiet(tmp_path, monkeypatch, "genome.fasta", f">ctgA\n{contig}\n")

    err = capsys.readouterr().err
    assert "stitching and cross-contig flank search are off" in err and "reported as NA" in err
    assert "Loading" not in err and "Input:" not in err
    header, *rows = (tmp_path / "out_report.tsv").read_text().splitlines()
    assert rows and {dict(zip(header.split("\t"), row.split("\t")))["fractional_depth"] for row in rows} == {"NA"}
