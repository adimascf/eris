import dataclasses

import numpy as np
import pytest

from eris.alignment import AlignmentBatch
from eris.constants import Context, Strand
from eris.graph import Edge, AssemblyGraph
from eris.target_end import beyond_contig_end, carried_past_link

F, R = Strand.FORWARD, Strand.REVERSE
UP, DOWN = Context.UPSTREAM, Context.DOWNSTREAM


def _batch(rows, strands=None):
    """rows: (q_length, q_start, q_end, t_name, t_length, t_start, t_end) of one IS1 alignment each; '+' by default."""
    ql, qs, qe, tn, tl, ts, te = zip(*rows)
    n = len(rows)
    blocks = [e - s for s, e in zip(qs, qe)]
    return AlignmentBatch(
        q_names=np.array(["IS1"] * n, dtype=object), q_lengths=np.array(ql, dtype=np.int32),
        q_starts=np.array(qs, dtype=np.int32), q_ends=np.array(qe, dtype=np.int32),
        t_names=np.array(tn, dtype=object), t_lengths=np.array(tl, dtype=np.int32),
        t_starts=np.array(ts, dtype=np.int32), t_ends=np.array(te, dtype=np.int32),
        strands=np.array(strands or [F] * n, dtype=np.int8),
        lengths=np.array(blocks, dtype=np.int32), matches=np.array(blocks, dtype=np.int32),
        mismatches=np.zeros(n, dtype=np.int32), qualities=np.full(n, 60, dtype=np.int32),
        cigars=np.array([f"{b}M" for b in blocks], dtype=object)
    )


def _records(rows, strands=None):
    batch = _batch(rows, strands)
    return [batch.get_record(i) for i in range(len(batch))]


def _graph(lengths, edges):
    return AssemblyGraph(edges, lengths)


# Past the contig end a flank walk leaves by (#66) ---------------------------------------------------------------------
A_TO_B = _graph({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgA", F, "ctgB", F, overlap=100)])


@pytest.mark.parametrize("strand, aln, side", [
    # A '+' IS aligns its first 300 of 500 bp up to 5 bp short of ctgA's end, so its last 200 bp run on into ctgB
    (F, (0, 300), DOWN),
    # Read backwards, its last 300 bp end there, so its first 200 bp run on into ctgB, upstream of it
    (R, (200, 500), UP),
])
def test_a_target_whose_alignment_runs_off_the_contig_runs_on_by_its_unaligned_bases(strand, aln, side):
    records = _records([(500, *aln, "ctgA", 1000, 695, 995)], [strand])

    assert beyond_contig_end(records, strand, side, A_TO_B) == 200


def test_a_target_whose_alignment_stops_inside_the_contig_runs_on_by_nothing():
    # 200 bp short of ctgA's end, more than the link's overlap and slack: that copy is truncated
    records = _records([(500, 0, 300, "ctgA", 1000, 500, 800)])

    assert beyond_contig_end(records, F, DOWN, A_TO_B) == 0


def test_a_target_whose_rest_would_fit_on_the_contig_runs_on_by_nothing():
    # 120 bp short of ctgA's end, within the link's overlap and slack, but only 20 bp of the target are unaligned: they
    # would lie on ctgA, where they do not align, so the copy is truncated
    records = _records([(500, 0, 480, "ctgA", 1000, 400, 880)])

    assert beyond_contig_end(records, F, DOWN, A_TO_B) == 0


@pytest.mark.parametrize("strand", [F, R])
def test_a_target_starting_at_its_contigs_start_runs_back_past_it_upstream(strand):
    # IS1 starts on ctgX at its position 200: on '+' its first 200 bp run back off ctgX's start onto ctgU; on '-', read
    # along ctgX-, off ctgX's end
    edges = [Edge("ctgU", F, "ctgX", F)] if strand == F else [Edge("ctgX", F, "ctgU", F)]
    records = _records([(1000, 200, 500, "ctgX", 300, 0, 300)], [strand])

    assert beyond_contig_end(records, strand, UP, _graph({"ctgU": 1000, "ctgX": 300}, edges)) == 200


def test_a_target_runs_on_past_the_contig_end_by_the_alignment_nearest_it():
    # Two near-identical targets on one copy: the one ending 5 bp short of ctgA's end leaves 200 bp, not the one 30 bp
    # short leaving 260
    records = _records([(500, 0, 240, "ctgA", 1000, 730, 970), (500, 0, 300, "ctgA", 1000, 695, 995)])

    assert beyond_contig_end(records, F, DOWN, A_TO_B) == 200


def test_a_synthetic_stitch_record_does_not_run_past_the_contig_end():
    [record] = _records([(500, 0, 300, "ctgA", 1000, 695, 995)])

    assert beyond_contig_end([dataclasses.replace(record, idx=-1)], F, DOWN, A_TO_B) == 0


# Carried past a hub branch's link (#65) -------------------------------------------------------------------------------
BRANCH_LENGTHS = {"b2": 1000, "b3": 1000, "a2": 5000}


def test_a_copy_entering_a_branch_carries_its_target_past_the_links_overlap():
    # b2 shares 100 bp with the hub; the copy runs to 200 on it
    alignments = {"b2": _batch([(1600, 1400, 1600, "b2", 1000, 0, 200)])}

    assert carried_past_link(Edge("hub", F, "b2", F, overlap=100), alignments, BRANCH_LENGTHS) == 100


@pytest.mark.parametrize("link, alignments", [
    # b3 is entered backwards, where its last 150 bp align
    (Edge("hub", F, "b3", R), {"b3": _batch([(1600, 1450, 1600, "b3", 1000, 850, 1000)], [R])}),
    # Upstream, a2 is entered backwards from the hub's start and ends with its copy's first 150 bp
    (Edge("hub", R, "a2", R), {"a2": _batch([(1600, 0, 150, "a2", 5000, 4850, 5000)])}),
])
def test_a_copy_entering_a_branch_read_backwards_carries_its_target_from_the_contigs_end(link, alignments):
    assert carried_past_link(link, alignments, BRANCH_LENGTHS) == 150


def test_an_alignment_starting_past_the_overlap_is_not_carried_past_the_link():
    # An alignment starting on b2 just where the 100 bp it shares with the hub ends is another IS beside this copy
    alignments = {"b2": _batch([(1200, 1050, 1200, "b2", 1000, 100, 250)])}

    assert carried_past_link(Edge("hub", F, "b2", F, overlap=100), alignments, BRANCH_LENGTHS) == 0


@pytest.mark.parametrize("start, carried", [(10, 200), (11, 0)])
def test_only_an_alignment_starting_within_the_entry_slack_is_carried_past_the_link(start, carried):
    alignments = {"b2": _batch([(1600, 1400, 1590 - start, "b2", 1000, start, 200)])}

    assert carried_past_link(Edge("hub", F, "b2", F), alignments, BRANCH_LENGTHS) == carried


def test_a_branch_with_no_alignments_carries_nothing():
    assert carried_past_link(Edge("hub", F, "b2", F, overlap=100), {}, BRANCH_LENGTHS) == 0
