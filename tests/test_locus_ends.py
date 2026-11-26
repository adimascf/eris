import dataclasses

import numpy as np
import pytest

from eris.alignment import AlignmentBatch
from eris.constants import Strand
from eris.flanks import Branch
from eris.graph import Edge, AssemblyGraph
from eris.hub import Hubs
from eris.locus_ends import LocusEnds

F, R = Strand.FORWARD, Strand.REVERSE


def _batch(rows, strands=None, q_name="IS1"):
    """rows: (q_length, q_start, q_end, t_name, t_length, t_start, t_end) of one alignment each; '+' by default."""
    ql, qs, qe, tn, tl, ts, te = zip(*rows)
    n = len(rows)
    blocks = [e - s for s, e in zip(qs, qe)]
    return AlignmentBatch(
        q_names=np.array([q_name] * n, dtype=object), q_lengths=np.array(ql, dtype=np.int32),
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


def _locus_ends(lengths, edges, alignments=None, hub_branches=True):
    graph = AssemblyGraph(edges, lengths)
    return LocusEnds(graph, Hubs(graph), alignments or {}, hub_branches)


# A local locus: where its target runs past the contig end (#66) ------------------------------------------------------
A_TO_B = _locus_ends({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgA", F, "ctgB", F, overlap=100)])


def test_a_local_locus_spans_its_copys_alignments_read_on_its_first_targets_strand():
    records = _records([(500, 0, 300, "ctgA", 1000, 400, 700), (500, 0, 450, "ctgA", 1000, 350, 650)], [R, F])

    ends = A_TO_B.local(records)

    assert ends.upstream is ends.downstream
    assert (ends.upstream.contig, ends.upstream.start, ends.upstream.end, ends.upstream.strand) == ("ctgA", 350, 700, R)
    assert ends.branches is None


@pytest.mark.parametrize("strand, aln", [
    # A '+' IS aligns its first 300 of 500 bp up to 5 bp short of ctgA's end, so its last 200 bp run on into ctgB
    (F, (0, 300)),
    # Read backwards, its last 300 bp end there, so its first 200 bp run on into ctgB, upstream of it
    (R, (200, 500)),
])
def test_a_target_whose_alignment_runs_off_the_contig_runs_on_by_its_unaligned_bases(strand, aln):
    span = A_TO_B.local(_records([(500, *aln, "ctgA", 1000, 695, 995)], [strand])).upstream

    beyond = span.beyond_downstream if strand == F else span.beyond_upstream
    assert beyond == 200


def test_a_target_whose_alignment_stops_inside_the_contig_runs_on_by_nothing():
    # 200 bp short of ctgA's end, more than the link's overlap and slack: that copy is truncated
    span = A_TO_B.local(_records([(500, 0, 300, "ctgA", 1000, 500, 800)])).upstream

    assert (span.beyond_upstream, span.beyond_downstream) == (0, 0)


def test_a_target_whose_rest_would_fit_on_the_contig_runs_on_by_nothing():
    # 120 bp short of ctgA's end, within the link's overlap and slack, but only 20 bp of the target are unaligned: they
    # would lie on ctgA, where they do not align, so the copy is truncated
    span = A_TO_B.local(_records([(500, 0, 480, "ctgA", 1000, 400, 880)])).upstream

    assert span.beyond_downstream == 0


@pytest.mark.parametrize("strand", [F, R])
def test_a_target_starting_at_its_contigs_start_runs_back_past_it_upstream(strand):
    # IS1 starts on ctgX at its position 200: on '+' its first 200 bp run back off ctgX's start onto ctgU; on '-', read
    # along ctgX-, off ctgX's end
    edges = [Edge("ctgU", F, "ctgX", F)] if strand == F else [Edge("ctgX", F, "ctgU", F)]
    records = _records([(1000, 200, 500, "ctgX", 300, 0, 300)], [strand])

    ends = _locus_ends({"ctgU": 1000, "ctgX": 300}, edges).local(records)

    assert ends.upstream.beyond_upstream == 200


def test_a_target_runs_on_past_the_contig_end_by_the_alignment_nearest_it():
    # Two near-identical targets on one copy: the one ending 5 bp short of ctgA's end leaves 200 bp, not the one 30 bp
    # short leaving 260
    records = _records([(500, 0, 240, "ctgA", 1000, 730, 970), (500, 0, 300, "ctgA", 1000, 695, 995)])

    assert A_TO_B.local(records).downstream.beyond_downstream == 200


# A stitched locus -----------------------------------------------------------------------------------------------------
def test_a_stitched_locus_runs_on_upstream_from_its_first_fragment_and_downstream_from_its_last():
    # IS1 runs off ctgA's end into ctgB; past ctgB's start upstream and ctgA's end downstream nothing is left over
    first, last = _records([(1000, 300, 600, "ctgB", 1000, 0, 300), (1000, 0, 300, "ctgA", 1000, 695, 995)])
    edges = [Edge("ctgU", F, "ctgB", F), Edge("ctgA", F, "ctgD", F)]

    ends = _locus_ends({"ctgU": 1000, "ctgB": 1000, "ctgA": 1000, "ctgD": 1000}, edges).stitched([first, last])

    assert (ends.upstream.contig, ends.upstream.beyond_upstream, ends.upstream.beyond_downstream) == ("ctgB", 300, 0)
    assert (ends.downstream.contig, ends.downstream.beyond_upstream, ends.downstream.beyond_downstream) == (
        "ctgA", 0, 700)


def test_a_synthetic_stitch_record_does_not_run_past_the_contig_end():
    [record] = _records([(500, 0, 300, "ctgA", 1000, 695, 995)])

    assert A_TO_B.stitched([dataclasses.replace(record, idx=-1)]).downstream.beyond_downstream == 0


# A repeat hub's branches (#56, #65) -----------------------------------------------------------------------------------
HUB_LENGTHS = {"hub": 1000, "a1": 5000, "a2": 5000, "b1": 1000, "b2": 1000, "b3": 1000}
HUB_EDGES = [Edge("a1", F, "hub", F), Edge("a2", F, "hub", F), Edge("hub", F, "b1", F),
             Edge("hub", F, "b2", F, overlap=100), Edge("hub", F, "b3", R)]


def _hub_path():
    """IS1 runs off the end of a1, across the unaligned 1 kb 'hub', and into the start of b1."""
    first, last = _records([(1600, 0, 300, "a1", 5000, 4700, 5000), (1600, 1300, 1600, "b1", 1000, 0, 300)])
    hub = dataclasses.replace(first, idx=-1, t_name="hub", t_length=1000, t_start=0, t_end=1000)
    return [first, hub, last]


def _hub_branches(alignments=None, extra_edges=(), hub_branches=True):
    locus_ends = _locus_ends(HUB_LENGTHS, HUB_EDGES + list(extra_edges), alignments, hub_branches)
    return locus_ends.stitched(_hub_path()).branches


def test_a_hub_has_a_branch_for_every_other_link_leaving_its_outermost_contigs():
    # The walk from b1 and a1, the path's own ends, already follows their links
    branches = _hub_branches()

    assert branches.upstream == (Branch(Edge("hub", R, "a2", R)),)
    assert branches.downstream == (Branch(Edge("hub", F, "b2", F, overlap=100)), Branch(Edge("hub", F, "b3", R)))
    assert branches.blocked == {"a1", "hub", "b1"}


def test_variant_mode_has_no_hub_branches():
    assert _hub_branches(hub_branches=False) is None


def test_a_stitched_path_with_no_hub_has_no_branches():
    # IS1 runs off ctgA's end straight into ctgB: no contig is covered end to end, so there is no hub
    fragments = _records([(1000, 0, 300, "ctgA", 1000, 700, 1000), (1000, 300, 1000, "ctgB", 1000, 0, 700)])

    assert A_TO_B.stitched(fragments).branches is None


def test_a_link_leading_back_into_the_hub_is_not_a_branch():
    # The hairpin 'hub+ -> hub-' leaves the hub's end but turns back into it
    branches = _hub_branches(extra_edges=[Edge("hub", F, "hub", R)])

    assert [b.link.v for b in branches.downstream] == ["b2", "b3"]


def test_a_link_out_of_the_middle_of_a_hub_is_not_a_branch():
    # IS1 runs off a1, across the two-contig hub h1 -> h2, and into b1. 'h2+ -> b2+' leaves the hub's edge, but
    # 'h1+ -> stub+' leaves its middle, into a stub of one copy's variant of the IS
    lengths = {"a1": 5000, "h1": 500, "h2": 500, "b1": 1000, "b2": 1000, "stub": 1000}
    edges = [Edge("a1", F, "h1", F), Edge("h1", F, "h2", F), Edge("h2", F, "b1", F), Edge("h2", F, "b2", F),
             Edge("h1", F, "stub", F)]
    first, _, last = _hub_path()
    h1, h2 = (dataclasses.replace(first, idx=-1, t_name=c, t_length=500, t_start=0, t_end=500) for c in ("h1", "h2"))

    branches = _locus_ends(lengths, edges).stitched([first, h1, h2, last]).branches

    assert branches.upstream == () and [b.link.v for b in branches.downstream] == ["b2"]


def test_a_path_ending_on_the_hubs_edge_has_no_branch_on_that_side():
    # The path's last fragment is the hub itself, so the walk from it already leaves by every link downstream
    first, hub, _ = _hub_path()

    branches = _locus_ends(HUB_LENGTHS, HUB_EDGES).stitched([first, hub]).branches

    assert [b.link.v for b in branches.upstream] == ["a2"] and branches.downstream == ()


def test_a_copy_carries_its_target_onto_each_branch_past_the_links_overlap():
    # Another copy's target runs past the hub: onto b2 to 200, 100 bp past the 100 bp it shares with the hub; onto b3,
    # entered backwards, by its last 150 bp; and upstream onto a2, entered backwards from the hub's start, by its first
    alignments = {
        "b2": _batch([(1600, 1400, 1600, "b2", 1000, 0, 200)]),
        "b3": _batch([(1600, 1450, 1600, "b3", 1000, 850, 1000)], [R]),
        "a2": _batch([(1600, 0, 150, "a2", 5000, 4850, 5000)]),
    }

    branches = _hub_branches(alignments)

    assert [b.carried for b in branches.downstream] == [100, 150]
    assert [b.carried for b in branches.upstream] == [150]


def test_an_alignment_starting_past_the_overlap_is_not_carried_onto_the_branch():
    # An alignment starting on b2 just where the 100 bp it shares with the hub ends is another IS beside this copy
    alignments = {"b2": _batch([(1200, 1050, 1200, "b2", 1000, 100, 250)])}

    assert _hub_branches(alignments).downstream[0].carried == 0


@pytest.mark.parametrize("start, carried", [(10, 200), (11, 0)])
def test_only_an_alignment_starting_within_the_entry_slack_is_carried_onto_the_branch(start, carried):
    # b2 is entered forwards at its start, b3 backwards at its end: an alignment `start` bp in starts that far in
    alignments = {"b2": _batch([(1600, 1400, 1590 - start, "b2", 1000, start, 300)]),
                  "b3": _batch([(1600, 1400, 1590 - start, "b3", 1000, 800, 1000 - start)], [R])}

    assert [b.carried for b in _hub_branches(alignments).downstream] == [carried, carried]
