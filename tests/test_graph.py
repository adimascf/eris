import pytest
import numpy as np

from eris.graph import AssemblyGraph, Edge, TopologyEngine
from eris.constants import Strand
from eris.alignment import AlignmentBatch

def test_topology_engine_bounded_paths():
    # Construct a mock DAG: ctgA -> ctgB -> ctgC
    edges = [
        Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=0),
        Edge("ctgB", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=0)
    ]
    contig_lengths = {"ctgA": 1000, "ctgB": 500, "ctgC": 1000}
    contig_depths = {"ctgA": 10.0, "ctgB": 2.0, "ctgC": 10.0}
    
    engine = TopologyEngine(AssemblyGraph(edges, contig_lengths), contig_depths)
    
    # Expected length across B is exactly 500bp
    paths = engine._find_bounded_paths(
        start_ctg="ctgA", start_strand=Strand.FORWARD,
        target_ctg="ctgC", target_strand=Strand.FORWARD,
        expected_len=500, tolerance=100
    )
    
    assert len(paths) == 1
    p = paths[0]
    assert p['contigs'] == ["ctgA", "ctgB", "ctgC"]
    assert p['length'] == 500  # The path walked across B entirely
    assert p['min_depth'] == 2.0  # Bottleneck depth was on ctgB

def test_topology_engine_collapsed_path():
    # Branching path ctgA -> ctgB1 (depth=1) -> ctgC
    #                ctgA -> ctgB2 (depth=10) -> ctgC
    edges = [
        Edge("ctgA", Strand.FORWARD, "ctgB1", Strand.FORWARD, overlap=0),
        Edge("ctgB1", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=0),
        Edge("ctgA", Strand.FORWARD, "ctgB2", Strand.FORWARD, overlap=0),
        Edge("ctgB2", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=0),
    ]
    contig_lengths = {"ctgA": 1000, "ctgB1": 500, "ctgB2": 500, "ctgC": 1000}
    contig_depths = {"ctgA": 10.0, "ctgB1": 1.0, "ctgB2": 10.0, "ctgC": 10.0}
    
    engine = TopologyEngine(AssemblyGraph(edges, contig_lengths), contig_depths)
    
    # The greedy path finder should pick B2 because it has the highest depth (10.0 > 1.0)
    paths = engine._find_collapsed_path(
        start_ctg="ctgA", start_strand=Strand.FORWARD,
        target_ctg="ctgC", target_strand=Strand.FORWARD,
        expected_len=500, tolerance=100
    )
    
    assert len(paths) == 1
    assert paths[0]['contigs'] == ["ctgA", "ctgB2", "ctgC"]
    assert paths[0]['min_depth'] == 10.0

def test_topology_engine_resolve():
    # Simulate a shattered alignment:
    # An IS element maps perfectly to the end of ctgA, and the beginning of ctgC
    edges = [
        Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=0),
        Edge("ctgB", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=0)
    ]
    contig_lengths = {"ctgA": 1000, "ctgB": 500, "ctgC": 1000}
    contig_depths = {"ctgA": 10.0, "ctgB": 10.0, "ctgC": 10.0}
    
    engine = TopologyEngine(AssemblyGraph(edges, contig_lengths), contig_depths, mode='variant')
    
    # Create two partial AlignmentRecords representing the shattered hit
    batch_a = AlignmentBatch(
        q_names=np.array(["IS1"]), q_lengths=np.array([2000], dtype=np.int32),
        q_starts=np.array([0], dtype=np.int32), q_ends=np.array([1000], dtype=np.int32),
        t_names=np.array(["ctgA"]), t_lengths=np.array([1000], dtype=np.int32),
        t_starts=np.array([0], dtype=np.int32), t_ends=np.array([1000], dtype=np.int32),
        strands=np.array([Strand.FORWARD], dtype=np.int8),
        lengths=np.array([1000], dtype=np.int32), matches=np.array([1000], dtype=np.int32),
        mismatches=np.array([0], dtype=np.int32), qualities=np.array([60], dtype=np.int32),
        cigars=np.array(["1000M"], dtype=object)
    )
    
    batch_c = AlignmentBatch(
        q_names=np.array(["IS1"]), q_lengths=np.array([2000], dtype=np.int32),
        q_starts=np.array([1500], dtype=np.int32), q_ends=np.array([2000], dtype=np.int32),
        t_names=np.array(["ctgC"]), t_lengths=np.array([1000], dtype=np.int32),
        t_starts=np.array([0], dtype=np.int32), t_ends=np.array([500], dtype=np.int32),
        strands=np.array([Strand.FORWARD], dtype=np.int8),
        lengths=np.array([500], dtype=np.int32), matches=np.array([500], dtype=np.int32),
        mismatches=np.array([0], dtype=np.int32), qualities=np.array([60], dtype=np.int32),
        cigars=np.array(["500M"], dtype=object)
    )
    
    alignments = {"ctgA": batch_a, "ctgC": batch_c}
    
    cleaned, paths = engine.resolve_split_alignments(alignments)
    
    # We should successfully stitch 1 multi-contig path over the gap
    assert len(paths) == 1
    stitched_path = paths[0]
    
    # The payload sequence should contain: [real_hit_A, synthetic_hit_B, real_hit_C]
    assert len(stitched_path) == 3
    assert stitched_path[0].t_name == "ctgA"
    assert stitched_path[1].t_name == "ctgB"
    assert stitched_path[1].idx == -1 # Verify it's a mock record for the unaligned node
    assert stitched_path[2].t_name == "ctgC"
    
    # Because both hits were stitched into the path, the 'cleaned' pool should be empty
    assert len(cleaned) == 0

def test_resolve_consumes_alignments_on_intermediate_contigs():
    # IS1 is split between the end of ctgA and the start of ctgC; the path runs through ctgB,
    # which only has an alignment of a cross-mapping relative (IS1_variant). Once ctgB is part
    # of the stitched path, its alignment must not also be reported as a local locus.
    edges = [
        Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=0),
        Edge("ctgB", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=0)
    ]
    contig_lengths = {"ctgA": 1000, "ctgB": 500, "ctgC": 1000}
    contig_depths = {"ctgA": 10.0, "ctgB": 10.0, "ctgC": 10.0}

    engine = TopologyEngine(AssemblyGraph(edges, contig_lengths), contig_depths, mode='collapse')

    def batch(q_name, q_length, q_start, q_end, t_name, t_length, t_start, t_end):
        length = q_end - q_start
        return AlignmentBatch(
            q_names=np.array([q_name]), q_lengths=np.array([q_length], dtype=np.int32),
            q_starts=np.array([q_start], dtype=np.int32), q_ends=np.array([q_end], dtype=np.int32),
            t_names=np.array([t_name]), t_lengths=np.array([t_length], dtype=np.int32),
            t_starts=np.array([t_start], dtype=np.int32), t_ends=np.array([t_end], dtype=np.int32),
            strands=np.array([Strand.FORWARD], dtype=np.int8),
            lengths=np.array([length], dtype=np.int32), matches=np.array([length], dtype=np.int32),
            mismatches=np.array([0], dtype=np.int32), qualities=np.array([60], dtype=np.int32),
            cigars=np.array([f"{length}M"], dtype=object)
        )

    alignments = {
        "ctgA": batch("IS1", 2000, 0, 1000, "ctgA", 1000, 0, 1000),
        "ctgB": batch("IS1_variant", 600, 50, 550, "ctgB", 500, 0, 500),
        "ctgC": batch("IS1", 2000, 1500, 2000, "ctgC", 1000, 0, 500),
    }

    cleaned, paths = engine.resolve_split_alignments(alignments)

    assert [f.t_name for f in paths[0]] == ["ctgA", "ctgB", "ctgC"]
    assert paths[0][1].idx == -1
    assert "ctgB" not in cleaned


def _fragment(q_name, q_length, q_start, q_end, t_name, t_length, t_start, t_end, strand=Strand.FORWARD):
    length = q_end - q_start
    return AlignmentBatch(
        q_names=np.array([q_name]), q_lengths=np.array([q_length], dtype=np.int32),
        q_starts=np.array([q_start], dtype=np.int32), q_ends=np.array([q_end], dtype=np.int32),
        t_names=np.array([t_name]), t_lengths=np.array([t_length], dtype=np.int32),
        t_starts=np.array([t_start], dtype=np.int32), t_ends=np.array([t_end], dtype=np.int32),
        strands=np.array([strand], dtype=np.int8),
        lengths=np.array([length], dtype=np.int32), matches=np.array([length], dtype=np.int32),
        mismatches=np.array([0], dtype=np.int32), qualities=np.array([60], dtype=np.int32),
        cigars=np.array([f"{length}M"], dtype=object)
    )


@pytest.mark.parametrize("mode", ["variant", "collapse"])
def test_resolve_chains_fragments_overlapping_by_edge_overlap(mode):
    # Contigs joined by 111 bp overlaps share 111 bp of the IS, so neighbouring fragments overlap on the query.
    # All three real fragments must be chained, not skipped over as synthetic nodes.
    edges = [
        Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=111),
        Edge("ctgB", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=111),
    ]
    contig_lengths = {"ctgA": 1000, "ctgB": 500, "ctgC": 5000}
    contig_depths = {"ctgA": 10.0, "ctgB": 10.0, "ctgC": 10.0}
    engine = TopologyEngine(AssemblyGraph(edges, contig_lengths), contig_depths, mode=mode)

    alignments = {
        "ctgA": _fragment("IS1", 1500, 0, 1000, "ctgA", 1000, 0, 1000),
        "ctgB": _fragment("IS1", 1500, 889, 1389, "ctgB", 500, 0, 500),
        "ctgC": _fragment("IS1", 1500, 1278, 1500, "ctgC", 5000, 0, 222),
    }

    cleaned, paths = engine.resolve_split_alignments(alignments)

    assert len(paths) == 1
    assert [(f.t_name, f.idx != -1) for f in paths[0]] == [("ctgA", True), ("ctgB", True), ("ctgC", True)]
    assert cleaned == {}


@pytest.mark.parametrize("mode", ["variant", "collapse"])
def test_resolve_follows_gfa_link_in_reverse_complement(mode):
    # ISEc23 on contigs_shovill_simulated.gfa: the IS starts on the reverse strand of ctgA and continues forward
    # on ctgB. The GFA only writes the link as 'ctgB - ctgA +'; its complement 'ctgA - ctgB +' is the same link.
    edges = [Edge("ctgB", Strand.REVERSE, "ctgA", Strand.FORWARD, overlap=111)]
    contig_lengths = {"ctgA": 1326, "ctgB": 5000}
    contig_depths = {"ctgA": 30.0, "ctgB": 30.0}
    engine = TopologyEngine(AssemblyGraph(edges, contig_lengths), contig_depths, mode=mode)

    alignments = {
        "ctgA": _fragment("IS1", 2532, 0, 1326, "ctgA", 1326, 0, 1326, Strand.REVERSE),
        "ctgB": _fragment("IS1", 2532, 1215, 2532, "ctgB", 5000, 0, 1317),
    }

    cleaned, paths = engine.resolve_split_alignments(alignments)

    assert [f.t_name for f in paths[0]] == ["ctgA", "ctgB"]
    assert cleaned == {}


def _fragments(rows, lengths):
    # rows: (q_name, q_length, q_start, q_end, t_name, t_start, t_end, strand); one batch per contig
    by_contig = {}
    for row in rows:
        by_contig.setdefault(row[4], []).append(row)
    batches = {}
    for contig, rs in by_contig.items():
        n = np.array([r[3] - r[2] for r in rs], dtype=np.int32)
        batches[contig] = AlignmentBatch(
            q_names=np.array([r[0] for r in rs]), q_lengths=np.array([r[1] for r in rs], dtype=np.int32),
            q_starts=np.array([r[2] for r in rs], dtype=np.int32), q_ends=np.array([r[3] for r in rs], dtype=np.int32),
            t_names=np.array([contig] * len(rs)), t_lengths=np.full(len(rs), lengths[contig], dtype=np.int32),
            t_starts=np.array([r[5] for r in rs], dtype=np.int32), t_ends=np.array([r[6] for r in rs], dtype=np.int32),
            strands=np.array([r[7] for r in rs], dtype=np.int8), lengths=n, matches=n,
            mismatches=np.zeros(len(rs), dtype=np.int32), qualities=np.full(len(rs), 60, dtype=np.int32),
            cigars=np.array([f"{k}M" for k in n], dtype=object)
        )
    return batches


@pytest.mark.parametrize("mode", ["variant", "collapse"])
def test_resolve_chains_the_is_body_not_both_ends_of_one_long_contig(mode):
    # ISEc8 on contigs_shovill_simulated.gfa: ctgLong (20.7 kb) has short IS alignments at both ends, 20 kb apart, while
    # the IS body is ctgBody. The two ends of ctgLong are 697 bp apart on the query, so they cannot be one copy.
    R, F = Strand.REVERSE, Strand.FORWARD
    edges = [
        Edge("ctgUp", R, "ctgBody", R, overlap=111), Edge("ctgUp", R, "ctgLong", R, overlap=111),
        Edge("ctgBody", R, "ctgDown", F, overlap=111), Edge("ctgLong", R, "ctgDown", F, overlap=111),
    ]
    contig_lengths = {"ctgUp": 159, "ctgLong": 20707, "ctgBody": 965, "ctgDown": 364}
    contig_depths = {"ctgUp": 65.1, "ctgLong": 33.7, "ctgBody": 33.4, "ctgDown": 70.3}
    engine = TopologyEngine(AssemblyGraph(edges, contig_lengths), contig_depths, mode=mode)

    alignments = _fragments([
        ("IS1", 2533, 0, 159, "ctgUp", 0, 159, R),
        ("IS1", 2533, 48, 168, "ctgLong", 20587, 20707, R),
        ("IS1", 2533, 48, 1013, "ctgBody", 0, 965, R),
        ("IS1", 2533, 865, 1013, "ctgLong", 0, 148, R),
        ("IS1", 2533, 902, 1266, "ctgDown", 0, 364, F),
    ], contig_lengths)

    _, paths = engine.resolve_split_alignments(alignments)

    assert [f.t_name for f in paths[0]] == ["ctgUp", "ctgBody", "ctgDown"]


@pytest.mark.parametrize("mode", ["variant", "collapse"])
def test_resolve_does_not_stitch_fragments_on_one_contig(mode):
    # EC958 chromosome: IS21's inverted repeats hit both ends of an IS100kyp copy (IS21 family), 2042 bp apart,
    # about the IS21 query gap. The body between them is IS100kyp, not IS21, so the ends must stay unstitched.
    contig_lengths = {"chr": 5000000}
    engine = TopologyEngine(AssemblyGraph([], contig_lengths), {"chr": 1.0}, mode=mode)
    alignments = _fragments([
        ("IS21", 2131, 0, 130, "chr", 2252235, 2252365, Strand.FORWARD),
        ("IS21", 2131, 1963, 2131, "chr", 2254407, 2254575, Strand.FORWARD),
    ], contig_lengths)

    cleaned, paths = engine.resolve_split_alignments(alignments)

    assert paths == []
    assert len(cleaned["chr"]) == 2


def _stitched_paths(mode, contigs, edges, rows, depths=None):
    """The contig names of each path stitched from rows (see _fragments) on contigs of {name: length}."""
    engine = TopologyEngine(AssemblyGraph(edges, contigs), depths or {c: 10.0 for c in contigs}, mode=mode)
    return [[f.t_name for f in p] for p in engine.resolve_split_alignments(_fragments(rows, contigs))[1]]


@pytest.mark.parametrize("mode", ["variant", "collapse"])
@pytest.mark.parametrize("a_strand, a_start, a_end, stitched", [
    # Read forward, the path leaves ctgA by its end: a fragment at its start leaves 4.6 kb of ctgA uncounted
    (Strand.FORWARD, 0, 400, False),
    (Strand.FORWARD, 4600, 5000, True),
    # Read backwards, the path leaves ctgA by its start
    (Strand.REVERSE, 4600, 5000, False),
    (Strand.REVERSE, 0, 400, True),
])
def test_resolve_stitches_only_a_fragment_hanging_off_the_contig_end_the_path_leaves_by(
        mode, a_strand, a_start, a_end, stitched):
    paths = _stitched_paths(mode, {"ctgA": 5000, "ctgB": 5000}, [Edge("ctgA", a_strand, "ctgB", Strand.FORWARD)], [
        ("IS1", 1000, 0, 400, "ctgA", a_start, a_end, a_strand),
        ("IS1", 1000, 400, 1000, "ctgB", 0, 600, Strand.FORWARD),
    ])

    assert paths == ([["ctgA", "ctgB"]] if stitched else [])


@pytest.mark.parametrize("mode", ["variant", "collapse"])
@pytest.mark.parametrize("b_strand, b_start, b_end, stitched", [
    # Read forward, the path enters ctgB by its start; read backwards, by its end
    (Strand.FORWARD, 4400, 5000, False),
    (Strand.FORWARD, 0, 600, True),
    (Strand.REVERSE, 0, 600, False),
    (Strand.REVERSE, 4400, 5000, True),
])
def test_resolve_stitches_only_a_fragment_hanging_off_the_contig_end_the_path_enters_by(
        mode, b_strand, b_start, b_end, stitched):
    paths = _stitched_paths(mode, {"ctgA": 5000, "ctgB": 5000}, [Edge("ctgA", Strand.FORWARD, "ctgB", b_strand)], [
        ("IS1", 1000, 0, 400, "ctgA", 4600, 5000, Strand.FORWARD),
        ("IS1", 1000, 400, 1000, "ctgB", b_start, b_end, b_strand),
    ])

    assert paths == ([["ctgA", "ctgB"]] if stitched else [])


@pytest.mark.parametrize("mode", ["variant", "collapse"])
def test_resolve_allows_a_fragment_to_stop_short_of_the_contig_end_by_the_link_overlap_and_slack(mode):
    # 111 bp overlap + 50 bp slack: a fragment ending 161 bp from ctgA's end hangs off it, 162 bp does not
    def stitched(gap):
        return _stitched_paths(mode, {"ctgA": 5000, "ctgB": 5000},
                               [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=111)], [
            ("IS1", 1000, 0, 400, "ctgA", 4600 - gap, 5000 - gap, Strand.FORWARD),
            ("IS1", 1000, 289, 1000, "ctgB", 0, 711, Strand.FORWARD),
        ])

    assert stitched(161) and not stitched(162)


@pytest.mark.parametrize("mode", ["variant", "collapse"])
@pytest.mark.parametrize("mid_end, chain", [
    # A fragment in the middle of a chain is entered and left, so must cover its contig: one covering only ctgM's
    # start ends the chain there, as ctgM's other 1.7 kb would otherwise land inside the locus uncounted
    (300, ["ctgA", "ctgM"]),
    (2000, ["ctgA", "ctgM", "ctgB"]),
])
def test_resolve_chains_through_a_middle_fragment_only_if_it_covers_its_contig(mode, mid_end, chain):
    contigs = {"ctgA": 5000, "ctgM": 2000, "ctgB": 5000}
    edges = [Edge("ctgA", Strand.FORWARD, "ctgM", Strand.FORWARD), Edge("ctgM", Strand.FORWARD, "ctgB", Strand.FORWARD)]
    paths = _stitched_paths(mode, contigs, edges, [
        ("IS1", 3000, 0, 500, "ctgA", 4500, 5000, Strand.FORWARD),
        ("IS1", 3000, 500, 500 + mid_end, "ctgM", 0, mid_end, Strand.FORWARD),
        ("IS1", 3000, 500 + mid_end, 3000, "ctgB", 0, 2500 - mid_end, Strand.FORWARD),
    ])

    assert paths == [chain]


def _mixed_overlap_graph():
    """ctgA+ -> ctgB+ overlap by 20 bp; elsewhere in the graph ctgC+ -> ctgD+ overlap by 200 bp."""
    contigs = {"ctgA": 5000, "ctgB": 5000, "ctgC": 5000, "ctgD": 5000}
    edges = [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=20),
             Edge("ctgC", Strand.FORWARD, "ctgD", Strand.FORWARD, overlap=200)]
    return contigs, edges


@pytest.mark.parametrize("mode", ["variant", "collapse"])
@pytest.mark.parametrize("query_overlap, stitched", [(70, True), (71, False)])
def test_resolve_lets_fragments_overlap_on_the_query_by_the_crossed_link_overlap_not_the_graph_largest(
        mode, query_overlap, stitched):
    # ctgA and ctgB share 20 bp, so their fragments may overlap on the IS by 20 + 50 bp; the 200 bp link elsewhere
    # would allow 250
    contigs, edges = _mixed_overlap_graph()
    paths = _stitched_paths(mode, contigs, edges, [
        ("IS1", 1000, 0, 400, "ctgA", 4600, 5000, Strand.FORWARD),
        ("IS1", 1000, 400 - query_overlap, 1000, "ctgB", 0, 600 + query_overlap, Strand.FORWARD),
    ])

    assert paths == ([["ctgA", "ctgB"]] if stitched else [])


@pytest.mark.parametrize("mode", ["variant", "collapse"])
@pytest.mark.parametrize("a_gap, b_gap, stitched", [
    (70, 0, True), (71, 0, False),  # ctgA's fragment stops short of the end the path leaves by
    (0, 70, True), (0, 71, False),  # ctgB's fragment starts in from the end the path enters by
])
def test_resolve_lets_a_fragment_stop_short_of_its_contig_end_by_the_crossed_link_overlap_not_the_graph_largest(
        mode, a_gap, b_gap, stitched):
    contigs, edges = _mixed_overlap_graph()
    paths = _stitched_paths(mode, contigs, edges, [
        ("IS1", 1000, 0, 400, "ctgA", 4600 - a_gap, 5000 - a_gap, Strand.FORWARD),
        ("IS1", 1000, 380, 1000, "ctgB", b_gap, 620 + b_gap, Strand.FORWARD),
    ])

    assert paths == ([["ctgA", "ctgB"]] if stitched else [])


def test_collapse_mode_backs_up_past_a_deeper_path_whose_links_reject_the_fragments():
    # Both routes from ctgA to ctgB fit the length; the deeper one, through ctgX, ends in a 20 bp link that ctgB's
    # fragment (starting 100 bp in) does not reach, while the shallower one, through ctgY, ends in a 200 bp link
    contigs = {"ctgA": 5000, "ctgX": 500, "ctgY": 680, "ctgB": 5000}
    edges = [Edge("ctgA", Strand.FORWARD, "ctgX", Strand.FORWARD, overlap=20),
             Edge("ctgX", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=20),
             Edge("ctgA", Strand.FORWARD, "ctgY", Strand.FORWARD, overlap=20),
             Edge("ctgY", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=200)]
    paths = _stitched_paths('collapse', contigs, edges, [
        ("IS1", 2000, 0, 500, "ctgA", 4500, 5000, Strand.FORWARD),
        ("IS1", 2000, 1000, 2000, "ctgB", 100, 1100, Strand.FORWARD),
    ], depths={"ctgA": 10.0, "ctgX": 20.0, "ctgY": 15.0, "ctgB": 10.0})

    assert paths == [["ctgA", "ctgY", "ctgB"]]


@pytest.mark.parametrize("mode", ["variant", "collapse"])
@pytest.mark.parametrize("query_overlap, stitched", [(122, True), (123, False)])
def test_resolve_lets_fragments_overlap_on_the_query_by_as_much_as_the_contig_path_overlaps_itself(
        mode, query_overlap, stitched):
    # ctgM (150 bp) shares 111 bp with ctgA and with ctgB, so ctgA's end and ctgB's start overlap by 72 bp: the
    # fragments may overlap on the IS by 72 + 50 bp, short of the 111 + 50 either link alone would allow
    contigs = {"ctgA": 5000, "ctgM": 150, "ctgB": 5000}
    edges = [Edge("ctgA", Strand.FORWARD, "ctgM", Strand.FORWARD, overlap=111),
             Edge("ctgM", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=111)]
    paths = _stitched_paths(mode, contigs, edges, [
        ("IS1", 1000, 0, 400, "ctgA", 4600, 5000, Strand.FORWARD),
        ("IS1", 1000, 400 - query_overlap, 1000, "ctgB", 0, 600 + query_overlap, Strand.FORWARD),
    ])

    assert paths == ([["ctgA", "ctgM", "ctgB"]] if stitched else [])


def test_links_from_a_contig_strand_come_in_the_order_read_with_each_links_complement():
    # 'ctgB+ -> ctgA+' leaves ctgA by its start as its complement, 'ctgA- -> ctgB-'. Order decides which flanks are
    # reported: a set's order depends on the string hash seed, which made flanks change between runs
    edges = [Edge("ctgA", Strand.FORWARD, "ctgC", Strand.FORWARD), Edge("ctgB", Strand.FORWARD, "ctgA", Strand.FORWARD),
             Edge("ctgA", Strand.FORWARD, "ctgD", Strand.REVERSE)]
    edges += [Edge("ctgA", Strand.FORWARD, f"ctg{i}", Strand.FORWARD) for i in range(10)]
    graph = AssemblyGraph(edges, {c: 1000 for e in edges for c in (e.u, e.v)})

    assert graph.links_from("ctgA", Strand.FORWARD) == [edges[0], edges[2], *edges[3:]]
    assert graph.links_from("ctgA", Strand.REVERSE) == [edges[1].complement()]
    assert graph.links_from("ctg3", Strand.REVERSE) == [edges[6].complement()]


def test_a_links_overlap_is_found_on_either_strand_and_is_0_without_a_link():
    graph = AssemblyGraph([Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, 55)], {"ctgA": 1000, "ctgB": 1000})

    assert graph.overlap("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE) == 55
    assert graph.overlap("ctgB", Strand.FORWARD, "ctgA", Strand.REVERSE) == 55
    assert graph.overlap("ctgB", Strand.FORWARD, "ctgA", Strand.FORWARD) == 0


@pytest.mark.parametrize("mode", ["variant", "collapse"])
def test_synthetic_node_carries_the_strand_its_contig_is_entered_on(mode):
    # IS1 runs off ctgA, through ctgB read backwards ('ctgA+ -> ctgB-'), and into ctgC
    edges = [
        Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, overlap=0),
        Edge("ctgB", Strand.REVERSE, "ctgC", Strand.FORWARD, overlap=0),
    ]
    contig_lengths = {"ctgA": 1000, "ctgB": 500, "ctgC": 1000}
    contig_depths = {"ctgA": 10.0, "ctgB": 10.0, "ctgC": 10.0}
    engine = TopologyEngine(AssemblyGraph(edges, contig_lengths), contig_depths, mode=mode)

    alignments = {
        "ctgA": _fragment("IS1", 2000, 0, 1000, "ctgA", 1000, 0, 1000),
        "ctgC": _fragment("IS1", 2000, 1500, 2000, "ctgC", 1000, 0, 500),
    }

    _, paths = engine.resolve_split_alignments(alignments)

    assert [(f.t_name, Strand(f.strand)) for f in paths[0]] == [
        ("ctgA", Strand.FORWARD), ("ctgB", Strand.REVERSE), ("ctgC", Strand.FORWARD)]


def test_resolve_chains_a_tangle_of_fragments_in_polynomial_searches(monkeypatch):
    # A target shredded across a ladder of bubbles: each layer has two parallel contigs carrying the same stretch of
    # the target, and every contig links to both of the next layer's. There are 2**layers chains through it, so a
    # search that enumerates chains never finishes on a real tangle (BPH2947, 79 fragments of one IS1182-family IS).
    layers, step = 12, 100
    contigs = [[f"L{layer}{side}" for side in "ab"] for layer in range(layers)]
    edges = [Edge(a, Strand.FORWARD, b, Strand.FORWARD, overlap=0)
             for left, right in zip(contigs, contigs[1:]) for a in left for b in right]
    contig_lengths = {c: step for layer in contigs for c in layer}
    engine = TopologyEngine(AssemblyGraph(edges, contig_lengths), dict.fromkeys(contig_lengths, 10.0), mode='variant')
    alignments = {c: _fragment("IS1", layers * step, i * step, (i + 1) * step, c, step, 0, step)
                  for i, layer in enumerate(contigs) for c in layer}

    searches = 0
    find = TopologyEngine._find_bounded_paths
    def counting(self, *args, **kwargs):
        nonlocal searches
        searches += 1
        return find(self, *args, **kwargs)
    monkeypatch.setattr(TopologyEngine, '_find_bounded_paths', counting)

    cleaned, paths = engine.resolve_split_alignments(alignments)

    n = len(alignments)
    assert searches <= n * n
    # Both strands of the ladder are stitched end to end, through one real fragment per layer
    assert len(paths) == 2
    for path in paths:
        assert (path[0].q_start, path[-1].q_end) == (0, layers * step)
        assert all(f.idx != -1 for f in path) and len(path) == layers
    assert not cleaned
