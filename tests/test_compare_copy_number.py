import importlib.util
from pathlib import Path

from eris.graph import Edge
from eris.interval import Strand

# The comparison is a spike script, not part of the eris package
_SCRIPT = Path(__file__).parent.parent / "scripts" / "compare_copy_number.py"
_spec = importlib.util.spec_from_file_location("compare_copy_number", _SCRIPT)
cn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cn)


def test_depth_tiers_keep_a_chromosome_gradient_together_and_split_off_the_plasmid():
    # EC958 IS1 hub anchors: chromosome at 30-47x along its 1.7-fold gradient, pEC958 at 75-79x
    assert cn.depth_tiers([77, 30, 44, 75, 47, 32, 79], tolerance=1.4) == [[30, 32, 44, 47], [75, 77, 79]]


F, R = Strand.FORWARD, Strand.REVERSE


def walk(edges, hub, lengths, repeats=frozenset()):
    return cn.find_exits(cn.link_graph(edges), hub, lengths, repeats, min_anchor=1000, stub_budget=500)


def test_find_exits_crosses_short_stubs_to_the_first_anchor_and_follows_edge_orientation():
    lengths = {"hub": 800, "stub": 300, "A": 5000, "beyond": 9000, "leak": 9000, "U": 4000}
    edges = [
        Edge("hub", F, "stub", F), Edge("stub", F, "A", F),
        Edge("A", F, "beyond", F),  # past the anchor: never reached
        Edge("leak", F, "stub", F),  # joins the stub from the hub's side, as the rRNA leak does: never reached
        Edge("U", F, "hub", F),
    ]
    exits = walk(edges, [("hub", F)], lengths)
    assert exits == {"downstream": [cn.Exit("A", F, "anchor")], "upstream": [cn.Exit("U", R, "anchor")]}


def test_find_exits_counts_an_anchor_between_two_inverted_copies_once_per_link():
    # Two copies collapsed into one hub, inverted around anchor A (IS26 around 2419936): A is reached from both ends
    lengths = {"hub": 800, "A": 11000, "B": 3000}
    edges = [Edge("hub", F, "A", F), Edge("A", F, "hub", R), Edge("B", F, "hub", F)]
    exits = walk(edges, [("hub", F)], lengths)
    assert sorted(exits["downstream"]) == sorted([cn.Exit("A", F, "anchor"), cn.Exit("A", R, "anchor")])


def test_find_exits_stops_at_another_targets_hub_and_at_dead_ends():
    # One IS26 flank runs into the ISEc9 hub after a 140 bp stub; another breaks into stubs longer than the budget
    lengths = {"hub": 800, "stub": 140, "ISEc9": 1024, "s1": 300, "s2": 300, "far": 5000}
    edges = [Edge("hub", F, "stub", F), Edge("stub", F, "ISEc9", F),
             Edge("s1", F, "hub", F), Edge("s2", F, "s1", F), Edge("far", F, "s2", F)]
    exits = walk(edges, [("hub", F)], lengths, repeats={"ISEc9"})
    assert exits == {"downstream": [cn.Exit("ISEc9", F, "repeat")], "upstream": [cn.Exit("s2", R, "dead_end")]}


def anchors(*depths):
    return [cn.Exit(f"a{i}_{d}", F, "anchor") for i, d in enumerate(depths)], {f"a{i}_{d}": d for i, d in enumerate(depths)}


def estimate(hub_depth, upstream, downstream, baseline=39.0):
    (up, up_depths), (down, down_depths) = upstream, downstream
    return cn.estimate(hub_depth, {"upstream": up, "downstream": down}, {**up_depths, **down_depths}, baseline)


def test_estimate_splits_a_hub_into_depth_tiers_by_its_anchors():
    # IS1 hub on EC958: 3 chromosome copies at ~37x and 2 pEC958 copies at ~76x, 3 x 37 + 2 x 76 = 263
    est = estimate(272, anchors(30, 42, 44, 75, 77), anchors(32, 44, 47, 78, 79))
    assert (est.link, est.anchor, est.low, est.high) == (5, 5, 5, 5)
    assert [copies for copies, _ in est.tiers] == [3, 2]


def test_estimate_reports_a_range_when_stubs_hide_links():
    # ISEc12: 4 chromosome copies at 153x, but one side breaks into stubs, so only 2 + 4 links reach anchors
    up_exits, up_depths = anchors(30, 39)
    up_exits += [cn.Exit("stub1", F, "dead_end"), cn.Exit("stub2", F, "dead_end")]
    est = estimate(153, (up_exits, up_depths), anchors(30, 29, 41, 45))
    assert (est.link, est.anchor, est.low, est.high, est.fallback) == (3, 4, 3, 4, False)


def test_estimate_falls_back_to_the_genome_baseline_without_anchors():
    est = estimate(153, ([cn.Exit("s", F, "dead_end")], {}), ([], {}), baseline=37.4)
    assert (est.anchor, est.fallback) == (4, True)


def test_group_by_hub_scores_each_hub_once_per_family():
    # Issue #14: IS26 and IS15 both walk through the IS6 hub; IS15 reads one hub contig on the other strand
    loci = [
        cn.LocusHub("l1", "IS6", "IS26", [("h1", F), ("h2", F)], 14),
        cn.LocusHub("l2", "IS6", "IS15", [("h2", R), ("h3", F)], 13),
        cn.LocusHub("l3", "IS1", "IS1A", [("h1", F)], 6),  # another family through a shared contig
        cn.LocusHub("l4", "IS6", "IS6100", [], 2),  # local loci in unique sequence have no hub
        cn.LocusHub("l5", "IS6", "IS6100", [], 1),
    ]
    groups = cn.group_by_hub(loci)
    assert [(g.family, g.names, g.locus_ids, g.hub, g.old) for g in groups] == [
        ("IS6", {"IS26", "IS15"}, ["l1", "l2"], [("h1", F), ("h2", F), ("h3", F)], 14),
        ("IS1", {"IS1A"}, ["l3"], [("h1", F)], 6),
        ("IS6", {"IS6100"}, ["l4"], [], 2),
        ("IS6", {"IS6100"}, ["l5"], [], 1),
    ]


def test_hub_contigs_are_those_the_target_covers_end_to_end():
    # IS15 on the real-read assembly: in from an anchor, twice round the hub, out to another anchor (71 bp overlaps)
    lengths = {"in": 12413, "arm": 72, "h1": 114, "h2": 551, "out": 11220}
    segments = [("in", 12343, 12413, F), ("arm", 0, 72, R), ("h1", 0, 114, F), ("h2", 0, 551, F),
                ("h1", 0, 114, F), ("h2", 0, 480, F), ("out", 0, 70, F)]
    assert cn.hub_contigs(segments, lengths, overlap=71) == [("arm", R), ("h1", F), ("h2", F)]
