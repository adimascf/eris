import numpy as np
import pytest

from eris.constants import Context, Strand
from eris.flanks import Branch, BudgetRule, FlankSearch, HubBranches, TargetSpan
from eris.graph import AssemblyGraph, Edge, circular_contigs
from eris.interval import IntervalBatch

PER_BRANCH, SHARED = BudgetRule.PER_BRANCH, BudgetRule.SHARED
F, R = Strand.FORWARD, Strand.REVERSE


def _search(lengths, edges, genes, budget=1, rule=SHARED, max_distance=None):
    """genes: contig -> [(start, end)] of its genes on '+', in start order. Circular contigs are found from `edges`."""
    graph = AssemblyGraph(edges, lengths, circular_contigs(edges))
    batches = {c: IntervalBatch(np.array([s for s, _ in spans]), np.array([e for _, e in spans]),
                                np.array([F] * len(spans), dtype=np.int8))
               for c, spans in genes.items()}
    return FlankSearch(graph, batches, budget, max_distance, rule)


def _found(candidates):
    """(contig, gene index, gap, hops) of each candidate, in order."""
    return [(c.contig, c.gene_index, c.gap, c.hops) for c in candidates]


def _contexts(candidates):
    return [(c.contig, c.context, c.gap) for c in candidates]


def test_a_target_span_needs_a_strand():
    with pytest.raises(ValueError, match="strand"):
        _search({"ctgA": 1000}, [], {}).search(TargetSpan("ctgA", 100, 200, Strand.UNSTRANDED))


# The target's own contig (hop 0) -------------------------------------------------------------------------------------
def test_hop_0_flanks_come_nearest_first_up_to_the_budget():
    search = _search({"ctgA": 5000}, [], {"ctgA": [(100, 200), (300, 400), (1500, 1600), (1700, 1800)]}, budget=2)

    flanks = search.search(TargetSpan("ctgA", 1000, 1200, F))

    assert _found(flanks.upstream) == [("ctgA", 1, 600, 0), ("ctgA", 0, 800, 0)]
    assert _found(flanks.downstream) == [("ctgA", 2, 300, 0), ("ctgA", 3, 500, 0)]


def test_a_target_read_backwards_has_its_upstream_past_its_end():
    search = _search({"ctgA": 5000}, [], {"ctgA": [(300, 400), (1500, 1600)]})

    flanks = search.search(TargetSpan("ctgA", 1000, 1200, R))

    assert _found(flanks.upstream) == [("ctgA", 1, 300, 0)]
    assert _found(flanks.downstream) == [("ctgA", 0, 600, 0)]


@pytest.mark.parametrize("strand", [F, R])
def test_hop_0_flanks_take_their_context_from_interval_relate(strand):
    search = _search({"ctgA": 5000}, [], {"ctgA": [(300, 400), (1500, 1600)]})

    flanks = search.search(TargetSpan("ctgA", 1000, 1200, strand))

    assert [c.context for c in flanks.upstream + flanks.downstream] == [Context.UPSTREAM, Context.DOWNSTREAM]


def test_quirk_a_hop_0_gene_over_the_whole_target_is_an_upstream_flank_when_gene_ends_are_unsorted():
    # Sorted by start, the gene ends run 900, 300, so the binary search for the target start takes both genes
    search = _search({"ctgA": 5000}, [], {"ctgA": [(100, 900), (200, 300)]}, budget=2)

    flanks = search.search(TargetSpan("ctgA", 500, 600, F))

    assert _contexts(flanks.upstream) == [("ctgA", Context.UPSTREAM, 200), ("ctgA", Context.OVERLAPPING, 0)]


@pytest.mark.parametrize("limit, reported", [(499, False), (500, True)])
@pytest.mark.parametrize("gene, side", [((1500, 1800), "downstream"), ((100, 300), "upstream")])
def test_a_flank_on_the_targets_own_contig_is_reported_only_within_the_max_flank_distance(limit, reported, gene, side):
    # The gene is 500 bp from the target on either side
    search = _search({"ctgA": 10000}, [], {"ctgA": [gene]}, max_distance=limit)

    flanks = search.search(TargetSpan("ctgA", 800, 1000, F))

    assert _found(getattr(flanks, side)) == ([("ctgA", 0, 500, 0)] if reported else [])


def test_quirk_upstream_hop_0_misses_a_gene_when_overlapping_genes_leave_ends_unsorted():
    # Sorted by start, the gene ends run 300, 1100, 400: the binary search on them for the target start lands
    # between the first two, so gene 2, nearest, is never looked at
    search = _search({"ctgA": 5000}, [], {"ctgA": [(100, 300), (200, 1100), (350, 400)]}, budget=3)

    flanks = search.search(TargetSpan("ctgA", 1000, 1200, F))

    assert _found(flanks.upstream) == [("ctgA", 0, 700, 0)]


def test_quirk_a_hop_0_gap_is_measured_from_the_target_span_not_where_the_target_ends():
    # The target runs 200 bp on past its span's end at 995, onto ctgB, yet the gene at 997 is 2 bp away
    search = _search({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgA", F, "ctgB", F, overlap=100)],
                     {"ctgA": [(997, 1000)], "ctgB": [(500, 800)]})

    flanks = search.search(TargetSpan("ctgA", 695, 995, F, beyond_downstream=200))

    assert _found(flanks.downstream) == [("ctgA", 0, 2, 0)]


# Across the graph ----------------------------------------------------------------------------------------------------
def test_downstream_flank_across_a_link_counts_the_overlap_once():
    # 'ctgA+ -> ctgB+' share 100 bp, so ctgB starts at ctgA's 900 and its gene at 300 sits at 1200 in the joined
    # sequence
    search = _search({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgA", F, "ctgB", F, overlap=100)], {"ctgB": [(300, 600)]})

    flanks = search.search(TargetSpan("ctgA", 800, 1000, F))

    assert _found(flanks.downstream) == [("ctgB", 0, 200, 1)]


def test_upstream_flank_across_a_link_entered_backwards_counts_the_overlap_once():
    # 'ctgB+ -> ctgA+' share 100 bp; walking upstream leaves ctgA by its start and reads ctgB on '-'. ctgA starts at
    # ctgB's 900, so the gene ending at 600 on ctgB is 300 bp before the target at ctgA's start
    search = _search({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgB", F, "ctgA", F, overlap=100)], {"ctgB": [(300, 600)]})

    flanks = search.search(TargetSpan("ctgA", 0, 200, F))

    assert _found(flanks.upstream) == [("ctgB", 0, 300, 1)]


def test_a_gene_on_a_contig_entered_backwards_is_read_on_its_other_strand():
    search = _search({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgB", F, "ctgA", F)], {"ctgA": [(500, 600)],
                                                                                   "ctgB": [(300, 600)]})

    flanks = search.search(TargetSpan("ctgA", 0, 200, F))

    assert [(c.contig, c.strand) for c in flanks.upstream + flanks.downstream] == [("ctgB", R), ("ctgA", F)]


def test_flank_two_links_away_counts_each_overlap_once():
    # 'ctgA+ -> ctgB+ -> ctgC+' share 100 and 50 bp; ctgB carries no gene. ctgB starts at ctgA's 900 and ctgC at
    # 900 + 500 - 50 = 1350, so ctgC's gene at 200 sits at 1550, 550 bp past the target's end
    search = _search({"ctgA": 1000, "ctgB": 500, "ctgC": 1000},
                     [Edge("ctgA", F, "ctgB", F, overlap=100), Edge("ctgB", F, "ctgC", F, overlap=50)],
                     {"ctgC": [(200, 400)]})

    flanks = search.search(TargetSpan("ctgA", 800, 1000, F))

    assert _found(flanks.downstream) == [("ctgC", 0, 550, 2)]


@pytest.mark.parametrize("strand, side, context", [
    # A '+' target ends at ctgA's end; downstream is across the link
    (F, "downstream", Context.OVERLAPPING_END),
    # A '-' target reads ctgA backwards, so the link past ctgA's end is upstream of it
    (R, "upstream", Context.OVERLAPPING_START),
])
def test_a_gene_on_the_next_contig_that_runs_back_into_the_target_end_overlaps_it(strand, side, context):
    # 'ctgA+ -> ctgB+' share 100 bp, so ctgB's gene at 30 sits at ctgA's 930, 70 bp inside the target ending at 1000
    search = _search({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgA", F, "ctgB", F, overlap=100)], {"ctgB": [(30, 600)]})

    flanks = search.search(TargetSpan("ctgA", 800, 1000, strand))

    assert _contexts(getattr(flanks, side)) == [("ctgB", context, 0)]


@pytest.mark.parametrize("strand, side, context", [
    (F, "upstream", Context.OVERLAPPING_START),
    (R, "downstream", Context.OVERLAPPING_END),
])
def test_a_gene_on_the_previous_contig_that_runs_into_the_target_start_overlaps_it(strand, side, context):
    # 'ctgB+ -> ctgA+' share 100 bp, so ctgA starts at ctgB's 900 and ctgB's gene ending at 970 runs 70 bp into it
    search = _search({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgB", F, "ctgA", F, overlap=100)], {"ctgB": [(300, 970)]})

    flanks = search.search(TargetSpan("ctgA", 0, 200, strand))

    assert _contexts(getattr(flanks, side)) == [("ctgB", context, 0)]


@pytest.mark.parametrize("strand, side, beyond", [(F, "downstream", "beyond_downstream"),
                                                  (R, "upstream", "beyond_upstream")])
def test_a_flank_across_a_link_is_measured_from_where_the_target_ends_past_the_contig_end(strand, side, beyond):
    # ctgB starts at ctgA's 900, so its gene at 500 sits at 1400, 405 bp past the span's end at 995: 205 bp past the
    # target's end 200 bp further on
    search = _search({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgA", F, "ctgB", F, overlap=100)], {"ctgB": [(500, 800)]})

    flanks = search.search(TargetSpan("ctgA", 695, 995, strand, **{beyond: 200}))

    assert _found(getattr(flanks, side)) == [("ctgB", 0, 205, 1)]


@pytest.mark.parametrize("rule", [SHARED, PER_BRANCH])
@pytest.mark.parametrize("limit, reported", [(549, False), (550, True)])
def test_a_flank_across_links_is_reported_only_within_the_max_flank_distance(rule, limit, reported):
    # ctgB carries no gene, so the budget alone does not stop the walk; ctgC's gene is 550 bp past the target's end
    search = _search({"ctgA": 1000, "ctgB": 500, "ctgC": 1000},
                     [Edge("ctgA", F, "ctgB", F, overlap=100), Edge("ctgB", F, "ctgC", F, overlap=50)],
                     {"ctgC": [(200, 400)]}, rule=rule, max_distance=limit)

    flanks = search.search(TargetSpan("ctgA", 800, 1000, F))

    assert _found(flanks.downstream) == ([("ctgC", 0, 550, 2)] if reported else [])


@pytest.mark.parametrize("limit, reported", [(299, False), (300, True)])
def test_an_upstream_flank_across_a_link_entered_backwards_is_reported_only_within_the_max_flank_distance(
        limit, reported):
    # 'ctgB+ -> ctgA+' share 100 bp, so the gene ending at 600 on ctgB is 300 bp before the target at ctgA's start
    search = _search({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgB", F, "ctgA", F, overlap=100)], {"ctgB": [(300, 600)]},
                     max_distance=limit)

    flanks = search.search(TargetSpan("ctgA", 0, 200, F))

    assert _found(flanks.upstream) == ([("ctgB", 0, 300, 1)] if reported else [])


def test_a_gene_beyond_the_max_flank_distance_does_not_use_up_the_flank_budget():
    # Two links leave ctgA: the gene on ctgFar is beyond the limit, so the one on ctgNear still gets reported
    search = _search({"ctgA": 1000, "ctgFar": 5000, "ctgNear": 1000},
                     [Edge("ctgA", F, "ctgFar", F), Edge("ctgA", F, "ctgNear", F)],
                     {"ctgFar": [(4000, 4300)], "ctgNear": [(100, 400)]}, max_distance=1000)

    flanks = search.search(TargetSpan("ctgA", 800, 1000, F))

    assert _found(flanks.downstream) == [("ctgNear", 0, 100, 1)]


def test_quirk_a_link_past_the_max_flank_distance_is_not_marked_visited():
    # ctgX is reached first through the long ctgP, past the limit, then through the short ctgQ, within it
    search = _search({"ctgA": 1000, "ctgP": 5000, "ctgQ": 100, "ctgX": 1000},
                     [Edge("ctgA", F, "ctgP", F), Edge("ctgA", F, "ctgQ", F),
                      Edge("ctgP", F, "ctgX", F), Edge("ctgQ", F, "ctgX", F)],
                     {"ctgX": [(100, 400)]}, max_distance=1000)

    flanks = search.search(TargetSpan("ctgA", 800, 1000, F))

    assert _found(flanks.downstream) == [("ctgX", 0, 200, 2)]


# The flank budget ----------------------------------------------------------------------------------------------------
def _collapsed_repeat(rule):
    """An IS contig whose two ends each link to three flanking contigs, one per copy, with three genes on each."""
    lengths = {"is": 1000, **{f"{side}{i}": 3000 for side in "ud" for i in range(3)}}
    edges = [Edge(f"u{i}", F, "is", F) for i in range(3)] + [Edge("is", F, f"d{i}", F) for i in range(3)]
    genes = {ctg: [(0, 900), (1000, 1900), (2000, 2900)] for ctg in lengths if ctg != "is"}
    return _search(lengths, edges, genes, rule=rule).search(TargetSpan("is", 0, 1000, F))


def test_per_branch_gives_each_copy_of_a_collapsed_repeat_its_own_flank_budget():
    flanks = _collapsed_repeat(PER_BRANCH)

    # Downstream reads each copy's first gene; upstream walks the links backwards, so reads each copy's last gene
    assert [(c.contig, c.gene_index) for c in flanks.downstream] == [("d0", 0), ("d1", 0), ("d2", 0)]
    assert [(c.contig, c.gene_index) for c in flanks.upstream] == [("u0", 2), ("u1", 2), ("u2", 2)]


def test_shared_spends_one_flank_budget_across_branches():
    flanks = _collapsed_repeat(SHARED)

    assert [(c.contig, c.gene_index) for c in flanks.downstream] == [("d0", 0)]
    assert [(c.contig, c.gene_index) for c in flanks.upstream] == [("u0", 2)]


def _siblings_and_a_child(rule):
    """ctgP and ctgQ both leave ctgA, and ctgR leaves ctgP; ctgQ has two genes, the others one."""
    search = _search({c: 1000 for c in ("ctgA", "ctgP", "ctgQ", "ctgR")},
                     [Edge("ctgA", F, "ctgP", F), Edge("ctgA", F, "ctgQ", F), Edge("ctgP", F, "ctgR", F)],
                     {"ctgP": [(100, 200)], "ctgQ": [(100, 200), (300, 400)], "ctgR": [(100, 200)]},
                     budget=2, rule=rule)
    return [(c.contig, c.gene_index, c.hops) for c in search.search(TargetSpan("ctgA", 800, 1000, F)).downstream]


def test_shared_siblings_draw_down_one_budget_and_children_inherit_what_their_parent_left():
    # ctgP, visited first, takes one slot, so ctgQ gets only the first of its two genes; ctgR, past ctgP, still gets
    # the slot ctgP left when it was visited, as each contig queued carries the budget left at the time
    assert _siblings_and_a_child(SHARED) == [("ctgP", 0, 1), ("ctgQ", 0, 1), ("ctgR", 0, 2)]


def test_per_branch_resets_the_budget_for_every_link():
    assert _siblings_and_a_child(PER_BRANCH) == [("ctgP", 0, 1), ("ctgQ", 0, 1), ("ctgQ", 1, 1), ("ctgR", 0, 2)]


def test_a_gene_left_on_the_targets_own_contig_spends_the_budget_before_the_graph_walk():
    search = _search({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgA", F, "ctgB", F)],
                     {"ctgA": [(900, 950)], "ctgB": [(100, 200)]})

    flanks = search.search(TargetSpan("ctgA", 600, 800, F))

    assert _found(flanks.downstream) == [("ctgA", 0, 100, 0)]


# Claimed genes (#54) -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rule", [SHARED, PER_BRANCH])
def test_a_gene_two_branches_of_one_locus_reach_is_reported_once(rule):
    # 'ctgA+ -> ctgB+' and 'ctgA+ -> ctgB-' enter ctgB on both strands, and both reach its one gene
    search = _search({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgA", F, "ctgB", F), Edge("ctgA", F, "ctgB", R)],
                     {"ctgB": [(300, 600)]}, rule=rule)

    flanks = search.search(TargetSpan("ctgA", 800, 1000, F))

    assert [(c.contig, c.gene_index) for c in flanks.downstream] == [("ctgB", 0)]


def test_two_loci_whose_walks_reach_the_same_gene_both_report_it():
    # 'ctgA+ -> ctgC+' and 'ctgB+ -> ctgC+': each target runs off its contig's end into ctgC
    search = _search({"ctgA": 1000, "ctgB": 1000, "ctgC": 1000},
                     [Edge("ctgA", F, "ctgC", F), Edge("ctgB", F, "ctgC", F)], {"ctgC": [(300, 600)]})

    for contig in ("ctgA", "ctgB", "ctgA"):
        assert _found(search.search(TargetSpan(contig, 800, 1000, F)).downstream) == [("ctgC", 0, 300, 1)]


def test_quirk_a_claimed_gene_still_takes_a_slot_of_the_flank_budget():
    # ctgB's start links to both ends of ctgA, so both sides' walks enter ctgB on '+'. Upstream, searched first, claims
    # gene 0; downstream cuts ctgB's genes to its budget of 1 before skipping the claimed one, so never reaches gene 1
    search = _search({"ctgA": 1000, "ctgB": 1000}, [Edge("ctgA", F, "ctgB", F), Edge("ctgB", R, "ctgA", F)],
                     {"ctgB": [(100, 200), (300, 400)]})

    flanks = search.search(TargetSpan("ctgA", 400, 600, F))

    assert _found(flanks.upstream) == [("ctgB", 0, 500, 1)]
    assert flanks.downstream == []


# Circular contigs ----------------------------------------------------------------------------------------------------
def _circle(length, genes, overlap=0, **search_args):
    """One circular contig 'chrom', closed by a self-link, carrying genes on '+'."""
    return _search({"chrom": length}, [Edge("chrom", F, "chrom", F, overlap=overlap)], {"chrom": genes}, **search_args)


def test_a_gene_just_past_the_origin_flanks_a_target_at_the_end_of_a_circular_contig():
    flanks = _circle(10000, [(300, 600), (5000, 5500)]).search(TargetSpan("chrom", 9800, 10000, F))

    assert _found(flanks.downstream) == [("chrom", 0, 300, 1)]
    assert _found(flanks.upstream) == [("chrom", 1, 4300, 0)]


def test_a_gene_just_before_the_origin_flanks_a_target_at_the_start_of_a_circular_contig():
    flanks = _circle(10000, [(300, 600), (5000, 5500)]).search(TargetSpan("chrom", 0, 200, F))

    assert _found(flanks.upstream) == [("chrom", 1, 4500, 1)]
    assert _found(flanks.downstream) == [("chrom", 0, 100, 0)]


@pytest.mark.parametrize("limit, reported", [(299, False), (300, True)])
def test_a_flank_across_the_origin_is_reported_only_within_the_max_flank_distance(limit, reported):
    flanks = _circle(10000, [(300, 600), (5000, 5500)], max_distance=limit).search(TargetSpan("chrom", 9800, 10000, F))

    assert _found(flanks.downstream) == ([("chrom", 0, 300, 1)] if reported else [])


def test_a_flank_across_the_origin_counts_the_self_link_overlap_once():
    # The contig's last 100 bp repeat its first 100 bp, so the gene at 300 sits 200 bp past the contig's end
    flanks = _circle(10000, [(300, 600), (5000, 5500)], overlap=100).search(TargetSpan("chrom", 9800, 10000, F))

    assert _found(flanks.downstream) == [("chrom", 0, 200, 1)]


def test_a_walk_across_the_origin_stops_before_the_target_it_started_from():
    # The only gene carries the target, so going round the circle finds no gene on either side of it
    flanks = _circle(3000, [(1000, 2000)], budget=3).search(TargetSpan("chrom", 1200, 1500, F))

    assert flanks.upstream == flanks.downstream == []


def test_quirk_a_hop_0_gene_is_never_claimed_so_flanks_both_sides_of_a_small_circle():
    flanks = _circle(3000, [(2000, 2500)]).search(TargetSpan("chrom", 500, 700, F))

    assert _found(flanks.downstream) == [("chrom", 0, 1300, 0)]
    assert _found(flanks.upstream) == [("chrom", 0, 1000, 1)]


def test_quirk_a_gene_reached_back_across_the_origin_is_claimed_even_when_dropped():
    # The target is on '-', so upstream, searched first, reads chrom forwards: it finds both genes at hop 0, then wraps
    # round and charges gene 0 to its last budget slot before dropping it, as it lies past the target. Downstream,
    # wrapping the other way, then skips gene 0 as claimed
    flanks = _circle(10000, [(1000, 1500), (3000, 3500)], budget=3).search(TargetSpan("chrom", 500, 700, R))

    assert _found(flanks.upstream) == [("chrom", 0, 300, 0), ("chrom", 1, 2300, 0)]
    assert _found(flanks.downstream) == [("chrom", 1, 7000, 1)]


def test_a_self_link_beside_other_links_is_not_walked_as_an_origin():
    # 'rep' is a collapsed tandem repeat: its self-link is not a way back to its own start
    search = _search({"rep": 2000, "ctgA": 5000}, [Edge("rep", F, "rep", F), Edge("ctgA", F, "rep", F)],
                     {"rep": [(100, 400)]})

    flanks = search.search(TargetSpan("rep", 1800, 2000, F))

    assert flanks.downstream == []


# Repeat hubs (#56) ---------------------------------------------------------------------------------------------------
_HUB_LENGTHS = {"hub": 1000, "a1": 5000, "a2": 5000, "b1": 1000, "b2": 1000, "b3": 1000}
# a1 -> hub -> b1 is the stitched locus; a2 also links into the hub, and b2 (sharing 100 bp) and b3 (entered
# backwards) also leave it
_HUB_EDGES = [Edge("a1", F, "hub", F), Edge("a2", F, "hub", F), Edge("hub", F, "b1", F),
              Edge("hub", F, "b2", F, overlap=100), Edge("hub", F, "b3", R)]
_HUB_GENES = {"a1": [(4000, 4500)], "a2": [(4200, 4600)], "b1": [(500, 800)], "b2": [(300, 600)], "b3": [(600, 750)]}
_FIRST, _LAST = TargetSpan("a1", 4700, 5000, F), TargetSpan("b1", 0, 300, F)


def _branches(carried=0, blocked=frozenset()):
    return HubBranches(upstream=(Branch(Edge("hub", R, "a2", R)),),
                       downstream=(Branch(Edge("hub", F, "b2", F, overlap=100), carried),
                                   Branch(Edge("hub", F, "b3", R))),
                       blocked=blocked)


def test_each_hub_branch_is_walked_from_the_hubs_edge_with_the_full_budget_after_the_locus_own_ends():
    # b2's gene at 300 is 200 bp past the 100 bp it shares with the hub; b3, read backwards, starts with its gene ending
    # at 750, 250 bp in. Upstream, a2's gene ends 400 bp before the hub
    search = _search(_HUB_LENGTHS, _HUB_EDGES, _HUB_GENES, rule=PER_BRANCH)

    flanks = search.search(_FIRST, _LAST, branches=_branches())

    assert _found(flanks.downstream) == [("b1", 0, 200, 0), ("b2", 0, 200, 1), ("b3", 0, 250, 1)]
    assert _found(flanks.upstream) == [("a1", 0, 200, 0), ("a2", 0, 400, 1)]


def test_a_hub_branch_is_measured_from_where_the_target_its_copy_carries_onto_it_ends():
    search = _search(_HUB_LENGTHS, _HUB_EDGES, _HUB_GENES, rule=PER_BRANCH)

    flanks = search.search(_FIRST, _LAST, branches=_branches(carried=100))

    assert ("b2", 0, 100, 1) in _found(flanks.downstream)


def test_a_search_never_enters_a_blocked_contig():
    search = _search(_HUB_LENGTHS, _HUB_EDGES, _HUB_GENES, rule=PER_BRANCH)

    flanks = search.search(_FIRST, _LAST, branches=_branches(blocked=frozenset({"b2", "a2"})))

    assert _found(flanks.downstream) == [("b1", 0, 200, 0), ("b3", 0, 250, 1)]
    assert _found(flanks.upstream) == [("a1", 0, 200, 0)]


def test_without_a_hub_a_stitched_locus_is_searched_only_from_its_ends():
    search = _search(_HUB_LENGTHS, _HUB_EDGES, _HUB_GENES)

    flanks = search.search(_FIRST, _LAST)

    assert _found(flanks.downstream) == [("b1", 0, 200, 0)]
    assert _found(flanks.upstream) == [("a1", 0, 200, 0)]
