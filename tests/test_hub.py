import pytest
from typing import Iterable

from eris.alignment import AlignmentRecord
from eris.constants import FeatureType, Strand
from eris.graph import Edge
from eris.hub import Hubs, RepeatHub
from eris.pipeline import GenomicFeature, LocationSegment, Locus

F, R = Strand.FORWARD, Strand.REVERSE


def record(contig: str, length: int, start: int, end: int, strand: Strand = F) -> AlignmentRecord:
    return AlignmentRecord(0, "IS1", 1000, 0, end - start, contig, length, start, end, strand, end - start,
                           end - start, 0, 60, f"{end - start}M")


def locus(*segments: tuple[str, int, int, Strand]) -> Locus:
    target = GenomicFeature("IS1", FeatureType.CDS, [LocationSegment(*s) for s in segments])
    return Locus("locus", "|".join(s[0] for s in segments), segments[0][1], segments[-1][2], [target], [], [], [])


def hubs(lengths: dict[str, int], edges: Iterable[Edge] = ()) -> Hubs:
    return Hubs(list(edges), lengths)


def test_a_contig_is_in_the_hub_only_if_the_target_covers_it_end_to_end():
    # Links overlap by 50, so a target may stop 50 + STRICT_SLACK short of a contig's linked ends
    h = hubs({"a": 5000, "hub": 800, "short": 800, "b": 5000},
             [Edge("a", F, "hub", F, 50), Edge("hub", F, "b", F, 50)])

    path = h.of_path([record("a", 5000, 4800, 5000), record("hub", 800, 30, 770), record("short", 800, 0, 700)])

    assert path.contigs == (("hub", F),)


# Contigs linked by 127 bp at their end only (is), at both ends (rep), and at their start only, by a link leaving
# backwards (back) (#85)
MIXED = hubs({"is": 1300, "rep": 1300, "back": 1300, "left": 5000, "right": 5000},
             [Edge("is", F, "right", F, 127), Edge("left", F, "rep", F, 127), Edge("rep", F, "right", F, 127),
              Edge("back", R, "left", R, 127)])


@pytest.mark.parametrize("rec, covered", [
    (record("is", 1300, 120, 1300), False),  # 120 bp short of a dead end is unique sequence beside the target
    (record("is", 1300, 11, 1300), False),  # past STRICT_SLACK at the dead end
    (record("is", 1300, 10, 1300), True),
    (record("is", 1300, 0, 1163), True),  # 137 bp short of the linked end: its overlap plus STRICT_SLACK
    (record("is", 1300, 0, 1162), False),
    (record("rep", 1300, 100, 1200), True),  # each end judged on its own, not their shortfalls summed
    (record("is", 1300, 0, 1163, R), True),  # a contig's ends are on its forward strand, whichever way it is read
    (record("back", 1300, 137, 1300), True),  # a link leaving on '-' leaves by the contig's start
    (record("back", 1300, 0, 1163), False),
])
def test_each_contig_end_is_reached_within_its_own_links_overlap(rec, covered):
    assert bool(MIXED.of_path([rec])) == covered


def test_a_contig_entered_backwards_has_its_links_overlap_at_the_end_it_is_entered_by():
    # left's end joins is's end, read backwards: is's end carries the 127 bp, its start is a dead end
    h = hubs({"is": 1300, "left": 5000}, [Edge("left", F, "is", R, 127)])

    assert h.of_locus(locus(("is", 0, 1163, R))) and not h.of_locus(locus(("is", 120, 1300, R)))


def test_the_hub_holds_each_contig_with_its_strand_in_path_order():
    h = hubs({"h1": 800, "h2": 600, "h3": 700})

    path = h.of_path([record("h2", 600, 0, 600, R), record("h1", 800, 0, 800), record("h3", 700, 0, 700, R)])
    of_locus = h.of_locus(locus(("h2", 0, 600, R), ("h1", 0, 800, F), ("h3", 0, 700, R)))

    assert path == of_locus == RepeatHub((("h2", R), ("h1", F), ("h3", R)))


def test_a_locus_in_unique_sequence_has_an_empty_hub():
    h = hubs({"chr": 100000})

    assert not h.of_locus(locus(("chr", 1000, 2000, F)))


def test_grouping_merges_hubs_that_share_a_contig_transitively():
    h = hubs({})
    a = RepeatHub((("h1", F),))
    b = RepeatHub((("h2", R),))
    c = RepeatHub((("h1", R), ("h2", F)))  # joins a and b
    d = RepeatHub((("h9", F),))
    empty = RepeatHub(())

    groups = h.group([a, empty, b, d, c, empty])

    # Groups are in the order their last member came; members of the groups a later hub joins come first, the
    # latest-merged group's members before an earlier one's; the first hub seen keeps a contig's strand
    assert groups == [
        (RepeatHub(()), [1]),
        (RepeatHub((("h9", F),)), [3]),
        (RepeatHub((("h1", F), ("h2", R))), [2, 0, 4]),
        (RepeatHub(()), [5]),
    ]


def test_copy_number_leaves_by_every_hub_contig_and_flanks_by_the_outermost_only():
    # hub = h1+ h2+ h3+; each has a link out downstream, and h1 has one out upstream
    edges = [Edge("u", F, "h1", F), Edge("h1", F, "h2", F), Edge("h2", F, "h3", F),
             Edge("h1", F, "x1", F), Edge("h2", F, "x2", F), Edge("h3", F, "x3", F)]
    h = hubs({"u": 5000, "h1": 800, "h2": 800, "h3": 800, "x1": 5000, "x2": 5000, "x3": 5000}, edges)
    hub = RepeatHub((("h1", F), ("h2", F), ("h3", F)))

    assert [e.v for e in h.leaving(hub, 1)] == ["x1", "x2", "x3"]
    assert [e.v for e in h.leaving_outermost(hub, 1)] == ["x3"]
    # Upstream, the outermost contig is the first one; the links are read on the complement strands
    assert [(e.u, e.v) for e in h.leaving(hub, -1)] == [("h1", "u")]
    assert [(e.u, e.v) for e in h.leaving_outermost(hub, -1)] == [("h1", "u")]


def test_a_reverse_strand_hub_leaves_by_its_contigs_starts():
    h = hubs({"h1": 800, "x": 5000, "y": 5000}, [Edge("x", F, "h1", F), Edge("h1", F, "y", F)])
    hub = RepeatHub((("h1", R),))

    assert [(e.u, e.v, e.v_strand) for e in h.leaving(hub, 1)] == [("h1", "x", R)]
