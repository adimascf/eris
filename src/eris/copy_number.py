"""
Collapse-mode copy number: the copies of a locus in the genome, counting each replicon once (docs/adr/0003).

A locus whose target sits in unique sequence is one copy. Copies that collapsed into a repeat hub are counted from the
hub's anchors: the hub's depth is shared out over the depth tiers of its anchors, and the number of links that reach
an anchor checks it, giving a range when the two disagree.
"""
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, NamedTuple, TYPE_CHECKING

from eris.graph import Edge, Graph
from eris.interval import Strand

if TYPE_CHECKING:
    from eris.io import GenomeAssembly
    from eris.pipeline import Locus

# Constants ------------------------------------------------------------------------------------------------------------
MIN_ANCHOR_LENGTH = 1000  # bp; a shorter contig next to a hub is a stub that the walk crosses
STUB_BUDGET = 500  # bp of stubs a walk may cross before it gives up (ISEc12's flanks break into 200-400 bp stubs)
TIER_TOLERANCE = 1.5  # largest depth ratio between neighbouring anchors in one depth tier
END_SLOP = 10  # bp a target alignment may stop short of a contig's ends, beyond the link overlap, and still cover it


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CopyNumber:
    """
    A locus's copy number. `estimate` comes from anchor depth; `low` and `high` are the lower and higher of it and the
    anchor link count. `tiers` holds (copies, mean anchor depth) per depth tier. `fallback` is set when the hub had no
    anchors and was measured against the genome's unique contigs instead. `hub` is the repeat hub's contigs, empty
    for a locus in unique sequence; loci that share a hub share its copy number.
    """
    estimate: int
    low: int
    high: int
    tiers: tuple[tuple[int, float], ...] = ()  # for a fallback, (copies, baseline depth)
    fallback: bool = False
    hub: frozenset[str] = frozenset()


UNIQUE = CopyNumber(1, 1, 1)


class WalkEndKind(StrEnum):
    """What a walk out of a repeat hub ended at."""
    ANCHOR = 'anchor'
    REPEAT = 'repeat'  # another hub's contig
    DEAD_END = 'dead_end'  # the stubs crossed exceeded STUB_BUDGET
    TIP = 'tip'  # a stub with no onward link


class WalkEnd(NamedTuple):
    """Where one walk out of a repeat hub ends."""
    contig: str
    strand: Strand
    kind: WalkEndKind


# Functions ------------------------------------------------------------------------------------------------------------
def assign_copy_numbers(loci: list['Locus'], genome: 'GenomeAssembly') -> None:
    """Sets each locus's copy number, and its `fractional_depth`, which the report prints as copy_number."""
    overlap = max((e.overlap for e in genome.edges), default=0)
    hubs = _merge_hubs([_hub(locus, genome.contig_lengths, overlap) for locus in loci])
    repeats = {contig for hub, _ in hubs for contig in hub}
    graph = Graph(genome.edges + [e.complement() for e in genome.edges])
    for hub, members in hubs:
        copies = UNIQUE
        if hub:
            ends = _walk_out(graph, hub, genome.contig_lengths, repeats - hub.keys())
            hub_depth = max(genome.contig_depths[c] for c in hub)
            copies = _estimate(hub_depth, ends, genome.contig_depths, frozenset(hub),
                               lambda: _baseline_depth(genome, repeats))
        for i in members:
            loci[i].copy_number = copies
            loci[i].fractional_depth = float(copies.estimate)


def depth_tiers(depths: list[float], tolerance: float = TIER_TOLERANCE) -> list[list[float]]:
    """
    Groups depths into depth tiers: sorted, a new tier starts where a depth exceeds the one before it by more than
    `tolerance`. Comparing neighbours, not each tier's extremes, keeps a gradient such as the chromosome's in one tier.
    """
    tiers: list[list[float]] = []
    for depth in sorted(depths):
        if tiers and depth <= tiers[-1][-1] * tolerance:
            tiers[-1].append(depth)
        else:
            tiers.append([depth])
    return tiers


def _hub(locus: 'Locus', lengths: dict[str, int], overlap: int) -> dict[str, Strand]:
    """
    The contigs a locus's targets cover end to end, each with the strand it is first read on. Together they are the
    locus's repeat hub; a contig with sequence beyond the target is unique sequence, not part of it.
    """
    hub: dict[str, Strand] = {}
    for target in locus.targets:
        for seg in target.segments:
            if lengths[seg.contig] - (seg.end - seg.start) <= overlap + END_SLOP:
                hub.setdefault(seg.contig, seg.strand)
    return hub


def _merge_hubs(hubs: list[dict[str, Strand]]) -> list[tuple[dict[str, Strand], list[int]]]:
    """
    Merges the hubs of loci that share a contig, so near-identical targets walking through one hub (#14) are counted
    once. Returns each merged hub with the indices of its loci; a locus with no hub stands alone.
    """
    merged: list[tuple[dict[str, Strand], list[int]]] = []
    for i, hub in enumerate(hubs):
        sharing = [m for m in merged if hub.keys() & m[0].keys()]
        combined: dict[str, Strand] = {}
        members = []
        for other, other_members in sharing:
            combined.update({c: s for c, s in other.items() if c not in combined})
            members += other_members
            merged.remove((other, other_members))
        combined.update({c: s for c, s in hub.items() if c not in combined})
        merged.append((combined, sorted(members + [i])))
    return merged


def _walk_out(graph: Graph, hub: dict[str, Strand], lengths: dict[str, int],
              repeats: set[str]) -> dict[str, list[WalkEnd]]:
    """
    Walks out of a repeat hub on each side, following edge orientation so a walk never turns back towards the hub's
    side of a contig (an undirected walk out of the IS26 hub reaches the rRNA repeat). A walk ends at the first contig
    of at least MIN_ANCHOR_LENGTH (an anchor), at another hub's contig, or when the stubs it crossed exceed
    STUB_BUDGET. Each distinct oriented end is one link, so an anchor between two inverted copies counts twice.
    """
    ends = {}
    for side, sign in (("upstream", -1), ("downstream", 1)):
        found: dict[tuple[str, Strand], WalkEnd] = {}
        stack = [(edge, 0) for contig, strand in hub.items() for edge in graph.get_neighbors(contig)
                 if edge.u_strand == strand * sign]
        seen = set()
        while stack:
            edge, used = stack.pop()
            node = (edge.v, edge.v_strand)
            if node in seen or edge.v in hub:
                continue
            seen.add(node)
            used += lengths[edge.v] - edge.overlap
            onward = [e for e in graph.get_neighbors(edge.v) if e.u_strand == edge.v_strand]
            if edge.v in repeats:
                kind = WalkEndKind.REPEAT
            elif lengths[edge.v] >= MIN_ANCHOR_LENGTH:
                kind = WalkEndKind.ANCHOR
            elif used > STUB_BUDGET:
                kind = WalkEndKind.DEAD_END
            elif not onward:
                kind = WalkEndKind.TIP
            else:
                stack.extend((e, used) for e in onward)
                continue
            found.setdefault(node, WalkEnd(edge.v, edge.v_strand, kind))
        ends[side] = list(found.values())
    return ends


def _baseline_depth(genome: 'GenomeAssembly', repeats: set[str]) -> float:
    """Length-weighted depth of the contigs of at least MIN_ANCHOR_LENGTH that are not in a repeat hub."""
    contigs = [c for c, n in genome.contig_lengths.items() if n >= MIN_ANCHOR_LENGTH and c not in repeats]
    total = sum(genome.contig_lengths[c] for c in contigs)
    return sum(genome.contig_depths[c] * genome.contig_lengths[c] for c in contigs) / total if total else 1.0


def _round_half_up(x: float) -> int:
    """Rounds .5 up, where Python's round() goes to the even neighbour (2.5 would give 2)."""
    return math.floor(x + 0.5)


def _estimate(hub_depth: float, ends: dict[str, list[WalkEnd]], depths: dict[str, float], hub: frozenset[str],
              baseline: Callable[[], float]) -> CopyNumber:
    """
    A repeat hub's copy number from where the walks out of its two sides end. The estimate shares the hub's depth out
    over the depth tiers of its anchors, each tier weighted by the anchors it has on the side that reaches more of
    them, so with one tier it is hub depth over mean anchor depth, and a tier one side misses still gets its copy.
    The check is the number of links to anchors, halved for the two sides. With no anchors, the estimate is hub depth
    over `baseline()`, flagged as a fallback; there are no links to check it against, so it gives no range.
    """
    anchors = {side: [depths[e.contig] for e in side_ends if e.kind == WalkEndKind.ANCHOR]
               for side, side_ends in ends.items()}
    if not any(anchors.values()):
        depth = baseline()
        estimate = max(1, _round_half_up(hub_depth / depth))
        return CopyNumber(estimate, estimate, estimate, ((estimate, depth),), True, hub)
    links = _round_half_up(sum(len(a) for a in anchors.values()) / 2)
    tiers = depth_tiers([d for side in anchors.values() for d in side])
    # (anchors on the side that reaches more of them, mean depth) per tier
    tier_anchors = [(max(sum(d in t for d in side) for side in anchors.values()), sum(t) / len(t)) for t in tiers]
    scale = hub_depth / sum(n * depth for n, depth in tier_anchors)
    tier_copies = tuple((max(1, _round_half_up(n * scale)), depth) for n, depth in tier_anchors)
    estimate = sum(copies for copies, _ in tier_copies)
    return CopyNumber(estimate, min(links, estimate), max(links, estimate), tier_copies, False, hub)
