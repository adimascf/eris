"""
Collapse-mode copy number: the copies of a locus in the genome, counting each replicon once (docs/adr/0003).

A locus whose target sits in unique sequence is one copy. Copies that collapsed into a repeat hub are counted from the
hub's anchors: the hub's depth is shared out over the depth tiers of its anchors. The number of links that reach an
anchor gives a second count, reported beside it so the two can be compared.
"""
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, NamedTuple, Optional, TYPE_CHECKING

from eris.graph import AssemblyGraph
from eris.hub import Hubs, RepeatHub
from eris.interval import Strand

if TYPE_CHECKING:
    from eris.io import GenomeAssembly
    from eris.pipeline import Locus

# Constants ------------------------------------------------------------------------------------------------------------
MIN_ANCHOR_LENGTH = 1000  # bp; a shorter contig next to a hub is a stub that the walk crosses
STUB_BUDGET = 500  # bp of stubs a walk may cross before it gives up (ISEc12's flanks break into 200-400 bp stubs)
TIER_TOLERANCE = 1.5  # largest depth ratio between neighbouring anchors in one depth tier


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CopyNumber:
    """
    A locus's copy number. `estimate` comes from anchor depth; `edges` is the anchor link count, the links that reach
    an anchor on both sides, halved (0 for a fallback). `tiers` holds (copies, mean anchor depth) per depth tier.
    `fallback` is set when the hub had no anchors and was measured against the genome's unique contigs instead. `hub`
    is the repeat hub's contigs, empty for a locus in unique sequence; loci that share a hub share its copy number.
    A hub's `estimate` is None when the assembly has no read depth, so only `edges` was measured.
    """
    estimate: Optional[int]
    edges: int
    tiers: tuple[tuple[int, float], ...] = ()  # for a fallback, (copies, baseline depth)
    fallback: bool = False
    hub: frozenset[str] = frozenset()


UNIQUE = CopyNumber(1, 1)  # a locus in unique sequence is one copy, with or without read depth


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
def assign_copy_numbers(loci: list['Locus'], genome: 'GenomeAssembly', graph: AssemblyGraph) -> None:
    """
    Sets each locus's copy number, and its `fractional_depth`, which the report prints as copy_number. Without read
    depth a repeat hub gets only its anchor link count; its estimate and `fractional_depth` are None. A locus in
    unique sequence is one copy either way. Loci whose hubs share a contig share one copy number (#14). `graph` is
    `genome`'s assembly graph.
    """
    hubs = Hubs(graph)
    groups = hubs.group([hubs.of_locus(locus) for locus in loci])
    repeats = {contig for hub, _ in groups for contig in hub}
    for hub, members in groups:
        copies = UNIQUE
        if hub:
            ends = _walk_out(graph, hubs, hub, repeats - hub.names)
            if genome.has_depth:
                hub_depth = max(genome.contig_depths[c] for c in hub)
                copies = _estimate(hub_depth, ends, genome.contig_depths, hub.names,
                                   lambda: _baseline_depth(genome, repeats))
            else:
                copies = CopyNumber(None, _anchor_links(ends), hub=hub.names)
        for i in members:
            loci[i].copy_number = copies
            loci[i].fractional_depth = None if copies.estimate is None else float(copies.estimate)


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


def _walk_out(graph: AssemblyGraph, hubs: Hubs, hub: RepeatHub, repeats: set[str]) -> dict[str, list[WalkEnd]]:
    """
    Walks out of a repeat hub on each side, following edge orientation so a walk never turns back towards the hub's
    side of a contig (an undirected walk out of the IS26 hub reaches the rRNA repeat). A walk ends at the first contig
    of at least MIN_ANCHOR_LENGTH (an anchor), at another hub's contig, or when the stubs it crossed exceed
    STUB_BUDGET. Each distinct oriented end is one link, so an anchor between two inverted copies counts twice. A walk
    starts on the links out of every contig in the hub (see `Hubs.leaving`), since a copy can leave the hub from any
    of them.
    """
    lengths = graph.contig_lengths
    ends = {}
    for side, sign in (("upstream", -1), ("downstream", 1)):
        found: dict[tuple[str, Strand], WalkEnd] = {}
        stack = [(edge, 0) for edge in hubs.leaving(hub, sign)]
        seen = set()
        while stack:
            edge, used = stack.pop()
            node = (edge.v, edge.v_strand)
            if node in seen or edge.v in hub:
                continue
            seen.add(node)
            used += lengths[edge.v] - edge.overlap
            onward = graph.links_from(edge.v, edge.v_strand)
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


def _anchor_links(ends: dict[str, list[WalkEnd]]) -> int:
    """The number of links from a repeat hub that reach an anchor, halved for its two sides."""
    return _round_half_up(sum(e.kind == WalkEndKind.ANCHOR for side in ends.values() for e in side) / 2)


def _round_half_up(x: float) -> int:
    """Rounds .5 up, where Python's round() goes to the even neighbour (2.5 would give 2)."""
    return math.floor(x + 0.5)


def _estimate(hub_depth: float, ends: dict[str, list[WalkEnd]], depths: dict[str, float], hub: frozenset[str],
              baseline: Callable[[], float]) -> CopyNumber:
    """
    A repeat hub's copy number from where the walks out of its two sides end. The estimate shares the hub's depth out
    over the depth tiers of its anchors, each tier weighted by the anchors it has on the side that reaches more of
    them, so with one tier it is hub depth over mean anchor depth, and a tier one side misses still gets its copy.
    `edges` is the number of links to anchors, halved for the two sides. With no anchors, the estimate is hub depth
    over `baseline()`, flagged as a fallback, and `edges` is 0.
    """
    anchors = {side: [depths[e.contig] for e in side_ends if e.kind == WalkEndKind.ANCHOR]
               for side, side_ends in ends.items()}
    if not any(anchors.values()):
        depth = baseline()
        estimate = max(1, _round_half_up(hub_depth / depth))
        return CopyNumber(estimate, 0, ((estimate, depth),), True, hub)
    links = _anchor_links(ends)
    tiers = depth_tiers([d for side in anchors.values() for d in side])
    # (anchors on the side that reaches more of them, mean depth) per tier
    tier_anchors = [(max(sum(d in t for d in side) for side in anchors.values()), sum(t) / len(t)) for t in tiers]
    scale = hub_depth / sum(n * depth for n, depth in tier_anchors)
    tier_copies = tuple((max(1, _round_half_up(n * scale)), depth) for n, depth in tier_anchors)
    estimate = sum(copies for copies, _ in tier_copies)
    return CopyNumber(estimate, links, tier_copies, False, hub)
