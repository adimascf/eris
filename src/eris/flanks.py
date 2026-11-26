"""
The flank search: the genes flanking one locus on each side, along the target's own contig and out across the
assembly graph.
"""
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import NamedTuple, Optional, cast

import numpy as np

from eris.constants import Context, Strand
from eris.graph import AssemblyGraph, Edge, is_self_link
from eris.interval import Interval, IntervalBatch


# Classes --------------------------------------------------------------------------------------------------------------
class TargetSpan(NamedTuple):
    """
    Where an IS copy's target lies on one contig.

    Attributes:
        contig: The contig.
        start: Start on the contig.
        end: End on the contig.
        strand: The strand the target is read on, FORWARD or REVERSE; it sets which side is upstream.
        beyond_upstream: Target bases that run on past the contig end upstream of the target (#66).
        beyond_downstream: Target bases that run on past the contig end downstream of it.
    """
    contig: str
    start: int
    end: int
    strand: Strand
    beyond_upstream: int = 0
    beyond_downstream: int = 0


class Branch(NamedTuple):
    """
    One link leaving a repeat hub, walked with the full flank budget (#56).

    Attributes:
        link: The link, from the hub's outermost contig on that side.
        carried: Target bases the copy along this branch carries onto `link.v` past the link's overlap (#65).
    """
    link: Edge
    carried: int = 0


class HubBranches(NamedTuple):
    """
    The other links leaving a collapse-mode stitched locus's repeat hub, besides those the search from the locus's own
    ends already follows.

    Attributes:
        upstream: The branches on the upstream side.
        downstream: The branches on the downstream side.
        blocked: Contigs never entered, on either side, so that no branch loops back into the hub and reports its genes
            as flanks.
    """
    upstream: tuple[Branch, ...] = ()
    downstream: tuple[Branch, ...] = ()
    blocked: frozenset[str] = frozenset()


class FlankCandidate(NamedTuple):
    """
    One gene flanking a locus.

    Attributes:
        contig: The gene's contig.
        gene_index: The gene's original index on its contig.
        context: UPSTREAM or DOWNSTREAM, or OVERLAPPING_START or OVERLAPPING_END for a gene that runs back over the
            target's end (#64); on the target's own contig (hop 0), whatever Interval.relate gives.
        gap: Flank distance in bp, never negative.
        hops: Links crossed to reach the gene; 0 on the target's own contig.
        strand: The gene's strand as the search reads it: flipped for a contig entered on its reverse strand.
    """
    contig: str
    gene_index: int
    context: Context
    gap: int
    hops: int
    strand: Strand


class Flanks(NamedTuple):
    """A locus's flanks on each side: hop-0 genes nearest first, then genes across the graph in the order reached."""
    upstream: list[FlankCandidate]
    downstream: list[FlankCandidate]


class BudgetRule(Enum):
    """
    How the flank budget is shared between the links a search follows. PER_BRANCH (collapse mode): every link starts
    with the budget of the contig it leaves, so every copy collapsed into a repeat gets flanks. SHARED (variant mode):
    links leaving one contig draw that contig's budget down in turn, as each copy is already its own locus and a budget
    per branch would only fan its flanks out.
    """
    PER_BRANCH = 'per_branch'
    SHARED = 'shared'


class _Side(NamedTuple):
    """One side's search, worked out from its target span."""
    contig: str
    boundary: int  # the target's end on this side, in contig coordinates
    direction: int  # 1 to walk the contig forwards, -1 backwards
    context: Context
    target: Interval
    beyond: int


class _Step(NamedTuple):
    """One contig queued in a side's graph walk."""
    contig: str
    exit_strand: Strand  # the strand it is read on, so it is left by its end, or on '-', its start
    budget: int  # flanks still to find past it
    hops: int  # links crossed on reaching its next contig
    start: int  # where it starts along the walk, relative to where the target ends
    links: Optional[tuple[Edge, ...]] = None  # the links to follow, or None for every link leaving it


class FlankSearch:
    """
    Searches the flanks of one locus at a time.

    Each side is searched first along the target's own contig (hop 0), then breadth-first across the assembly graph out
    of the contig end on that side, and, for a repeat hub, along each of its branches on that side. A side reports at
    most `budget` genes from its own contig, and the graph walk takes whatever budget is left at each contig (see
    BudgetRule). Genes further than `max_distance` from the target are left out wherever they lie (#55), and a gene the
    graph walk reports for one side is not reported again for the other (#54).
    """
    __slots__ = ('graph', 'genes', 'budget', 'max_distance', 'rule')

    def __init__(self, graph: AssemblyGraph, genes: Mapping[str, IntervalBatch], budget: int,
                 max_distance: Optional[int], rule: BudgetRule):
        """
        Args:
            graph: The assembly graph.
            genes: Each contig's genes, sorted by start.
            budget: Max flanks on each side.
            max_distance: Max flank distance in bp, on the target's own contig or across the graph; None for no limit.
            rule: How the budget is shared between links.
        """
        self.graph = graph
        self.genes = genes
        self.budget = budget
        self.max_distance = max_distance
        self.rule = rule

    def search(self, first: TargetSpan, last: Optional[TargetSpan] = None, *,
               branches: Optional[HubBranches] = None) -> Flanks:
        """
        The flanks of one locus: upstream of `first` and downstream of `last` (by default `first`), plus along the
        branches of its repeat hub, if any.
        """
        branches = branches or HubBranches()
        claimed: set[tuple[str, int]] = set()  # genes the graph walks report; shared by both sides, never by loci
        upstream = self._side_flanks(self._upstream(first), branches.upstream, branches.blocked, claimed)
        downstream = self._side_flanks(self._downstream(last or first), branches.downstream, branches.blocked,
                                       claimed)
        return Flanks(upstream, downstream)

    @staticmethod
    def _upstream(span: TargetSpan) -> _Side:
        forward = _read_forward(span.strand)
        return _Side(span.contig, span.start if forward else span.end, -1 if forward else 1, Context.UPSTREAM,
                     Interval(span.start, span.end, span.strand), span.beyond_upstream)

    @staticmethod
    def _downstream(span: TargetSpan) -> _Side:
        forward = _read_forward(span.strand)
        return _Side(span.contig, span.end if forward else span.start, 1 if forward else -1, Context.DOWNSTREAM,
                     Interval(span.start, span.end, span.strand), span.beyond_downstream)

    def _side_flanks(self, side: _Side, branches: tuple[Branch, ...], blocked: frozenset[str],
                     claimed: set[tuple[str, int]]) -> list[FlankCandidate]:
        """
        One side's flanks. The graph walk places every contig by where it starts along the walk, relative to where the
        target ends: past the side's boundary by `side.beyond`, or for a branch, past the hub's edge by its `carried`.
        Linked contigs share their link's overlap, so each overlap is counted once.
        """
        limit = self.max_distance
        found = self._own_contig(side)
        left = self.budget - len(found)

        length = self.graph.contig_lengths[side.contig]
        walk_boundary = side.boundary if side.direction == 1 else length - side.boundary
        exit_strand = Strand.FORWARD if side.direction == 1 else Strand.REVERSE
        queue = [_Step(side.contig, exit_strand, left, 1, -(walk_boundary + side.beyond))]
        queue += [_Step(b.link.u, b.link.u_strand, self.budget, 1, -self.graph.contig_lengths[b.link.u] - b.carried,
                        (b.link,)) for b in branches]
        visited = {(step.contig, step.exit_strand) for step in queue}
        wrapped = False  # whether this side's walk has gone round a circular contig's origin, from any start

        while queue:
            step = queue.pop(0)
            if step.budget <= 0:
                continue
            left = step.budget
            links = self.graph.links_from(step.contig, step.exit_strand) if step.links is None else step.links
            for link in links:
                if link.u_strand != step.exit_strand or link.v in blocked:
                    continue
                if self.rule is BudgetRule.PER_BRANCH:
                    left = step.budget
                entered = link.v
                entered_start = step.start + self.graph.contig_lengths[step.contig] - link.overlap
                if limit is not None and entered_start > limit:
                    continue  # checked before the visited mark, so it may still be entered by a shorter way
                # The self-link of a circular contig leads back to its start across the origin: go round it once
                if entered in self.graph.circular and is_self_link(link) and not wrapped:
                    wrapped = True
                elif (entered, link.v_strand) in visited:
                    continue
                visited.add((entered, link.v_strand))

                reported, charged = self._across_link(side, entered, link.v_strand, entered_start, step.hops, left,
                                                      claimed)
                found += reported
                left -= charged
                if left > 0:
                    queue.append(_Step(entered, link.v_strand, left, step.hops + 1, entered_start))
        return found

    def _own_contig(self, side: _Side) -> list[FlankCandidate]:
        """The flanks on the target's own contig, nearest first, up to the budget."""
        genes = self.genes.get(side.contig)
        if not genes:
            return []
        if side.direction == -1:
            # Upstream of a '+' target; the ends are unsorted where genes overlap
            idx = int(np.searchsorted(genes.ends, side.boundary, side='right'))
            nearest_first: Iterable[int] = reversed(range(max(0, idx - self.budget), idx))
            gaps = side.boundary - genes.ends
        else:
            idx = int(np.searchsorted(genes.starts, side.boundary, side='left'))
            nearest_first = range(idx, min(len(genes), idx + self.budget))
            gaps = genes.starts - side.boundary
        original_indices = cast(np.ndarray, genes.original_indices)
        found = []
        for i in nearest_first:
            gap = max(0, int(gaps[i]))
            if self.max_distance is not None and gap > self.max_distance:
                continue
            strand = Strand(genes.strands[i])
            context = side.target.relate(Interval(int(genes.starts[i]), int(genes.ends[i]), strand))
            found.append(FlankCandidate(side.contig, int(original_indices[i]), context, gap, 0, strand))
        return found

    def _across_link(self, side: _Side, contig: str, strand: Strand, start: int, hops: int, budget: int,
                     claimed: set[tuple[str, int]]) -> tuple[list[FlankCandidate], int]:
        """
        The flanks on a contig the graph walk enters on `strand`, starting at `start` relative to the target's end:
        up to `budget` genes from the end it is entered by, within the max flank distance, that no walk of this locus
        has reported. Returns them and the budget they use up, which counts genes reached back across a circular
        origin that are then dropped.
        """
        genes = self.genes.get(contig)
        if genes is None:
            return [], 0
        n = len(genes)
        forward = strand == Strand.FORWARD
        first_n = np.arange(0, min(n, budget)) if forward else np.arange(max(0, n - budget), n)[::-1]
        length = self.graph.contig_lengths[contig]
        # Where each gene starts once the contig is read along the walk, relative to the target's end
        starts = (genes.starts[first_n] if forward else length - genes.ends[first_n]) + start
        if self.max_distance is not None:
            keep = starts <= self.max_distance
            first_n, starts = first_n[keep], starts[keep]
        original_indices = cast(np.ndarray, genes.original_indices)
        unclaimed = [(int(i), int(s)) for i, s in zip(first_n, starts)
                     if (contig, int(original_indices[i])) not in claimed]
        across_origin = contig == side.contig and contig in self.graph.circular

        found = []
        for i, gene_start in unclaimed:
            index = int(original_indices[i])
            claimed.add((contig, index))
            if across_origin and not _before_target(genes, i, side):
                continue
            context, gap = side.context, gene_start
            if gap < 0:
                # A gene that runs back across the target's end, such as one starting in the next contig's overlap
                # (#64): upstream crosses the target's start and downstream its end, as Interval.relate gives
                context = Context.OVERLAPPING_START if side.context == Context.UPSTREAM else Context.OVERLAPPING_END
                gap = 0
            gene_strand = Strand(genes.strands[i] if forward else -genes.strands[i])
            found.append(FlankCandidate(contig, index, context, gap, hops, gene_strand))
        return found, len(unclaimed)


# Functions ------------------------------------------------------------------------------------------------------------
def _read_forward(strand: Strand) -> bool:
    """Whether a target on `strand` is read along its contig; ValueError if it has no strand."""
    if strand == Strand.UNSTRANDED:
        raise ValueError("A target span needs a strand to tell its upstream from its downstream")
    return strand == Strand.FORWARD


def _before_target(genes: IntervalBatch, i: int, side: _Side) -> bool:
    """
    Whether gene `i`, reached back on the target's own contig across a circular origin, lies wholly before the target
    in the walk's direction. Going any further round the circle would reach the target itself. On a circle with few
    genes, a gene can still flank the target on both sides.
    """
    if side.direction == 1:
        return bool(genes.ends[i] <= side.target.start)
    return bool(genes.starts[i] >= side.target.end)
