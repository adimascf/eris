"""
Locus ends: where each side of a locus's target ends, and the branches a repeat hub's copies leave by. The flank search
starts from them (see `eris.flanks.TargetSpan` and `eris.flanks.HubBranches`).
"""
from collections.abc import Iterable, Mapping
from typing import NamedTuple, Optional

from eris.alignment import AlignmentBatch, AlignmentRecord
from eris.constants import Context, Strand
from eris.flanks import Branch, HubBranches, TargetSpan
from eris.graph import OVERLAP_SLACK, STRICT_SLACK, AssemblyGraph, Edge
from eris.hub import Hubs


# Classes --------------------------------------------------------------------------------------------------------------
class Ends(NamedTuple):
    """
    A locus's ends, as the flank search takes them.

    Attributes:
        upstream: The target span the upstream flanks are searched from; its `beyond_upstream` is set.
        downstream: The target span the downstream flanks are searched from; its `beyond_downstream` is set. For a local
            locus it is `upstream`.
        branches: The branches of a collapse-mode stitched locus's repeat hub; None if it has none.
    """
    upstream: TargetSpan
    downstream: TargetSpan
    branches: Optional[HubBranches] = None


class LocusEnds:
    """
    Works out the ends of the loci of one run.

    Args:
        graph: The assembly graph.
        hubs: The repeat hubs of `graph`.
        alignments: Each contig's real alignments, before stitching: synthetic stitch records overshoot by one overlap.
        hub_branches: Whether a stitched locus's repeat hub has branches, as in collapse mode, where one locus stands
            for every copy collapsed into its hub. In variant mode each path through a hub is its own locus.
    """
    def __init__(self, graph: AssemblyGraph, hubs: Hubs, alignments: Mapping[str, AlignmentBatch],
                 hub_branches: bool):
        self._graph = graph
        self._hubs = hubs
        self._alignments = alignments
        self._hub_branches = hub_branches

    def local(self, records: list[AlignmentRecord]) -> Ends:
        """
        The ends of a local locus: the span of its IS copy's alignments on one contig, read on its first target's
        strand. Both sides run on past the contig end by what the alignment nearest it leaves unaligned (see
        `_beyond_contig_end`).
        """
        strand = Strand(records[0].strand)
        start, end = int(min(r.t_start for r in records)), int(max(r.t_end for r in records))
        span = TargetSpan(records[0].t_name, start, end, strand,
                          _beyond_contig_end(records, strand, Context.UPSTREAM, self._graph),
                          _beyond_contig_end(records, strand, Context.DOWNSTREAM, self._graph))
        return Ends(span, span)

    def stitched(self, fragments: list[AlignmentRecord]) -> Ends:
        """
        The ends of a stitched locus: upstream from its first fragment, downstream from its last, each read on its own
        strand, plus its repeat hub's branches.
        """
        first, last = fragments[0], fragments[-1]
        upstream = TargetSpan(first.t_name, first.t_start, first.t_end, Strand(first.strand),
                              beyond_upstream=_beyond_contig_end([first], Strand(first.strand), Context.UPSTREAM,
                                                                 self._graph))
        downstream = TargetSpan(last.t_name, last.t_start, last.t_end, Strand(last.strand),
                                beyond_downstream=_beyond_contig_end([last], Strand(last.strand), Context.DOWNSTREAM,
                                                                     self._graph))
        return Ends(upstream, downstream, self._branches(fragments))

    def _branches(self, fragments: list[AlignmentRecord]) -> Optional[HubBranches]:
        """
        The branches of a stitched path's repeat hub (see `_branch_links`), each with the target its copy carries onto
        the branch (see `_carried_past_link`); None if it has none. Once a hub's branches are walked, none may loop
        back into the stitched path: its genes are passengers.
        """
        u_links, d_links = self._branch_links(fragments)
        if not (u_links or d_links):
            return None
        lengths = self._graph.contig_lengths
        return HubBranches(tuple(Branch(e, _carried_past_link(e, self._alignments, lengths)) for e in u_links),
                           tuple(Branch(e, _carried_past_link(e, self._alignments, lengths)) for e in d_links),
                           frozenset(f.t_name for f in fragments))

    def _branch_links(self, fragments: list[AlignmentRecord]) -> tuple[list[Edge], list[Edge]]:
        """
        The links leaving a stitched path's repeat hub upstream and downstream, other than those the flank search from
        the path's own end contig already follows (#56). A hub reported once stands for every copy collapsed into it,
        so each of its links is a branch with its own flanks. Only the links leaving the hub's outermost contig on each
        side count (see `Hubs.leaving_outermost`). None without hub branches.
        """
        if not self._hub_branches:
            return [], []
        hub = self._hubs.of_path(fragments)
        if not hub:
            return [], []
        # fragments are in the target's direction, so the hub's first contig is upstream
        (first, _), (last, _) = hub.contigs[0], hub.contigs[-1]
        links_by_side: list[list[Edge]] = []
        for side, end, outer in ((-1, fragments[0], first), (1, fragments[-1], last)):
            if end.t_name == outer:  # the walk starts on the hub's edge, so it already leaves by every link
                links_by_side.append([])
                continue
            # The search from the end contig follows the branch along the link into it
            links_by_side.append([e for e in self._hubs.leaving_outermost(hub, side)
                                  if (e.v, e.v_strand) != (end.t_name, end.strand * side)])
        return links_by_side[0], links_by_side[1]


# Functions ------------------------------------------------------------------------------------------------------------
def _beyond_contig_end(records: Iterable[AlignmentRecord], strand: Strand, side: Context,
                       graph: AssemblyGraph) -> int:
    """
    How much of a locus's target runs on past the contig end a flank search leaves by (#66): the target bases an
    alignment leaves unaligned at that end, when it reaches the end within the largest overlap of a link leaving it
    plus OVERLAP_SLACK and those bases are more than the contig has left, so the rest of the copy is on the next
    contig. Of several alignments that do, the one nearest the end counts. 0 if none does: an alignment that stops
    inside the contig, or short of it by more than its unaligned bases, is a truncated copy, whose missing bases are
    nowhere. Synthetic stitch records are left out: they overshoot by one overlap.

    Args:
        records: The locus's alignments on the contig.
        strand: The strand the locus is read on, which sets which contig end is upstream.
        side: Context.UPSTREAM or Context.DOWNSTREAM, the side the flank search leaves by.
        graph: The assembly graph, for the overlaps of the links leaving the contig.
    """
    walk_direction = strand if side == Context.DOWNSTREAM else -strand
    exit_strand = Strand.FORWARD if walk_direction == 1 else Strand.REVERSE
    best: Optional[tuple[int, int]] = None  # (gap to the contig end, target bases past the alignment there)
    for rec in records:
        if rec.idx == -1:
            continue
        tolerance = graph.exit_overlap(rec.t_name, exit_strand) + OVERLAP_SLACK
        gap = rec.t_length - rec.t_end if walk_direction == 1 else rec.t_start
        if gap > tolerance:
            continue
        # The target's own 3' end lies past the contig end it is read towards, its 5' end past the other
        reads_towards = (rec.strand == Strand.FORWARD) == (walk_direction == 1)
        past = rec.q_length - rec.q_end if reads_towards else rec.q_start
        if past <= gap:
            continue  # the rest of the target would fit on this contig, where it does not align
        if best is None or (gap, -past) < (best[0], -best[1]):
            best = (gap, past)
    return 0 if best is None else best[1]


def _carried_past_link(link: Edge, alignments: Mapping[str, AlignmentBatch],
                       contig_lengths: Mapping[str, int]) -> int:
    """
    How far a copy's target runs onto a hub branch's first contig `link.v` past the overlap it shares with the hub
    (#65), read along the walk: the furthest end of an alignment that starts where the contig is entered, within
    STRICT_SLACK. 0 if none does. The hub's edge is target end to end, so a copy carrying on past it starts in
    the overlap; an alignment starting past it is another IS beside this one.

    Args:
        link: The link from the hub into the branch.
        alignments: Each contig's real alignments, before stitching.
        contig_lengths: Each contig's length.
    """
    batch = alignments.get(link.v)
    if batch is None or not len(batch):
        return 0
    if link.v_strand == Strand.FORWARD:
        starts, ends = batch.t_starts, batch.t_ends
    else:
        length = contig_lengths[link.v]
        starts, ends = length - batch.t_ends, length - batch.t_starts
    entering = starts <= STRICT_SLACK
    if not entering.any():
        return 0
    return max(0, int(ends[entering].max()) - link.overlap)
