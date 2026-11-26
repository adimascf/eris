"""
Where an IS copy's target ends when it runs on past the contig it aligns to: the bases the flank search measures a
flank's distance from (see `eris.flanks.TargetSpan` and `eris.flanks.Branch`).
"""
from collections.abc import Iterable, Mapping
from typing import Optional

from eris.alignment import AlignmentBatch, AlignmentRecord
from eris.constants import Context, Strand
from eris.graph import OVERLAP_SLACK, STRICT_SLACK, Edge, TopologyEngine


# Functions ------------------------------------------------------------------------------------------------------------
def beyond_contig_end(records: Iterable[AlignmentRecord], strand: Strand, side: Context,
                      topology_engine: TopologyEngine) -> int:
    """
    How much of a locus's target runs on past the contig end a flank walk leaves by (#66): the target bases an
    alignment leaves unaligned at that end, when it reaches the end within the largest overlap of a link leaving it
    plus OVERLAP_SLACK and those bases are more than the contig has left, so the rest of the copy is on the next
    contig. Of several alignments that do, the one nearest the end counts. 0 if none does: an alignment that stops
    inside the contig, or short of it by more than its unaligned bases, is a truncated copy, whose missing bases are
    nowhere. Synthetic stitch records are left out: they overshoot by one overlap.

    Args:
        records: The locus's alignments on the contig.
        strand: The strand the locus is read on, which sets which contig end is upstream.
        side: Context.UPSTREAM or Context.DOWNSTREAM, the side the flank walk leaves by.
        topology_engine: The assembly graph, for the overlaps of the links leaving the contig.
    """
    walk_direction = strand if side == Context.DOWNSTREAM else -strand
    exit_strand = Strand.FORWARD if walk_direction == 1 else Strand.REVERSE
    best: Optional[tuple[int, int]] = None  # (gap to the contig end, target bases past the alignment there)
    for rec in records:
        if rec.idx == -1:
            continue
        tolerance = topology_engine.exit_overlap(rec.t_name, exit_strand) + OVERLAP_SLACK
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


def carried_past_link(link: Edge, alignments: Mapping[str, AlignmentBatch], contig_lengths: Mapping[str, int]) -> int:
    """
    How far a copy's target runs onto a hub branch's first contig `link.v` past the overlap it shares with the hub
    (#65), read along the walk: the furthest end of an alignment that starts where the contig is entered, within
    STRICT_SLACK. 0 if none does. The hub's edge is target end to end, so a copy carrying on past it starts in
    the overlap; an alignment starting past it is another IS beside this one.

    Args:
        link: The link from the hub into the branch.
        alignments: Each contig's real alignments, before stitching: synthetic stitch records overshoot by one overlap.
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
