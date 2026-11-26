"""
Repeat hubs: the contigs one collapsed repeat covers end to end (docs/adr/0003, 0008).

Flank branches and copy number both start from a locus's repeat hub. They share how a hub is found, how hubs that share
a contig are merged, and which links leave it; each keeps its own walk out of it.
"""
from dataclasses import dataclass
from typing import Iterable, Iterator, TYPE_CHECKING

from eris.graph import STRICT_SLACK, Edge, EndOverlaps, Graph
from eris.interval import Strand

if TYPE_CHECKING:
    from eris.alignment import AlignmentRecord
    from eris.io import GenomeAssembly
    from eris.pipeline import Locus

# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RepeatHub:
    """
    A repeat hub: the contigs a target covers end to end, each with the strand it is first read on, in path order. Empty
    for a target in unique sequence.
    """
    contigs: tuple[tuple[str, Strand], ...] = ()

    @property
    def names(self) -> frozenset[str]:
        """The hub's contigs, without their strands."""
        return frozenset(c for c, _ in self.contigs)

    def __bool__(self) -> bool:
        return bool(self.contigs)

    def __iter__(self) -> Iterator[str]:
        return (c for c, _ in self.contigs)

    def __contains__(self, contig: object) -> bool:
        return any(c == contig for c, _ in self.contigs)


class Hubs:
    """
    Finds the repeat hubs of an assembly's stitched paths and loci, merges those that share a contig, and gives the
    links that leave them.
    """
    __slots__ = ('graph', 'contig_lengths', '_end_overlaps')

    def __init__(self, edges: list[Edge], contig_lengths: dict[str, int]):
        # A GFA link is written once but holds on both strands, so add its complement for walking the other way
        self.graph = Graph(edges + [e.complement() for e in edges])
        self.contig_lengths = contig_lengths
        self._end_overlaps = EndOverlaps(edges)

    @classmethod
    def from_genome(cls, genome: 'GenomeAssembly') -> 'Hubs':
        """The hubs of an assembly, from its links and contig lengths."""
        return cls(list(genome.edges), genome.contig_lengths)

    def _covers(self, contig: str, start: int, end: int) -> bool:
        """True if `start` to `end` covers `contig` end to end: each end within its links' overlap plus STRICT_SLACK."""
        return self._end_overlaps.reaches_both(contig, self.contig_lengths[contig], start, end, STRICT_SLACK)

    @staticmethod
    def _hub(covered: Iterable[tuple[str, Strand]]) -> RepeatHub:
        """A hub of the contigs in `covered`, in order, each on the first strand it comes with."""
        contigs: dict[str, Strand] = {}
        for contig, strand in covered:
            contigs.setdefault(contig, strand)
        return RepeatHub(tuple(contigs.items()))

    def of_path(self, records: list['AlignmentRecord']) -> RepeatHub:
        """The repeat hub of a stitched path: the contigs its records cover end to end, in the target's direction."""
        return self._hub((f.t_name, Strand(f.strand)) for f in records
                         if self._covers(f.t_name, f.t_start, f.t_end))

    def of_locus(self, locus: 'Locus') -> RepeatHub:
        """
        The repeat hub of a locus: the contigs its targets cover end to end. A contig with sequence beyond the target is
        unique sequence, not part of it.
        """
        return self._hub((seg.contig, seg.strand) for target in locus.targets for seg in target.segments
                         if self._covers(seg.contig, seg.start, seg.end))

    @classmethod
    def group(cls, hubs: list[RepeatHub]) -> list[tuple[RepeatHub, list[int]]]:
        """
        Merges hubs that share a contig, directly or through another hub, so near-identical targets walking through one
        collapsed repeat (#14) are one. Returns each merged hub with its members' indices into `hubs`; an empty hub
        stands alone. A merged group goes to the end, its members after those of the groups merged into it, the
        latest of those first. A contig keeps the strand of the first hub it was seen in.
        """
        merged: list[tuple[RepeatHub, list[int]]] = []
        for i, hub in enumerate(hubs):
            names = hub.names
            covered: list[tuple[str, Strand]] = []
            members: list[int] = []
            kept = []
            for other, other_members in merged:
                if other.names & names:
                    covered += other.contigs
                    members = other_members + members
                else:
                    kept.append((other, other_members))
            merged = kept + [(cls._hub(covered + list(hub.contigs)), members + [i])]
        return merged

    def _links_out(self, contig: str, strand: Strand, side: int, hub: RepeatHub) -> list[Edge]:
        """The links out of `contig`, read on `strand`, on `side` (see `leaving`) to a contig outside `hub`."""
        return [e for e in self.graph.get_neighbors(contig) if e.u_strand == strand * side and e.v not in hub]

    def leaving(self, hub: RepeatHub, side: int) -> list[Edge]:
        """
        The links that leave a repeat hub on one side, out of every contig in it: `side` 1 for the side the hub's
        contigs are read out of on their strands, -1 for the side they are read into. A link leaves from the end on
        that side of one of its contigs, so following edge orientation from it never turns back towards the hub, and
        goes to a contig outside the hub. Copy number walks out by these: each copy collapsed into the hub reaches an
        anchor by one of them, wherever in the hub it leaves. Flank branches use `leaving_outermost` instead.
        """
        return [e for contig, strand in hub.contigs for e in self._links_out(contig, strand, side, hub)]

    def leaving_outermost(self, hub: RepeatHub, side: int) -> list[Edge]:
        """
        The links that leave a repeat hub on one side (see `leaving`), out of its outermost contig on that side only:
        the first for `side` -1, the last for 1. Flank branches leave by these, where copy number uses the links out of
        every contig in the hub: a link out of the middle of the hub enters one copy's stub of a target variant, so
        its genes would be false flanks (#56).
        """
        if not hub:
            return []
        contig, strand = hub.contigs[0 if side < 0 else -1]
        return self._links_out(contig, strand, side, hub)
