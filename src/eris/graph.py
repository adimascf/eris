from typing import NamedTuple, Iterable, Mapping, Callable, TYPE_CHECKING
from collections import defaultdict
from functools import partial

import numpy as np

from eris.interval import Strand
from eris.alignment import AlignmentBatch, AlignmentRecord

if TYPE_CHECKING:
    from eris.io import GenomeAssembly

OVERLAP_SLACK = 50  # bp a fragment may stop short of, or overlap, a link beyond the link's own overlap
STRICT_SLACK = 10  # bp a hub contig's or branch's target may miss an end by: a false reach pulls in other sequence (#85)


# Classes --------------------------------------------------------------------------------------------------------------
class Edge(NamedTuple):
    """
    Represents a directed link between two contigs in an assembly graph.

    Attributes:
        u: Source contig name.
        u_strand: Strand of the source contig (Strand.FORWARD or Strand.REVERSE).
        v: Destination contig name.
        v_strand: Strand of the destination contig.
        overlap: Length of the sequence overlap between the two contigs in base pairs.
    """
    u: str
    u_strand: Strand
    v: str
    v_strand: Strand
    overlap: int = 0

    def reverse(self) -> 'Edge':
        """Returns a new Edge object representing the reverse traversal (v -> u)."""
        return Edge(self.v, self.v_strand, self.u, self.u_strand, self.overlap)

    def complement(self) -> 'Edge':
        """Returns the same link read on the opposite strands, e.g. 'A+ -> B-' becomes 'B+ -> A-'."""
        return Edge(self.v, Strand(-self.v_strand), self.u, Strand(-self.u_strand), self.overlap)


def is_self_link(edge: Edge) -> bool:
    """Returns whether the link joins a contig's end back to its own start: 'A+ -> A+', or the same as 'A- -> A-'."""
    return edge.u == edge.v and edge.u_strand == edge.v_strand


def circular_contigs(edges: Iterable[Edge]) -> frozenset[str]:
    """
    Contigs whose only link is a self-link (ADR 0001). A self-link alongside links to other contigs is a collapsed
    repeat, not a closed replicon, and a hairpin ('A+ -> A-') joins an end to itself, not to the start.
    """
    self_linked: set[str] = set()
    linked_elsewhere: set[str] = set()
    for edge in edges:
        if is_self_link(edge):
            self_linked.add(edge.u)
        else:
            linked_elsewhere.update((edge.u, edge.v))
    return frozenset(self_linked - linked_elsewhere)


class AssemblyGraph:
    """
    The assembly graph: the contigs of an assembly and the links between their ends. It is built once per assembly and
    shared by stitching, repeat hubs, the flank search and copy number's anchor walk, each of which runs its own walk
    over it (docs/adr/0008).

    A GFA link is written once but holds on both strands, so every link is also read as its complement. The links
    leaving a contig come back in the order they were read, each link before the complements: that order decides which
    flanks are reported.
    """
    __slots__ = ('contig_lengths', 'circular', '_out', '_exit', '_overlap')

    def __init__(self, edges: Iterable[Edge], contig_lengths: Mapping[str, int], circular: Iterable[str] = ()):
        """
        Args:
            edges: The links, each written once.
            contig_lengths: Each contig's length.
            circular: Circular contigs, each closed by a self-link among `edges` (docs/adr/0001).
        """
        edges = list(edges)
        self.contig_lengths: Mapping[str, int] = contig_lengths
        self.circular: frozenset[str] = frozenset(circular)
        # Links are kept in dicts (ordered sets): a set's order depends on the string hash seed, which made traversal,
        # and so the reported flanks, change between runs on the same assembly
        self._out: dict[tuple[str, Strand], dict[Edge, None]] = defaultdict(dict)
        for e in edges + [e.complement() for e in edges]:
            self._out[(e.u, e.u_strand)][e] = None
        # A link leaves `u` by its end on '+' or its start on '-', and enters `v` by the other end; its complement
        # leaves `v` by that end, so keying each link and its complement by where it leaves covers both
        self._exit: dict[tuple[str, Strand], int] = defaultdict(int)
        self._overlap: dict[Edge, int] = {}
        for e in edges:
            for key in ((e.u, e.u_strand), (e.v, Strand(-e.v_strand))):
                self._exit[key] = max(self._exit[key], e.overlap)
            for side in (e, e.complement()):
                self._overlap.setdefault(side._replace(overlap=0), e.overlap)

    @classmethod
    def from_genome(cls, genome: 'GenomeAssembly') -> 'AssemblyGraph':
        """The assembly graph of a genome assembly, from its links, contig lengths and circular contigs."""
        return cls(genome.edges, genome.contig_lengths, genome.circular)

    def links_from(self, contig: str, strand: Strand) -> list[Edge]:
        """
        The links leaving `contig` read on `strand`, so by its end or, on '-', its start, in the order they were read,
        each link's complement included.
        """
        return list(self._out.get((contig, strand), ()))

    def overlap(self, u: str, u_strand: Strand, v: str, v_strand: Strand) -> int:
        """The overlap of the link 'u -> v' read on the given strands, either way it was written, or 0 if none."""
        return self._overlap.get(Edge(u, u_strand, v, v_strand), 0)

    def exit_overlap(self, contig: str, strand: Strand) -> int:
        """
        The largest overlap of a link leaving `contig` read on `strand`, so by its end or, on '-', its start; 0 at a
        dead end. An alignment may stop short of that end by this, plus a slack, and still reach it (#85).
        """
        return self._exit.get((contig, strand), 0)

    def entry_overlap(self, contig: str, strand: Strand) -> int:
        """The largest overlap of a link entering `contig` read on `strand`, so by its start or, on '-', its end."""
        return self.exit_overlap(contig, Strand(-strand))

    def reaches_both(self, contig: str, start: int, end: int, slack: int) -> bool:
        """True if `start` to `end` on `contig` reaches each of its ends within its links' overlap plus `slack`."""
        return (start <= self.exit_overlap(contig, Strand.REVERSE) + slack
                and self.contig_lengths[contig] - end <= self.exit_overlap(contig, Strand.FORWARD) + slack)


class TopologyEngine:
    """
    Engine for traversing assembly graphs and resolving complex alignments.

    It provides tools for "stitching" together partial alignments that span
    multiple contigs by finding valid physical paths through the graph.
    """
    __slots__ = ('graph', 'contig_depths', 'mode')

    def __init__(self, graph: AssemblyGraph, contig_depths: dict[str, float], mode: str = 'variant'):
        """
        Initialize the TopologyEngine.

        Args:
            graph: The assembly graph.
            contig_depths: Dictionary mapping contig ID to read depth.
            mode: 'variant' or 'collapse'.
        """
        self.graph: AssemblyGraph = graph
        self.contig_depths: dict[str, float] = contig_depths
        self.mode: str = mode

    def resolve_split_alignments(self, alignments: dict) -> tuple[dict, list[list['AlignmentRecord']]]:
        """
        Stitch together partial alignments that span multiple graph nodes.

        Uses the assembly graph to find valid paths between fragments of the
        same query that are mapped to different contigs.

        Args:
            alignments: Dictionary mapping contig ID to AlignmentBatch.

        Returns:
            tuple: (cleaned_alignments, resolved_paths)
                - cleaned_alignments: Original alignments minus those stitched.
                - resolved_paths: Lists of AlignmentRecords forming stitched paths.
        """
        partial_alns = defaultdict(list)

        # 1. Gather all partial fragments (Do NOT remove them from the main pipeline yet)
        for contig_id, batch in alignments.items():
            for i in range(len(batch)):
                rec = batch.get_record(i)
                if rec.is_partial:
                    partial_alns[rec.q_name].append(rec)

        resolved_paths = []
        used_records = set()  # Tracks (t_name, idx) of fragments successfully stitched

        # 2. Fragment chaining: the best chain from each fragment, searching each pair of fragments once
        for q_name, fragments in partial_alns.items():
            # Sort strictly by 5' -> 3' query coordinates
            fragments.sort(key=lambda x: x.q_start)
            used_in_this_qname = set()

            # Shortcuts for _fits, to skip the graph search, by the largest link overlap at each end: a fragment that
            # doesn't reach its contig end by it reaches it by no link's, and as a link never overlaps more than either
            # contig's length, no path overlaps itself by more either
            exit_overlaps = [self.graph.exit_overlap(f.t_name, f.strand) for f in fragments]
            entry_overlaps = [self.graph.entry_overlap(f.t_name, f.strand) for f in fragments]
            can_leave = [f.hangs_3p(o + OVERLAP_SLACK) for f, o in zip(fragments, exit_overlaps)]
            can_enter = [f.hangs_5p(o + OVERLAP_SLACK) for f, o in zip(fragments, entry_overlaps)]

            # The synthetic nodes joining fragment a to fragment b, or None if the graph doesn't join them. Each pair
            # is searched once: a tangled repeat shreds a target into dozens of fragments with exponentially many
            # chains between them, and searching per chain never finishes (BPH2947)
            joins = {}

            def join(a: int, b: int) -> list['AlignmentRecord'] | None:
                if (a, b) not in joins:
                    joins[(a, b)] = self._join_fragments(fragments[a], fragments[b])
                return joins[(a, b)]

            def best_chain_from(a: int, best: dict, visiting: set) -> tuple[tuple[int, int], list, set[int]]:
                """The chain from fragment a reaching furthest along the target, then through the most fragments.

                Each fragment's best chain is found once and reused by every chain reaching it. A step back into a
                fragment still being searched would loop, so it is skipped; the graph between fragments of one
                target is almost always acyclic, and then this is the best chain over all of them. On a tie, the
                later fragment in target order wins.
                """
                if a in best:
                    return best[a]
                visiting.add(a)
                curr_frag = fragments[a]
                result = ((curr_frag.q_end, 1), [curr_frag], {a})
                for b in reversed(range(len(fragments))):
                    next_frag = fragments[b]
                    if b == a or b in used_in_this_qname or b in visiting:
                        continue
                    # Stitching joins contigs; two fragments on one contig at the right spacing are the inverted
                    # repeats of a cross-mapping relative, and the graph search would call them adjacent
                    if next_frag.t_name == curr_frag.t_name:
                        continue
                    expected_gap = next_frag.q_start - curr_frag.q_end
                    if not (can_leave[a] and can_enter[b] and
                            expected_gap >= -min(exit_overlaps[a], entry_overlaps[b]) - OVERLAP_SLACK):
                        continue
                    if (between := join(a, b)) is None:
                        continue
                    (reach, n_frags), chain, used = best_chain_from(b, best, visiting)
                    # On equal coverage, prefer the chain through more real fragments over one that
                    # jumps past them and turns them into synthetic nodes
                    if a not in used and (reach, n_frags + 1) > result[0]:
                        result = ((reach, n_frags + 1), [curr_frag, *between, *chain], {a} | used)
                visiting.discard(a)
                best[a] = result
                return result

            for i in range(len(fragments)):
                if i in used_in_this_qname:
                    continue

                # Chains found from earlier starts use up their fragments, which changes later best chains
                _, best_chain, best_chain_used = best_chain_from(i, {}, set())

                # If the search successfully chained multiple fragments together
                if len(best_chain_used) > 1:

                    # Temporarily store all valid chains for this specific IS
                    if 'qname_chains' not in locals(): qname_chains = []
                    qname_chains.append(best_chain)
                    used_in_this_qname.update(best_chain_used)

            if 'qname_chains' in locals() and qname_chains:
                if self.mode == 'collapse':
                    # Pick the single chain that covers the most IS sequence
                    dominant_chain = max(qname_chains, key=lambda c: c[-1].q_end - c[0].q_start)
                    resolved_paths.append(dominant_chain)

                    # Consume ALL fragments from ALL chains to ensure the alternate
                    # bubble flanks don't accidentally get processed as unstitched debris
                    for chain in qname_chains:
                        self._consume_chain(chain, alignments, used_records)

                else:
                    # VARIANT MODE: Keep all resolved structural paths
                    for chain in qname_chains:
                        resolved_paths.append(chain)
                        self._consume_chain(chain, alignments, used_records)

                del qname_chains # Clean up for the next IS target

        # 3. Rebuild cleaned alignments (The Safety Net)
        cleaned_alignments = {}
        for contig_id, batch in alignments.items():
            # Keep everything EXCEPT the fragments that were successfully stitched!
            mask = np.zeros(len(batch), dtype=bool)
            for i in range(len(batch)):
                if (contig_id, i) not in used_records:
                    mask[i] = True

            intact_batch = batch.filter(mask)
            if len(intact_batch) > 0:
                cleaned_alignments[contig_id] = intact_batch

        return cleaned_alignments, resolved_paths

    def _join_fragments(self, curr_frag: 'AlignmentRecord',
                        next_frag: 'AlignmentRecord') -> list['AlignmentRecord'] | None:
        """The synthetic nodes for the best graph path from one fragment to the next, or None if no path scores."""
        expected_gap = next_frag.q_start - curr_frag.q_end
        # Ask the graph if these two fragments physically connect
        fits = partial(self._fits, curr_frag, next_frag, expected_gap)
        search = self._find_collapsed_path if self.mode == 'collapse' else self._find_bounded_paths
        paths = search(curr_frag.t_name, curr_frag.strand, next_frag.t_name, next_frag.strand,
                       expected_gap, tolerance=2000, fits=fits)
        if not paths:
            return None
        path_lengths = np.array([p['length'] for p in paths])
        bottleneck_depths = np.array([p['min_depth'] for p in paths])
        source_depth = self.contig_depths.get(curr_frag.t_name, 1.0)

        # CRITICAL FIX: Safe, absolute normalization for negative gaps
        norm_factor = np.abs(expected_gap) + 50
        len_penalty = np.maximum(1.0 - (np.abs(path_lengths - expected_gap) / norm_factor), 0)

        depth_fraction = bottleneck_depths / source_depth
        scores = len_penalty * depth_fraction

        best_p_idx = np.argmax(scores)
        if scores[best_p_idx] <= 0.05:  # No biological winner
            return None
        winning = paths[best_p_idx]
        return self._build_stitching_payload(curr_frag, next_frag, winning['contigs'], winning['strands'])[1:-1]

    @staticmethod
    def _consume_chain(chain: list['AlignmentRecord'], alignments: dict, used_records: set) -> None:
        """Marks every real alignment that a stitched chain accounts for, so it is not reported again as local."""
        for f in chain:
            if f.idx != -1:
                used_records.add((f.t_name, f.idx))
            elif (batch := alignments.get(f.t_name)) is not None:
                # Synthetic nodes claim a whole intermediate contig, so any alignment on it belongs to this path
                inside = (batch.t_starts >= f.t_start) & (batch.t_ends <= f.t_end)
                used_records.update((f.t_name, int(i)) for i in np.flatnonzero(inside))

    @staticmethod
    def _fits(u: 'AlignmentRecord', v: 'AlignmentRecord', expected_gap: int, path: dict) -> bool:
        """
        True if fragments u and v can sit at either end of the contig path, give or take the links' overlaps.

        The path length counts neither end contig, so u must run off the end of its contig the path leaves by, within
        the first link's overlap, and v off the end it enters by, within the last link's; else the rest of that contig
        lands in the locus uncounted. u and v may overlap on the query by no more than the path overlaps itself: the
        link's overlap for two linked contigs, nothing across a contig longer than the links joining it.
        """
        return (u.hangs_3p(path['overlaps'][0] + OVERLAP_SLACK) and v.hangs_5p(path['overlaps'][-1] + OVERLAP_SLACK)
                and expected_gap >= min(path['length'], 0) - OVERLAP_SLACK)

    def _build_stitching_payload(self, h_u: 'AlignmentRecord', h_v: 'AlignmentRecord', path_contigs: list[str],
                                 path_strands: list[Strand]) -> list['AlignmentRecord']:
        """
        Converts a list of graph contig names into a continuous sequence of AlignmentRecords.

        Generates 'synthetic' records for unaligned intermediate nodes so they
        can be processed by the LocusBuilder as part of a single path. Each synthetic record
        takes the strand its contig was entered on, as read in the target's direction.
        """
        payload = [h_u]

        # Iterate over only the intermediate contigs (excluding the h_u and h_v anchors)
        for ctg, strand in zip(path_contigs[1:-1], path_strands[1:-1]):
            ctg_len = self.graph.contig_lengths.get(ctg, 0)

            # Create a synthetic alignment that claims the entire unaligned contig
            synthetic_rec = AlignmentRecord(
                idx=-1,  # Flag as a synthetic/mock record
                q_name=h_u.q_name,
                q_length=h_u.q_length,
                q_start=h_u.q_end,  # Conceptually sits between the anchors
                q_end=h_v.q_start,
                t_name=ctg,
                t_length=ctg_len,
                t_start=0,  # The entire contig is part of the path
                t_end=ctg_len,
                strand=strand,
                length=ctg_len,
                match=0,
                mismatch=0,
                quality=0,
                cigar="*"  # No CIGAR exists for synthetic nodes
            )
            payload.append(synthetic_rec)

        payload.append(h_v)

        return payload

    def _find_bounded_paths(self, start_ctg: str, start_strand: Strand, target_ctg: str,
                            target_strand: Strand, expected_len: int, tolerance: int,
                            fits: Callable[[dict], bool] = lambda path: True) -> list[dict]:
        """
        Finds all physical paths between two contigs within a length constraint.

        Args:
            start_ctg: Starting contig ID.
            start_strand: Strand to exit the start contig from.
            target_ctg: Target contig ID.
            target_strand: Required strand to enter the target contig.
            expected_len: The gap distance observed in the query sequence.
            tolerance: bp tolerance for the path length matching the expected length.
            fits: Whether a path that reaches the target is accepted.

        Returns:
            list[dict]: Valid paths found, with 'contigs', 'strands' (each contig's entry strand), 'overlaps' (of each
                link crossed), 'length', and 'min_depth'.
        """
        # Stack payload: (current_contig, exit_strand, path_list, strand_list, overlap_list, accumulated_len,
        # bottleneck_depth)
        stack: list[tuple[str, Strand, list[str], list[Strand], list[int], int, float]] = [
            (start_ctg, start_strand, [start_ctg], [start_strand], [], 0, float('inf'))]
        valid_paths = []

        while stack:
            curr_ctg, curr_strand, path, strands, overlaps, dist, min_dp = stack.pop()

            # Base Case: Reached the Sink anchor
            if curr_ctg == target_ctg:
                found = {'contigs': path, 'strands': strands, 'overlaps': overlaps, 'length': dist, 'min_depth': min_dp}
                if curr_strand == target_strand and fits(found):  # Arrived on the right strand, at fitting fragments?
                    valid_paths.append(found)
                continue

            # Prune: We have wandered too far down a dead end
            if dist > expected_len + tolerance:
                continue

            # Graph Traversal
            for edge in self.graph.links_from(curr_ctg, curr_strand):
                n_ctg = edge.v
                if n_ctg in path:
                    continue  # Prevent cyclic infinite loops

                n_len = self.graph.contig_lengths.get(n_ctg, 0)
                n_dp = self.contig_depths.get(n_ctg, 1.0)
                overlap_len = getattr(edge, 'overlap', 0)

                # CRITICAL FIX: If this is the target anchor, its length doesn't belong in the gap!
                added_dist = (n_len - overlap_len) if n_ctg != target_ctg else -overlap_len

                stack.append((
                    n_ctg,
                    edge.v_strand,
                    path + [n_ctg],
                    strands + [edge.v_strand],
                    overlaps + [overlap_len],
                    dist + added_dist,
                    min(min_dp, n_dp)
                ))

        return valid_paths

    def _find_collapsed_path(self, start_ctg: str, start_strand: Strand, target_ctg: str,
                             target_strand: Strand, expected_len: int, tolerance: int,
                             fits: Callable[[dict], bool] = lambda path: True) -> list[dict]:
        """
        Greedy pathfinder for 'collapse' mode. Always traverses the highest depth neighbor option if finds multiple branchs,
        and backtracks to the next deepest when that branch cannot reach the target, or reaches it on a path that
        `fits` rejects.
        """
        # Stack payload: (current_contig, exit_strand, path_list, strand_list, overlap_list, accumulated_len, peak_depth)
        start_dp = float(self.contig_depths.get(start_ctg, 1.0))
        stack: list[tuple[str, Strand, list[str], list[Strand], list[int], int, float]] = [
            (start_ctg, start_strand, [start_ctg], [start_strand], [], 0, start_dp)]

        while stack:
            curr_ctg, curr_strand, path, strands, overlaps, dist, max_dp = stack.pop()

            # Base Case: Reached the Sink anchor
            if curr_ctg == target_ctg:
                found = {'contigs': path, 'strands': strands, 'overlaps': overlaps, 'length': dist, 'min_depth': max_dp}
                if curr_strand == target_strand and fits(found):
                    return [found]
                continue

            # Prune: We have wandered too far down a dead end
            if dist > expected_len + tolerance:
                continue

            # Greedy Graph Traversal: push the shallowest first so the deepest valid neighbor is popped next
            neighbors = [e for e in self.graph.links_from(curr_ctg, curr_strand)
                         if e.v not in path]  # Prevent cyclic infinite loops
            # Sort on the name too, as set order is not stable between runs
            neighbors.sort(key=lambda e: (self.contig_depths.get(e.v, 1.0), e.v))

            for edge in neighbors:
                n_ctg = edge.v
                n_dp = self.contig_depths.get(n_ctg, 1.0)
                n_len = self.graph.contig_lengths.get(n_ctg, 0)
                overlap_len = getattr(edge, 'overlap', 0)

                # Absolute normalization for target anchor
                added_dist = (n_len - overlap_len) if n_ctg != target_ctg else -overlap_len

                stack.append((
                    n_ctg,
                    edge.v_strand,
                    path + [n_ctg],
                    strands + [edge.v_strand],
                    overlaps + [overlap_len],
                    dist + added_dist,
                    max(max_dp, n_dp) # Track the peak depth of the collapsed hub
                ))

        # If it reaches a dead end before finding the target anchor
        return []
