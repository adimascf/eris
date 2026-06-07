from typing import NamedTuple, Iterable
from collections import defaultdict

import numpy as np

from eris.interval import Strand, IntervalBatch
from eris.alignment import AlignmentBatch, AlignmentRecord


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


class Graph:
    """
    Manages an assembly graph, supporting both directed and undirected traversal.

    Nodes represent contigs (or fragments), and edges represent physical
    adjacencies (e.g., from an assembly GFA file).

    Example:
        >>> graph = Graph(directed=True)
        >>> graph.add_edge(Edge("ctg1", Strand.FORWARD, "ctg2", Strand.FORWARD, 50))
        >>> neighbors = graph.get_neighbors("ctg1")
    """
    __slots__ = ('adj', 'in_adj', 'edges', '_nodes', 'directed')
    def __init__(self, edges: Iterable[Edge] = None, directed: bool = True):
        """
        Initialize the graph.

        Args:
            edges: Optional iterable of Edge objects to seed the graph.
            directed: If True, edges are strictly one-way. If False, every
                added edge u -> v implicitly adds v -> u.
        """
        # Adjacency list: maps node ID to a set of outgoing Edge objects *starting* uom that node.
        # For undirected graphs, this will include edges representing reverse traversal.
        self.adj: dict[str, set[Edge]] = defaultdict(set)
        # In-degree adjacency list for efficient reverse lookups
        self.in_adj: dict[str, set[Edge]] = defaultdict(set)
        # Set of unique Edge objects fundamentally added to the graph.
        self.edges: set[Edge] = set()
        self._nodes: set[str] = set()
        self.directed: bool = directed
        if edges is not None:
            for edge in edges:
                self.add_edge(edge)
    
    def __repr__(self):
        # Note: len(self.edges) counts only the *unique* edge objects added,
        # not the total number of traversable connections in the undirected case.
        return f"{'Directed' if self.directed else 'Undirected'} Graph with {len(self._nodes)} nodes and {len(self.edges)} defined edges"

    def __iter__(self):
        return iter(self.edges)

    def __len__(self):
        return len(self.edges)

    def add_node(self, node: str):
        """Adds a node to the graph if it doesn't already exist."""
        self._nodes.add(node)

    def add_edge(self, edge: Edge):
        """
        Adds an edge to the graph.

        If the graph is undirected, the reverse connection is also added
        to the adjacency lists to allow bidirectional traversal.
        """
        # Add the nodes to the set of known node IDs
        self.add_node(edge.u)
        self.add_node(edge.v)

        # This helps track the originally added edges vs implicit reverse ones
        self.edges.add(edge)  # Add the primary edge representation if it's new

        # Check if this specific edge object is already in the adjacency list for the 'u' node
        self.adj[edge.u].add(edge)
        self.in_adj[edge.v].add(edge)

        # If the graph is undirected, add reverse connectivity as well
        if not self.directed:
            # Create a conceptual reverse edge for traversal and add to the adjacency list of the 'v' node
            reverse_edge = edge.reverse()
            self.in_adj[edge.u].add(reverse_edge)
            self.adj[edge.v].add(reverse_edge)
            # Note: We do not add the reverse_edge to self.edges unless it's explicitly added later by the user

    def get_neighbors(self, node_id: str) -> set[Edge]:
        """
        Returns the set of outgoing edges for a given node ID.

        Respects graph directionality. For undirected graphs, this includes
        implicit reverse connections.
        """
        return self.adj.get(node_id, set())


class TopologyEngine:
    """
    Engine for traversing assembly graphs and resolving complex alignments.

    It provides tools for "stitching" together partial alignments that span
    multiple contigs by finding valid physical paths through the graph.
    """
    __slots__ = ('_graph', 'contig_lengths', 'contig_depths', 'features', '_visited_nodes', 'mode', 'median_depth')

    def __init__(self, edges: Iterable[Edge], contig_lengths: dict[str, int], contig_depths: dict[str, float],
                 features: dict[str, IntervalBatch] = None, mode: str = 'variant', median_depth: float = 1.0):
        """
        Initialize the TopologyEngine.

        Args:
            edges: Assembly graph edges.
            contig_lengths: Dictionary mapping contig ID to length in bp.
            contig_depths: Dictionary mapping contig ID to read depth.
            features: Dictionary mapping contig ID to its annotated intervals.
        """
        self._graph = Graph(edges)  # Build the graph from parsed edges
        self.contig_lengths: dict[str, int] = contig_lengths
        self.contig_depths: dict[str, float] = contig_depths
        self.features: dict[str, IntervalBatch] = features or {}
        self._visited_nodes: set[tuple[str, int]] = set()

        self.mode: str = mode
        self.median_depth: float = median_depth


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

        # 2. Fragment Chaining via DAG DFS
        for q_name, fragments in partial_alns.items():
            # Sort strictly by 5' -> 3' query coordinates
            fragments.sort(key=lambda x: x.q_start)
            used_in_this_qname = set()

            for i in range(len(fragments)):
                if i in used_in_this_qname:
                    continue

                start_frag = fragments[i]
                # Stack: (current_frag_index, sequence_of_records, set_of_used_indices)
                stack = [(i, [start_frag], {i})]

                best_chain = []
                best_chain_len = 0
                best_chain_used = set()

                while stack:
                    curr_idx, curr_path, curr_used = stack.pop()
                    curr_frag = curr_path[-1]  # The last real alignment in the chain

                    extended = False
                    for j in range(len(fragments)):
                        if j in curr_used or j in used_in_this_qname:
                            continue

                        next_frag = fragments[j]
                        expected_gap = next_frag.q_start - curr_frag.q_end

                        # Fragments must be sequentially downstream on the query
                        if expected_gap < -50:
                            continue

                        # Ask the graph if these two fragments physically connect
                        if self.mode == 'collapse':
                            paths = self._find_collapsed_path(
                                curr_frag.t_name, curr_frag.strand,
                                next_frag.t_name, next_frag.strand,
                                expected_gap, tolerance=2000
                            )
                        else:
                            paths = self._find_bounded_paths(
                                curr_frag.t_name, curr_frag.strand,
                                next_frag.t_name, next_frag.strand,
                                expected_gap, tolerance=2000
                            )


                        if paths:
                            path_lengths = np.array([p['length'] for p in paths])
                            bottleneck_depths = np.array([p['min_depth'] for p in paths])
                            source_depth = self.contig_depths.get(curr_frag.t_name, 1.0)

                            # CRITICAL FIX: Safe, absolute normalization for negative gaps
                            norm_factor = np.abs(expected_gap) + 50
                            len_penalty = np.maximum(1.0 - (np.abs(path_lengths - expected_gap) / norm_factor), 0)

                            depth_fraction = bottleneck_depths / source_depth
                            scores = len_penalty * depth_fraction

                            best_p_idx = np.argmax(scores)
                            if scores[best_p_idx] > 0.05:  # Biological winner

                                winning_contigs = paths[best_p_idx]['contigs']
                                extension = self._build_stitching_payload(curr_frag, next_frag, winning_contigs)

                                # Extend the path with the synthetic nodes + next_frag and recurse!
                                new_path = curr_path + extension[1:]
                                stack.append((j, new_path, curr_used | {j}))
                                extended = True

                    # If we can't extend this branch further, evaluate its total coverage
                    if not extended:
                        chain_cov = curr_path[-1].q_end - curr_path[0].q_start
                        if chain_cov > best_chain_len:
                            best_chain = curr_path
                            best_chain_len = chain_cov
                            best_chain_used = curr_used

                # If the DFS successfully chained multiple fragments together
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
                        for f in chain:
                            if f.idx != -1:
                                used_records.add((f.t_name, f.idx))

                else:
                    # VARIANT MODE: Keep all resolved structural paths
                    for chain in qname_chains:
                        resolved_paths.append(chain)
                        for f in chain:
                            if f.idx != -1:
                                used_records.add((f.t_name, f.idx))

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

    def _build_stitching_payload(self, h_u: 'AlignmentRecord', h_v: 'AlignmentRecord', path_contigs: list[str]) -> list[
        'AlignmentRecord']:
        """
        Converts a list of graph contig names into a continuous sequence of AlignmentRecords.

        Generates 'synthetic' records for unaligned intermediate nodes so they
        can be processed by the LocusBuilder as part of a single path.
        """
        payload = [h_u]

        # Iterate over only the intermediate contigs (excluding the h_u and h_v anchors)
        for ctg in path_contigs[1:-1]:
            ctg_len = self.contig_lengths.get(ctg, 0)

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
                strand=Strand.FORWARD,  # Default to forward for the traversal sequence
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
                            target_strand: Strand, expected_len: int, tolerance: int) -> list[dict]:
        """
        Finds all physical paths between two contigs within a length constraint.

        Args:
            start_ctg: Starting contig ID.
            start_strand: Strand to exit the start contig from.
            target_ctg: Target contig ID.
            target_strand: Required strand to enter the target contig.
            expected_len: The gap distance observed in the query sequence.
            tolerance: bp tolerance for the path length matching the expected length.

        Returns:
            list[dict]: Valid paths found, with 'contigs', 'length', and 'min_depth'.
        """
        # Stack payload: (current_contig, exit_strand, path_list, accumulated_len, bottleneck_depth)
        stack = [(start_ctg, start_strand, [start_ctg], 0, float('inf'))]
        valid_paths = []

        while stack:
            curr_ctg, curr_strand, path, dist, min_dp = stack.pop()

            # Base Case: Reached the Sink anchor
            if curr_ctg == target_ctg:
                if curr_strand == target_strand:  # Did we arrive on the correct biological strand?
                    valid_paths.append({'contigs': path, 'length': dist, 'min_depth': min_dp})
                continue

            # Prune: We have wandered too far down a dead end
            if dist > expected_len + tolerance:
                continue

            # Graph Traversal
            for edge in self._graph.get_neighbors(curr_ctg):
                if edge.u_strand != curr_strand:
                    continue

                n_ctg = edge.v
                if n_ctg in path:
                    continue  # Prevent cyclic infinite loops

                n_len = self.contig_lengths.get(n_ctg, 0)
                n_dp = self.contig_depths.get(n_ctg, 1.0)
                overlap_len = getattr(edge, 'overlap', 0)

                # CRITICAL FIX: If this is the target anchor, its length doesn't belong in the gap!
                added_dist = (n_len - overlap_len) if n_ctg != target_ctg else -overlap_len

                stack.append((
                    n_ctg,
                    edge.v_strand,
                    path + [n_ctg],
                    dist + added_dist,
                    min(min_dp, n_dp)
                ))

        return valid_paths

    def _find_collapsed_path(self, start_ctg: str, start_strand: Strand, target_ctg: str,
                             target_strand: Strand, expected_len: int, tolerance: int) -> list[dict]:
        """
        Greedy pathfinder for 'collapse' mode. Always traverses the highest depth neighbor option if finds multiple branchs.
        """
        # Stack payload: (current_contig, exit_strand, path_list, accumulated_len, peak_depth)
        stack = [(start_ctg, start_strand, [start_ctg], 0, 0.0)]

        while stack:
            curr_ctg, curr_strand, path, dist, max_dp = stack.pop()

            # Base Case: Reached the Sink anchor
            if curr_ctg == target_ctg:
                if curr_strand == target_strand:
                    return [{'contigs': path, 'length': dist, 'min_depth': max_dp}] 
                continue

            # Prune: We have wandered too far down a dead end
            if dist > expected_len + tolerance:
                continue

            # Greedy Graph Traversal: Find the single deepest valid neighbor
            deepest_neighbor = None
            highest_depth = -1

            for edge in self._graph.get_neighbors(curr_ctg):
                if edge.u_strand != curr_strand: continue
                if edge.v in path: continue # Prevent cyclic infinite loops

                n_dp = self.contig_depths.get(edge.v, 1.0)
                if n_dp > highest_depth:
                    highest_depth = n_dp
                    deepest_neighbor = edge

            if deepest_neighbor:
                n_ctg = deepest_neighbor.v
                n_len = self.contig_lengths.get(n_ctg, 0)
                overlap_len = getattr(deepest_neighbor, 'overlap', 0)

                # Absolute normalization for target anchor
                added_dist = (n_len - overlap_len) if n_ctg != target_ctg else -overlap_len

                stack.append((
                    n_ctg,
                    deepest_neighbor.v_strand,
                    path + [n_ctg],
                    dist + added_dist,
                    max(max_dp, highest_depth) # Track the peak depth of the collapsed hub
                ))

        # If it reaches a dead end before finding the target anchor
        return []

    def traverse(self, start_node: str, exit_strand: Strand, hops_needed: int) -> list[tuple[str, int, IntervalBatch]]:
        """
        Uses a breadth-first search to traverse the assembly graph.

        Finds neighboring contigs and projects their annotated features into
        the coordinate space of the starting contig.

        Args:
            start_node: Starting contig ID.
            exit_strand: Strand to exit from.
            hops_needed: Maximum number of features (intervals) to find.

        Returns:
            list: (contig_id, hop_index, projected_IntervalBatch)
        """
        queue = [(start_node, exit_strand, hops_needed, 1, 0)]
        projected_results = []
        visited_edges = {(start_node, exit_strand)}

        while queue:
            curr_ctg, curr_exit, rem_hops, node_depth, shift = queue.pop(0)
            if rem_hops <= 0: continue

            for edge in self._graph.get_neighbors(curr_ctg):
                if edge.u_strand != curr_exit: continue

                v = edge.v
                if (v, edge.v_strand) in visited_edges: continue
                visited_edges.add((v, edge.v_strand))

                if v in self.features:
                    n_ints = self.features[v]
                    raw_indices = np.arange(0, min(len(n_ints), rem_hops)) if edge.v_strand == Strand.FORWARD \
                        else np.arange(max(0, len(n_ints) - rem_hops), len(n_ints))[::-1]

                    valid_indices = [i for i in raw_indices if
                                     (v, n_ints.original_indices[i]) not in self._visited_nodes]
                    if valid_indices:
                        for i in valid_indices:
                            self._visited_nodes.add((v, n_ints.original_indices[i]))

                        batch = n_ints.filter(valid_indices)
                        new_shift = shift + self.contig_lengths[curr_ctg]
                        flip_len = self.contig_lengths[v] if edge.v_strand == Strand.REVERSE else None

                        projected_batch = batch.project(shift=new_shift, flip_length=flip_len)
                        projected_results.append((v, node_depth, projected_batch))

                        found_count = len(valid_indices)
                        rem_hops -= found_count

                if rem_hops > 0:
                    queue.append((v, edge.v_strand, rem_hops, node_depth + 1, shift + self.contig_lengths[curr_ctg]))

        return projected_results
