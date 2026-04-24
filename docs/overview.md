# Pipeline Architecture & Algorithmic Overview

eris is designed to find, resolve, and contextually annotate targeted genomic features (like mobile genetic elements or structural variants) from fragmented genome assemblies.

Standard aligners fail when an insertion shatters across multiple micro-contigs. eris overcomes this by treating the genome not as a flat text file, but as a topological graph, using fractional read depths to navigate through assembly bubbles and stitch shattered targets back together.

### The Separation of Concerns
The architecture is strictly divided into two distinct worlds:
* **The Vectorized Pipeline:** Extremely fast, array-based math utilizing `NumPy` to handle bulk alignments and gene predictions.
* **The Object-Oriented Assembler:** Slower, highly precise topological logic that handles the biological realities of graph navigation and feature extraction.

---

## Step 1: Ingestion & Vectorized Alignment
The pipeline begins by loading the assembly graph (GFA) and the target database (e.g., an IS element library).

1.  **Alignment (`mappy`):** The pipeline maps the target sequences against the genome.
2.  **The `AlignmentBatch` (Structure of Arrays):** Instead of creating thousands of slow Python objects for every alignment, the results are loaded into an `AlignmentBatch`. This uses a Structure of Arrays (SoA) architecture. Every property (starts, ends, strands, qualities) is stored as a contiguous C-level `NumPy` array. This allows the engine to instantly filter thousands of alignments using boolean masks with virtually zero memory overhead.

## Step 2: Feature Prediction
To provide genomic context (passenger genes and flanks), eris predicts the locations of all Coding Sequences (CDS) on the genome.

1.  **Gene Finding (`pyfgs`):** The pipeline runs rapid gene prediction on every contig.
2.  **The `IntervalBatch`:** Similar to the alignments, these gene coordinates are instantly packed into an `IntervalBatch`. This allows eris to perform vectorized spatial queries (e.g., "Give me all genes between coordinate 500 and 1500") in microseconds without iterating over lists.

---

## Step 3: Graph Traversal & Stitching (`TopologyEngine`)
This is the mathematical core of eris. The `TopologyEngine` acts as a cartographer. It knows nothing about biological DNA—it only understands graph nodes, edges, lengths, and read depths.

Its primary job is to find **partial alignments** (fragments of a target that hit the edge of a contig) and ask the graph if they connect to other fragments.

### Anchor-and-Traverse Algorithm (DAG DFS)
When the engine finds a shattered query (e.g., a transposon split into three pieces), it attempts to chain them together using a Depth-First Search (DFS) on a Directed Acyclic Graph (DAG).

1.  **Anchor Pairing:** It looks at Fragment A and Fragment B, calculating the expected gap between them on the query sequence:

    $$E_{gap}=Start_{next}-End_{curr}$$

2.  **Graph Walk:** The DFS stack walks through the GFA edges from Fragment A's contig. At every step, it calculates
    the physical length of the accumulated path. Because assembly graphs represent overlapping k-mers, it subtracts
    the physical edge overlaps so the path length isn't artificially inflated:

    $$L_{path}=\sum L_{contig}-\sum L_{overlap}$$

3.  **Strand Validation:** Upon reaching Fragment B's contig, it verifies that the graph traversal entered on the
    correct biological strand to match the aligner's orientation.


### Topological Scoring Math
If a path connects two fragments, eris evaluates its biological validity to filter out artificial assembler errors (like 1bp sequencing errors that cause sequence bubbles).

1.  **Length Penalty ($P_{len}$):** How well does the physical graph path match the missing query sequence?
    (A 50bp buffer is added to the denominator to prevent explosive scores from negative gaps caused by fuzzy mapping overlaps).

    $$P_{len}=\max\left(1.0-\frac{|L_{path}-E_{gap}|}{|E_{gap}|+50},0\right)$$

2.  **Depth Fraction ($F_{depth}$):** eris finds the lowest read-depth node along the traversal (the bottleneck)
    and divides it by the read-depth of the starting anchor. This reveals the sub-clonal abundance of the path.

    $$F_{depth}=\frac{D_{bottleneck}}{D_{source}}$$

3.  **Final Score:**

    $$S=P_{len}\times F_{depth}$$

If the score is `< 0.05`, the path represents less than 5% of the biological flow (likely an error bubble) and is
silently destroyed. If multiple paths are valid, the engine utilizes a **Tie-Breaker** to select the path that
consumes the most *real* alignment fragments over *synthetic* unaligned nodes.

---

## Step 4: Contextual Assembly (`LocusBuilder`)
Once the `TopologyEngine` has cleaned the alignments and chained the shattered fragments, it passes them to the `LocusBuilder`.

The `LocusBuilder` is the "Biologist" of the pipeline. It translates the mathematical paths back into biological realities.

1.  **Inside Passengers:** It sweeps across every node in the stitched path, querying the `IntervalBatch` to pull out any genes trapped *inside* the insertion (e.g., AMR genes carried by a transposon).
2.  **Upstream / Downstream Flanks:** It looks at the extreme boundaries of the insertion and extracts a user-defined number of flanking genes to establish the genomic neighborhood. If the insertion sits at the absolute end of a contig, the `LocusBuilder` will spill over into the `TopologyEngine` to walk the graph and find flanking genes on adjacent contigs.
3.  **Mutation Checking:** If an insertion disrupted a gene, the builder accesses the raw DNA bytes to calculate if the feature suffered a frame-shift or physical truncation.

## Step 5: The Output (`OutputManager`)
Finally, the generated `Locus` objects are flushed to disk. The pipeline exports the isolated DNA of the variant (`.fasta`), the global genomic features (`.gff` / `.faa`), and a rigorous tabular report detailing the precise spatial, topological, and strand-relative orientation of every gene involved in the event.
