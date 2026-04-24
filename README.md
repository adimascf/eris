# eris

[![PyPI version](https://img.shields.io/pypi/v/eris.svg)](https://pypi.org/project/eris/)
[![Python versions](https://img.shields.io/pypi/pyversions/eris.svg)](https://pypi.org/project/eris/)
[![Documentation](https://img.shields.io/badge/docs-GitHub_Pages-blue.svg)](https://tomdstanton.github.io/eris/)
[![License](https://img.shields.io/github/license/tomdstanton/eris.svg)](LICENSE)

**Graph-aware contextual annotation of targeted genomic features.**

eris is a bioinformatics pipeline designed to resolve and annotate shattered genomic
features (like mobile genetic elements or structural variants) directly from assembly graphs.
By employing a dynamic "Anchor-and-Traverse" depth-first search through de Bruijn/string graphs (GFA),
eris stitches together split alignments across multiple micro-contigs, seamlessly bridging sequence
bubbles and gaps.

Crucially, it utilizes fractional read depth flow to distinguish between dominant wild-type structures and rare
sub-clonal insertions, returning detailed evolutionary context including upstream flanks, downstream flanks, and
trapped passenger genes.

---

## 🚀 Features

* **Graph-Aware Traversal:** Overcomes aligner limitations by traversing assembly graph topology to bridge unaligned micro-contigs and resolve shattered targets.
* **Sub-Clonal Resolution:** Integrates graph read-depth (`dp`/`rd` tags) to calculate fractional copy number, easily distinguishing low-frequency variant bubbles from dominant paths.
* **Contextual Annotation:** Sweeps across stitched paths to identify internal passenger genes (e.g., AMR genes trapped inside transposons) and flanking genomic context.
* **High Performance:** Built for speed with a hybrid architecture utilizing `numpy`, `numba`, and `mappy` (Minimap2).
* **Standardized Outputs:** Generates detailed tabular reports (TSV), locus-specific FASTAs, and standard annotations (`pyfgs` powered GFF3/FAA).

## 📦 Installation

eris requires **Python 3.11 or later**.

Install directly from PyPI:
```bash
pip install eris
```

### Dependencies
eris relies on the following core libraries:
* `numpy` (>=2.4)
* `numba` (>=0.65)
* `mappy` (>=2.3)
* `pyfgs` (>=0.0.1)
* `biopython` (>=1.87)

## 🛠️ Usage

eris installs a command-line interface `eris` for immediate use.

```bash
eris -i assembly.gfa -d targets.fasta -o results/sample_A
```

### Basic Arguments
* `-i`, `--genome`: Path to a genome in fasta or GFA format (can be compressed). GFA is heavily recommended to utilize topological stitching.
* `-d`, `--targets`: Path to target nucleotide features (fasta or pre-indexed `.mmi`).
* `-o`, `--outprefix`: Prefix for all generated output files.
* `-f`, `--feature-type`: The type of feature to annotate (default: `CDS`).
* `--hops`: Maximum number of contextual genes to sweep upstream/downstream (default: `3`).
* `--tolerance`: Distance tolerance in base pairs for merging clustered alignments (default: `0`).

### Outputs
Depending on the arguments provided, eris will output:
1. **`{prefix}_report.tsv`**: A detailed report of every resolved locus, including targets, context (INSIDE, UPSTREAM, DOWNSTREAM), biological effects, topological hops, and fractional depths.
2. **`{prefix}_loci.fasta`**: The stitched nucleotide sequences for each assembled structural variant.
3. **`{prefix}_assembly.gff`**: GFF3 annotation of the global features.
4. **`{prefix}_proteins.faa`**: Amino acid fasta of the global features.

## 📚 Documentation

For detailed guides, API reference, and advanced configuration, visit the [eris Documentation](https://tomdstanton.github.io/eris/).

## 💻 Development

eris uses [`hatch`](https://hatch.pypa.io/) for build and environment management.

To set up a local development environment ([uv](https://docs.astral.sh/uv/getting-started/installation/) needs to be installed):
```bash
# Clone the repository
git clone [https://github.com/tomdstanton/eris.git](https://github.com/tomdstanton/eris.git)
cd eris
make dev
```

### Running Tests & Linting
eris enforces rigorous type checking and linting.
```bash
# Run tests
pytest

# Run static type checking
mypy

# Run linting and formatting
ruff check
ruff format
```

## 📝 License

This project is licensed under the terms of the LICENSE file included in the repository.

## 🤝 Authors

* **Tom Stanton** - [tomdstanton@gmail.com](mailto:tomdstanton@gmail.com)