"""
Module for managing the CLI layer on top of the API; also contains the CLI entry point under `main()`.
"""
import argparse
import dataclasses
from pathlib import Path
from sys import stderr

from eris._version import __version__
from eris.constants import FeatureType


# Classes --------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(slots=True, frozen=True)
class Log:
    quiet: bool = False
    stream = stderr

    def msg(self, msg: str, flush: bool = False) -> None:
        if self.quiet: return
        self.stream.write(msg)
        if flush: self.stream.flush()


# Main CLI entry-point -------------------------------------------------------------------------------------------------
def main():

    # Define args ------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Graph-aware contextual annotation of targeted genomic features',
        usage="%(prog)s -i genome.gfa -d targets.fasta [options]", add_help=False, prog=__package__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inputs = parser.add_argument_group('📁', 'Input arguments')
    inputs.add_argument('-i', '--genome', type=Path, required=True, metavar='',
                        help='Path to a genome in fasta or GFA format; File may be compressed.')
    inputs.add_argument('-d', '--targets', type=Path, required=True, metavar='',
                        help='Path to target nucleotide features in fasta or mmi format')

    outs = parser.add_argument_group('💾', 'Output arguments')
    outs.add_argument('-o', '--outprefix', type=str, default=None, metavar='',
                      help='Prefix for output files (if absent, prints TSV to stdout)')
    outs.add_argument('--no-gff', action='store_true',
                      help='Do not write GFF3 output for global genes')
    outs.add_argument('--no-faa', action='store_true',
                      help='Do not write FAA output for global proteins')

    pipeline_args = parser.add_argument_group('⚙️', 'Pipeline arguments')
    pipeline_args.add_argument('--hops', type=int, default=3, metavar='',
                        help='Maximum number of contextual genes to sweep upstream/downstream')
    pipeline_args.add_argument('-l', '--tolerance', type=int, default=0, metavar='',
                        help='Distance tolerance (bp) for merging clustered target alignments')
    pipeline_args.add_argument('-t', '--max-workers', type=int, default=None, metavar='',
                        help='Maximum number of worker threads for alignment and CDS prediction')

    targets = parser.add_argument_group('🎯', 'Target arguments')
    targets.add_argument('-f', '--feature-type', choices=[e.value for e in FeatureType], metavar='', 
                         default=FeatureType.CDS.value, help='Type of feature to annotate')
    targets.add_argument('--indexing-threads', type=int, default=3, metavar='',
                        help='Number of threads to use for indexing if not already indexed')

    opts = parser.add_argument_group('🛠️', 'Other options')
    opts.add_argument('-q', '--quiet', action='store_true',
                      help='Suppress console logging output')
    opts.add_argument('-v', '--version', action='version', version=__version__,
                      help='Show version number and exit')
    opts.add_argument('-h', '--help', action='help',
                      help='Show this help message and exit')

    # Parse args -------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    log = Log(quiet=args.quiet)

    # Load inputs ------------------------------------------------------------------------------------------------------
    from eris.io import GenomeAssembly, TargetDatabase, OutputManager

    log.msg("🎯 Loading targets...\n", flush=True)
    target_db = TargetDatabase(args.targets, FeatureType(args.feature_type), args.indexing_threads)
    log.msg("✅ Targets loaded!\n")

    log.msg("🧬 Loading genome...\n", flush=True)
    genome = GenomeAssembly.from_file(args.genome)
    log.msg("✅ Genome loaded!\n")

    # Run pipeline -----------------------------------------------------------------------------------------------------
    from eris.pipeline import Pipeline

    # Hook up the OutputManager parameters explicitly
    with (Pipeline(target_db, args.hops, args.tolerance, args.max_workers) as pipeline,
          OutputManager(args.outprefix, write_gff=not args.no_gff, write_faa=not args.no_faa) as out):

        log.msg("⌛️ Running topological traversal...\n", flush=True)

        for locus in pipeline(genome, out=out):
            # Use the clean, encapsulated IO methods we built!
            out.write_locus_relations(locus)

            if out.locus_fasta:
                out.write_locus_fasta(locus.id, locus.extract_sequence(genome))

    log.msg("🎉 Pipeline completed!\n")

if __name__ == "__main__":
    main()