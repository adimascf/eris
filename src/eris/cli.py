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


# Functions ------------------------------------------------------------------------------------------------------------
def fraction(value: str) -> float:
    """Argparse type for a fraction from 0.0 to 1.0."""
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not 0.0 <= f <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not between 0.0 and 1.0")
    return f


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
    outs.add_argument('--no-fna', action='store_true',
                      help='Do not write FNA output for global transcripts (CDS)')

    pipeline_args = parser.add_argument_group('⚙️', 'Pipeline arguments')
    pipeline_args.add_argument('--hops', type=int, default=3, metavar='',
                        help='Maximum number of contextual genes to sweep upstream/downstream; in collapse mode, '
                             'along each graph branch, so every copy of a collapsed repeat gets flanks')
    pipeline_args.add_argument('-l', '--tolerance', type=int, default=0, metavar='',
                        help='Distance tolerance (bp) for merging clustered target alignments')
    pipeline_args.add_argument('-t', '--max-workers', type=int, default=None, metavar='',
                        help='Maximum number of worker threads for alignment and CDS prediction')
    pipeline_args.add_argument('-m', '--mode', choices=['variant', 'collapse'], default='variant',
                        help='Resolution mode: "variant" for all possible structural path, "collapse" for a consensus')
    pipeline_args.add_argument('-c', '--min-coverage', type=fraction, default=0.30, metavar='',
                        help='Minimum target coverage (0.0 to 1.0) required to report a locus target')
    pipeline_args.add_argument('--min-identity', type=fraction, default=0.0, metavar='',
                        help='Minimum target identity (0.0 to 1.0; matching bases over aligned bases, gaps counted) '
                             'required to report a locus target. Stitched loci are pooled over their fragments')
    pipeline_args.add_argument('--merge-synonyms', nargs='?', const=True, default=None, metavar='IS.csv',
                        help='Rename IS targets that ISfinder lists as synonyms/isoforms (e.g. IS1A/IS1B, IS30/IS30D) '
                             'to the first name found in the genome, so they can be stitched and counted together. '
                             'Downloads and caches ISfinder\'s IS.csv, or give a local IS.csv path')

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

    synonyms = None
    if args.merge_synonyms:
        from eris.synonyms import SynonymResolver, fetch_isfinder_table
        if args.merge_synonyms is True:
            log.msg("📥 Fetching ISfinder synonym table...\n", flush=True)
            table = fetch_isfinder_table()
        else:
            table = Path(args.merge_synonyms)
        synonyms = SynonymResolver(table)
        log.msg(f"✅ Synonym table loaded: {table}\n")

    log.msg("🧬 Loading genome...\n", flush=True)
    genome = GenomeAssembly.from_file(args.genome)
    log.msg("✅ Genome loaded!\n")

    # Run pipeline -----------------------------------------------------------------------------------------------------
    from eris.pipeline import Pipeline

    # Hook up the OutputManager parameters explicitly
    with (Pipeline(target_db, args.hops, args.tolerance, args.max_workers, mode=args.mode,
                   min_coverage=args.min_coverage, min_identity=args.min_identity, synonyms=synonyms) as pipeline,
          OutputManager(args.outprefix, write_gff=not args.no_gff, write_faa=not args.no_faa, write_fna=not args.no_fna, mode=args.mode) as out):

        log.msg("⌛️ Running topological traversal...\n", flush=True)

        for locus in pipeline(genome, out=out):
            # Use the clean, encapsulated IO methods we built!
            out.write_locus_relations(locus)

            if out.locus_fasta:
                out.write_locus_fasta(locus, locus.extract_sequence(genome))

    log.msg("🎉 Pipeline completed!\n")

if __name__ == "__main__":
    main()