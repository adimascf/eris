import pytest
from types import SimpleNamespace
from pathlib import Path

from eris.cli import describe_input
from eris.io import GenomeAssembly, TargetDatabase
from eris.pipeline import Pipeline
from eris.constants import FeatureType
from eris.copy_number import CopyNumber
from isoforms import read_isoform_groups
from relation_scoring import GenePlacer, read_truth, report, score_relations

DATA_DIR = Path("data")

# The benchmark data is large and kept out of git; skip these tests when it is absent
REQUIRED_FILES = [
    "IS.mmi",
    "EC958_complete_genome.fasta", "complete_ec958_IS.reanotated.sorted.bed",
    "contigs_shovill_simulated.gfa", "ec958_IS.reannotated.sorted.bed",
]
needs_data = pytest.mark.skipif(
    not all((DATA_DIR / f).exists() for f in REQUIRED_FILES),
    reason=f"integration data not found in {DATA_DIR}/",
)

def compute_metrics(truth_file: Path, eris_loci: list, groups: dict[str, str] | None = None, count: str = 'max',
                    column: str = 'copy_number'):
    """
    Evaluates the IS element copy counts against the ground truth. See `count_copies` for the arguments.
    """
    gt_counts, eris_counts = count_copies(truth_file, eris_loci, groups, count, column)

    # Calculate TP, FP, FN
    all_is = set(gt_counts.keys()).union(set(eris_counts.keys()))
    tp = fp = fn = 0

    for is_name in sorted(all_is):
        gt_c = gt_counts.get(is_name, 0)
        er_c = eris_counts.get(is_name, 0)

        # Copy-number-aware metric logic
        c_tp = min(gt_c, er_c)
        c_fp = max(0, er_c - gt_c)
        c_fn = max(0, gt_c - er_c)
        tp += c_tp
        fp += c_fp
        fn += c_fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def count_copies(truth_file: Path, eris_loci: list, groups: dict[str, str] | None = None,
                 count: str = 'max', column: str = 'copy_number') -> tuple[dict[str, int], dict[str, int]]:
    """
    Returns the (truth, eris) IS copy counts, keyed by IS name.

    If `groups` maps IS names to groups, counts are compared per group, and a name it does not list is its own group.
    With isoform groups (`read_isoform_groups`), a naming swap such as IS600 called where the truth has ISSd1 is not
    scored as an error; with IS families, unrelated members of one family also offset each other's errors.

    `count` sets how an IS's loci become a copy count: 'max' takes max(#loci, max copy number), which
    forgives a locus reported twice; 'sum' adds up every locus's copy number, as a user summing the report would;
    'hub' is 'sum' with each repeat hub counted once per group, a guard against near-identical names each
    reporting the same hub with the hub's copy number (#14). 'hub' needs `groups`.

    `column` picks the report column a locus's copy number is read from: 'copy_number' (anchor depth) or
    'copy_number_edges' (anchor link count).
    """
    # Parse Ground Truth
    gt_counts = {}
    with open(truth_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                # Extract base name: 'ISEc23_1' -> 'ISEc23'
                is_name = parts[3].split('_')[0]
                gt_counts[is_name] = gt_counts.get(is_name, 0) + 1

    # Parse Eris Output
    eris_loci_dict = {}
    eris_depth = {}
    eris_sum = {}
    counted_hubs = set()

    for locus in eris_loci:
        if not locus.targets: continue
        
        # Extract base names from every target feature ID, e.g. 'IS30_1_stitched' -> 'IS30'.
        # A locus can hold several adjacent IS (IS26 then ISEc9 on pEC958); each one is a copy.
        is_names = {target.id.split(',')[0].split('_')[0] for target in locus.targets}

        for is_name in is_names:
            if is_name not in eris_loci_dict:
                eris_loci_dict[is_name] = set()
            eris_loci_dict[is_name].add(locus.id)

            copies = int(locus.fractional_depth) if column == 'copy_number' else locus.copy_number.edges
            # Track the max copy number estimate seen for this IS
            eris_depth[is_name] = max(eris_depth.get(is_name, 0), copies)
            if count == 'hub' and (hub := locus.copy_number.hub):
                if (groups.get(is_name, is_name), hub) in counted_hubs:
                    continue
                counted_hubs.add((groups.get(is_name, is_name), hub))
            eris_sum[is_name] = eris_sum.get(is_name, 0) + copies

    # Resolve Final Copy Counts for Eris
    eris_counts = {}
    for is_name in eris_loci_dict:
        # Eris copy count = max of the depth estimate OR the distinct physical loci anchored
        eris_counts[is_name] = max(len(eris_loci_dict[is_name]), eris_depth[is_name]) if count == 'max' else eris_sum.get(is_name, 0)

    # Sum the per-name counts within each group, after the per-name copy estimate above
    if groups:
        gt_counts, eris_counts = _sum_by_group(gt_counts, groups), _sum_by_group(eris_counts, groups)

    return gt_counts, eris_counts


def print_isoform_scores(truth_file: Path, eris_loci: list, groups: dict[str, str]) -> None:
    """
    Prints isoform-group P/R/F1 under each count for both copy-number columns. The benchmarks assert on copy_number
    (anchor depth); copy_number_edges (anchor link count) is printed beside it to compare the two (#47).
    """
    for column in ('copy_number', 'copy_number_edges'):
        if column == 'copy_number' and any(locus.fractional_depth is None for locus in eris_loci):
            continue  # The assembly has no read depth, so copy_number is NA
        for count in ('max', 'sum', 'hub'):
            print(f"isoform {count} {column:<17} P=%.3f R=%.3f F1=%.3f"
                  % compute_metrics(truth_file, eris_loci, groups, count, column))


def find_overlapping_claims(eris_loci: list, groups: dict[str, str]) -> list[tuple[str, str, str]]:
    """
    Returns (locus_a, locus_b, contig) for every two loci of the same group (see `count_copies`) whose targets claim overlapping
    sequence on a contig. Each region of the genome holds at most one copy of an IS, so one should never be reported twice.
    """
    claims = {}  # (group, contig) -> [(start, end, locus_id)]
    for locus in eris_loci:
        if not locus.targets: continue
        for target in locus.targets:
            name = target.id.split('_')[0]
            for seg in target.segments:
                claims.setdefault((groups.get(name, name), seg.contig), []).append((seg.start, seg.end, locus.id))

    overlaps = set()
    for (_, contig), spans in claims.items():
        spans.sort()
        for i, (_, end_a, id_a) in enumerate(spans):
            for start_b, _, id_b in spans[i + 1:]:
                if start_b >= end_a: break
                if id_a != id_b:
                    overlaps.add((*sorted((id_a, id_b)), contig))
    return sorted(overlaps)


def _sum_by_group(counts: dict[str, int], groups: dict[str, str]) -> dict[str, int]:
    summed = {}
    for name, n in counts.items():
        group = groups.get(name, name)
        summed[group] = summed.get(group, 0) + n
    return summed


@pytest.fixture(scope="module")
def target_db():
    # Only initialize the Minimap2 database once for the entire module to save time
    return TargetDatabase(DATA_DIR / "IS.mmi", FeatureType.CDS, indexing_threads=1)


@pytest.fixture(scope="module")
def families(target_db):
    # Target names are '<name>_<family>_<group>', e.g. 'IS600_IS3_IS3'
    return {n.rsplit('_', 2)[0]: n.rsplit('_', 2)[1] for n in target_db.aligner.seq_names if n.count('_') >= 2}


@pytest.fixture(scope="module")
def isoform_groups():
    # The benchmarks score per isoform group, built by scripts/build_isoform_groups.py (docs/adr/0004)
    return read_isoform_groups()


# User mentioned EC958.fasta but that file only contains the chromosome.
# The actual complete genome (chromosome + plasmids) is EC958_complete_genome.fasta.
# The third field is the report column scored. FASTA input has no read depth, so the complete genome's copy_number is
# NA; it is scored on copy_number_edges, which is 1 for every locus in unique sequence, as every locus there is.
BENCHMARKS = {
    "complete": ("EC958_complete_genome.fasta", "complete_ec958_IS.reanotated.sorted.bed", "copy_number_edges"),
    # Fragmented graphs are incredibly messy, but eris should still achieve >85% recovery
    "fragmented": ("contigs_shovill_simulated.gfa", "ec958_IS.reannotated.sorted.bed", "copy_number"),
}

# Isoform groups whose summed copy count is still wrong, as {group: (truth, eris)}. Every other group must match the
# truth exactly. A group may get closer to its truth but never further away; update this table when one improves.
KNOWN_COUNT_ERRORS = {
    "complete": {"ISEc49": (0, 2), "ISSen1": (1, 0), "ISSen10": (1, 0)},
    "fragmented": {"IS1A": (3, 4), "ISEc49": (0, 2), "ISEc8": (1, 2), "ISSen1": (1, 0), "ISSen10": (1, 0)},
}


@pytest.fixture(scope="module", params=BENCHMARKS)
def benchmark(request, target_db):
    genome_file, truth_file, column = BENCHMARKS[request.param]
    # max_feature_hops=0 completely disables the PyFGS gene calling for extreme speed
    with Pipeline(target_db, max_feature_hops=0, mode='collapse', min_coverage=0.30) as pipeline:
        loci = list(pipeline(GenomeAssembly.from_file(DATA_DIR / genome_file)))
    return request.param, DATA_DIR / truth_file, loci, column


@needs_data
def test_benchmark_precision_recall(benchmark, families, isoform_groups):
    _, truth_file, loci, column = benchmark

    # Isoforms (e.g., IS1A vs IS1R) cross-map heavily in nucleotide space, so exact name matches are impossible;
    # score per isoform group, and report exact names and IS families for reference.
    print("exact-name P=%.3f R=%.3f F1=%.3f" % compute_metrics(truth_file, loci, column=column))
    print("family hub P=%.3f R=%.3f F1=%.3f" % compute_metrics(truth_file, loci, families, count='hub', column=column))
    print_isoform_scores(truth_file, loci, isoform_groups)
    precision, recall, f1 = compute_metrics(truth_file, loci, isoform_groups, count='hub', column=column)

    # Score summed copy numbers, each hub once: the max-based count cannot see an IS reported twice
    assert f1 >= 0.94, f"Isoform F1 on summed copy numbers must be >= 0.94, got {f1:.3f}"


@needs_data
def test_benchmark_isoform_copy_counts_do_not_regress(benchmark, isoform_groups):
    # The F1 threshold hides a single double-counted IS (issue #1 scored 0.857); compare each isoform group instead
    name, truth_file, loci, column = benchmark
    gt_counts, eris_counts = count_copies(truth_file, loci, isoform_groups, count='sum', column=column)
    known = KNOWN_COUNT_ERRORS[name]

    worse = {}
    for group in sorted(set(gt_counts) | set(eris_counts)):
        truth, eris = gt_counts.get(group, 0), eris_counts.get(group, 0)
        allowed = abs(known[group][1] - known[group][0]) if group in known else 0
        if abs(eris - truth) > allowed:
            worse[group] = (truth, eris)
    assert worse == {}, f"(truth, eris) copy counts got worse; known errors are {known}"


@needs_data
def test_benchmark_reports_each_copy_once(benchmark, isoform_groups):
    _, _, loci, _ = benchmark
    assert find_overlapping_claims(loci, isoform_groups) == []


# The simulated assembly's contigs as FASTA, without the graph: no F1 threshold, as recall drops without stitching
FRAGMENTED_FASTA = DATA_DIR / "contigs_shovill_simulated.fasta"


@needs_data
@pytest.mark.skipif(not FRAGMENTED_FASTA.exists(), reason=f"{FRAGMENTED_FASTA} not found")
def test_fragmented_fasta_warns_and_reports_no_stitched_loci_and_no_double_claims(target_db, isoform_groups):
    genome = GenomeAssembly.from_file(FRAGMENTED_FASTA)
    with Pipeline(target_db, max_feature_hops=0, mode='collapse', min_coverage=0.30) as pipeline:
        loci = list(pipeline(genome))

    _, warnings = describe_input(genome)
    assert any("stitching and cross-contig flank search are off" in w for w in warnings)
    assert any("reported as NA" in w for w in warnings)
    assert loci and [locus.id for locus in loci if any(len(t.segments) > 1 for t in locus.targets)] == []
    assert find_overlapping_claims(loci, isoform_groups) == []


# The real-read assembly collapses repeats that the simulated one keeps apart, including copies on the plasmids
REAL_READS = ("contigs_shovill_realfastq.gfa", "complete_ec958_IS.reanotated.sorted.bed")
needs_real_reads = pytest.mark.skipif(not (DATA_DIR / REAL_READS[0]).exists(),
                                      reason=f"real-read assembly not found in {DATA_DIR}/")


@pytest.fixture(scope="module")
def real_reads(target_db):
    with Pipeline(target_db, max_feature_hops=0, mode='collapse', min_coverage=0.30) as pipeline:
        return list(pipeline(GenomeAssembly.from_file(DATA_DIR / REAL_READS[0])))


@needs_data
@needs_real_reads
def test_real_reads_count_each_replicon_once(real_reads, families, isoform_groups):
    truth_file = DATA_DIR / REAL_READS[1]
    print("exact-name P=%.3f R=%.3f F1=%.3f" % compute_metrics(truth_file, real_reads))
    print("family hub P=%.3f R=%.3f F1=%.3f" % compute_metrics(truth_file, real_reads, families, count='hub'))
    print_isoform_scores(truth_file, real_reads, isoform_groups)
    precision, recall, f1 = compute_metrics(truth_file, real_reads, isoform_groups, count='hub')
    # Lower than the other assemblies: an IS629 copy offset a missed ISSen1 at family level (0.953) but not here
    assert f1 >= 0.93, f"Isoform F1 with each hub counted once must be >= 0.93, got {f1:.3f}"


@needs_data
@needs_real_reads
def test_real_reads_report_each_hub_once(real_reads, families, isoform_groups):
    # Near-identical names (IS26, IS15, IS15DI, IS15DIV) each stitched a path through the IS26 hub (#14)
    assert find_overlapping_claims(real_reads, isoform_groups) == []
    # ... and the hub is still reported, once per family, so no other IS6 or IS1 name claims it either:
    # IS6 through the IS26 hub, IS1 through the 200-270x IS1 contigs
    hubs = {"IS6": {"2421250", "2408206", "2421022"}, "IS1": {"2420366", "2060500", "2421094", "2421096", "2420860"}}
    claiming = {family: [locus.id for locus in real_reads
                         if any(families.get(t.id.split('_')[0]) == family and {s.contig for s in t.segments} & hub
                                for t in locus.targets)]
                for family, hub in hubs.items()}
    assert {family: len(ids) for family, ids in claiming.items()} == {"IS6": 1, "IS1": 1}


@needs_data
@needs_real_reads
def test_real_read_hubs_split_their_copies_by_depth_tier(real_reads):
    # Copies per depth tier, chromosome first, from the complete EC958 genome's annotation; none from the fallback
    expected = {"IS1A": (5, (3, 2)), "ISEc23": (4, (3, 1)), "ISEc9": (2, (1, 1)), "ISEc12": (4, (4,))}
    found = {}
    for locus in real_reads:
        for target in locus.targets:
            name = target.id.split('_')[0]
            if name in expected and (copies := locus.copy_number).hub and not copies.fallback:
                found.setdefault(name, set()).add((copies.estimate, tuple(n for n, _ in copies.tiers)))
    assert found == {name: {copies} for name, copies in expected.items()}


def test_compute_metrics_isoform_level_merges_only_naming_swaps(tmp_path):
    # Truth has one IS600 and one ISSd1, isoforms of each other; eris calls the shared node IS600 with 2 copies
    truth = tmp_path / "truth.bed"
    truth.write_text("chr\t0\t1266\tISSd1\t0\t+\nchr\t5000\t6265\tIS600\t0\t-\n")
    stitched = SimpleNamespace(id="locus_1", fractional_depth=2.0,
                               targets=[SimpleNamespace(id="IS600_IS3_IS3_stitched")])
    groups = read_isoform_groups()

    assert compute_metrics(truth, [stitched]) == (0.5, 0.5, 0.5)
    assert compute_metrics(truth, [stitched], groups=groups) == (1.0, 1.0, 1.0)

    # IS629 and ISSen1 share the IS3 family but not an isoform group: an extra IS629 must not offset a missed ISSen1
    truth.write_text("chr\t0\t1310\tISSen1\t0\t+\nchr\t5000\t6309\tIS629\t0\t-\n")
    extra = SimpleNamespace(id="locus_1", fractional_depth=2.0, targets=[SimpleNamespace(id="IS629_IS3_IS3")])
    assert compute_metrics(truth, [extra], groups=groups) == (0.5, 0.5, 0.5)


def test_count_copies_scores_the_edge_count_when_asked(tmp_path):
    # Anchor depth gives the hub 3 copies; the links reaching anchors give 2
    truth = tmp_path / "truth.bed"
    truth.write_text("".join(f"chr\t{i}000\t{i}768\tIS1A\t0\t+\n" for i in range(3)))
    hub = SimpleNamespace(id="locus_1", fractional_depth=3.0, copy_number=CopyNumber(3, 2, hub=frozenset({"h"})),
                          targets=[SimpleNamespace(id="IS1A_IS1_IS1")])

    assert count_copies(truth, [hub], count='sum') == ({"IS1A": 3}, {"IS1A": 3})
    assert count_copies(truth, [hub], count='sum', column='copy_number_edges') == ({"IS1A": 3}, {"IS1A": 2})
    assert compute_metrics(truth, [hub], column='copy_number_edges') == (1.0, 2 / 3, 0.8)


def test_find_overlapping_claims_flags_leftover_half_of_stitched_locus():
    # Issue #1: the first half of ISEc23 on ctgA was reported both inside the stitch and as a local locus
    def locus(id_, *segments):
        return SimpleNamespace(id=id_, targets=[SimpleNamespace(
            id="ISEc23_IS66_unknown", segments=[SimpleNamespace(contig=c, start=a, end=b) for c, a, b in segments])])

    stitched = locus("locus_stitched", ("ctgA", 0, 1326), ("ctgB", 0, 1135))
    leftover = locus("locus_local", ("ctgA", 0, 1326))
    elsewhere = locus("locus_other", ("ctgA", 2000, 3000))
    families = {"ISEc23": "IS66"}

    assert find_overlapping_claims([stitched, leftover, elsewhere], families) == [("locus_local", "locus_stitched", "ctgA")]
    assert find_overlapping_claims([stitched, elsewhere], families) == []


# Relation benchmark -------------------------------------------------------------------------------------------------
# Scores how eris relates each IS to its flanks against the curated EC958 truth, built by
# scripts/build_relation_truth.py. See docs/adr/0002-relation-benchmark-truth.md and tests/relation_scoring.py.
RELATION_TRUTH = DATA_DIR / "ec958_IS_cds_relations.tsv"
REFERENCE = DATA_DIR / "EC958_complete_genome.fasta"
needs_relation_data = pytest.mark.skipif(
    not all(f.exists() for f in (DATA_DIR / "IS.mmi", RELATION_TRUTH, REFERENCE, DATA_DIR / "contigs_shovill_simulated.gfa")),
    reason=f"relation benchmark data not found in {DATA_DIR}/",
)
RELATION_BENCHMARKS = {
    # assembly, the replicons its truth relations lie on
    "complete": ("EC958_complete_genome.fasta", None),
    "fragmented": ("contigs_shovill_simulated.gfa", {"HG941718.1"}),  # simulated from the chromosome only
}
RELATION_HOPS = 3  # eris's default
REACH_HOPS = 20  # a miss found with this budget is put down to the budget, not to a wrong gene

# Truth relations eris misses, as {(isoform group, side, locus tag): cause}. The cause is "budget" when a run with
# REACH_HOPS finds the relation, else "gene". A new miss fails, and so does a fixed one or one whose cause changes,
# until this table is updated, so the benchmark can only get better.
KNOWN_RELATION_MISSES = {
    "complete": {
        ("IS1A", "upstream", "EC958_RS30185"): "gene",
        ("IS1X1", "downstream", "EC958_RS30560"): "gene",
        ("IS1X1", "upstream", "EC958_RS30140"): "gene",
        ("IS2", "downstream", "EC958_RS30340"): "gene",
        ("IS26", "downstream", "EC958_RS25830"): "budget",
        ("IS26", "downstream", "EC958_RS25840"): "budget",
        ("IS26", "downstream", "EC958_RS26485"): "gene",
        ("IS26", "downstream", "EC958_RS29660"): "gene",
        ("IS26", "upstream", "EC958_RS29475"): "gene",
        ("IS30", "downstream", "EC958_RS16655"): "gene",
        ("IS3411", "upstream", "EC958_RS29355"): "gene",
        ("IS5", "downstream", "EC958_RS24665"): "gene",
        ("IS5", "upstream", "EC958_RS24675"): "gene",
        ("IS600", "downstream", "EC958_RS16660"): "gene",
        ("IS600", "downstream", "EC958_RS28175"): "gene",
        ("IS600", "upstream", "EC958_RS16650"): "gene",
        ("IS600", "upstream", "EC958_RS24690"): "gene",
        ("IS609", "downstream", "EC958_RS10865"): "gene",
        ("IS609", "upstream", "EC958_RS10860"): "gene",
        ("IS6100", "upstream", "EC958_RS29565"): "gene",
        ("ISEc22", "downstream", "EC958_RS29550"): "gene",
        ("ISEc23", "downstream", "EC958_RS01090"): "gene",
        ("ISEc23", "downstream", "EC958_RS11750"): "gene",
        ("ISEc52", "downstream", "EC958_RS28255"): "gene",
        ("ISEc52", "downstream", "EC958_RS29425"): "gene",
        ("ISEc52", "upstream", "EC958_RS24680"): "gene",
        ("ISEc53", "downstream", "EC958_RS25065"): "gene",
        ("ISEc9", "downstream", "EC958_RS24670"): "gene",
        ("ISSen1", "downstream", "EC958_RS24890"): "gene",
        ("ISSen1", "upstream", "EC958_RS29720"): "gene",
        ("ISSen10", "downstream", "EC958_RS20820"): "gene",
        ("ISSen10", "upstream", "EC958_RS20830"): "gene",
    },
    "fragmented": {
        ("IS100", "upstream", "EC958_RS30525"): "gene",
        ("IS1397", "downstream", "EC958_RS11060"): "gene",
        ("IS1X1", "downstream", "EC958_RS29430"): "gene",
        ("IS1X1", "downstream", "EC958_RS30560"): "gene",
        ("IS1X1", "upstream", "EC958_RS30140"): "gene",
        ("IS2", "downstream", "EC958_RS30340"): "gene",
        ("IS200", "downstream", "EC958_RS11105"): "gene",
        ("IS200", "upstream", "EC958_RS11880"): "gene",
        ("IS3411", "downstream", "EC958_RS20830"): "gene",
        ("IS5", "downstream", "EC958_RS24665"): "gene",
        ("IS5", "upstream", "EC958_RS24675"): "gene",
        ("IS600", "downstream", "EC958_RS16660"): "gene",
        ("IS600", "upstream", "EC958_RS16650"): "gene",
        ("IS600", "upstream", "EC958_RS24690"): "gene",
        ("IS609", "downstream", "EC958_RS10865"): "gene",
        ("IS609", "upstream", "EC958_RS10860"): "gene",
        ("IS609", "upstream", "EC958_RS19340"): "gene",
        ("IS682", "upstream", "EC958_RS30005"): "gene",
        ("ISEc12", "downstream", "EC958_RS07600"): "gene",
        ("ISEc12", "downstream", "EC958_RS11780"): "gene",
        ("ISEc12", "downstream", "EC958_RS24865"): "gene",
        ("ISEc12", "upstream", "EC958_RS07585"): "budget",
        ("ISEc12", "upstream", "EC958_RS15715"): "budget",
        ("ISEc22", "downstream", "EC958_RS29550"): "gene",
        ("ISEc22", "upstream", "EC958_RS16805"): "gene",
        ("ISEc23", "downstream", "EC958_RS01090"): "gene",
        ("ISEc23", "downstream", "EC958_RS11750"): "gene",
        ("ISEc23", "upstream", "EC958_RS30575"): "gene",
        ("ISEc52", "downstream", "EC958_RS28255"): "gene",
        ("ISEc52", "downstream", "EC958_RS29425"): "gene",
        ("ISEc52", "upstream", "EC958_RS24680"): "gene",
        ("ISEc53", "downstream", "EC958_RS25065"): "gene",
        ("ISEc9", "downstream", "EC958_RS24670"): "gene",
        ("ISSen1", "downstream", "EC958_RS24890"): "gene",
        ("ISSen1", "upstream", "EC958_RS29720"): "gene",
        ("ISSen10", "downstream", "EC958_RS20820"): "gene",
        ("ISSen10", "upstream", "EC958_RS20830"): "gene",
    },
}
# Found relations whose eris gene does not have the RefSeq CDS's sequence; checked on the complete genome only
# (ISEc52's EC958_RS24680 was one while an unrelated IS3 locus could find it; per isoform group it is a miss)
KNOWN_NUCLEOTIDE_MISMATCHES = set()
# Floors for (precision, recall) of DISRUPTED or TRUNCATED against truth disrupted; raise them as eris improves
MIN_DISRUPTED = {"complete": (0.17, 0.42), "fragmented": (0.05, 0.05)}


class GeneSequences:
    """Collects the coding sequence of every eris gene call, as the _cds.fna output would hold them."""

    def __init__(self):
        self.by_id = {}

    def write_global_genes(self, contig_id, sequence, genes):
        for g in genes:
            # The gene id eris gives each call in Pipeline._process_contig
            self.by_id[f"{contig_id}_{g.start}_{g.end}"] = g.sequence()


def assert_unchanged(found, known, table: str):
    """A ratchet: what was found must equal the known table, so fixes get noticed as well as regressions."""
    assert found == known, f"eris changed; if it got better, update {table}"


@pytest.fixture(scope="module")
def placer():
    return GenePlacer(REFERENCE)


@pytest.fixture(scope="module", params=RELATION_BENCHMARKS)
def relation_benchmark(request, target_db, families, isoform_groups, placer):
    genome_file, replicons = RELATION_BENCHMARKS[request.param]
    runs, genes = {}, GeneSequences()  # gene calls do not depend on the hops budget
    for hops in (RELATION_HOPS, REACH_HOPS):
        with Pipeline(target_db, max_feature_hops=hops, mode='collapse', min_coverage=0.30) as pipeline:
            loci = list(pipeline(GenomeAssembly.from_file(DATA_DIR / genome_file), genes))
        runs[hops] = loci

    placements = {}

    def place(gene_id):
        if gene_id not in placements:
            placements[gene_id] = placer.place(genes.by_id[gene_id])
        return placements[gene_id]

    truths = read_truth(RELATION_TRUTH, isoform_groups, replicons)
    score = score_relations(truths, runs[RELATION_HOPS], isoform_groups, place, reach_loci=runs[REACH_HOPS])
    by_family = score_relations(read_truth(RELATION_TRUTH, families, replicons), runs[RELATION_HOPS], families, place)
    print(f"\n{request.param}:\n{report(score, by_family)}")
    return request.param, score


@pytest.mark.slow
@needs_relation_data
def test_relation_misses_do_not_change(relation_benchmark):
    name, score = relation_benchmark
    assert_unchanged(score.miss_causes, KNOWN_RELATION_MISSES[name], "KNOWN_RELATION_MISSES")


@pytest.mark.slow
@needs_relation_data
def test_disrupted_calls_meet_thresholds(relation_benchmark):
    name, score = relation_benchmark
    min_precision, min_recall = MIN_DISRUPTED[name]
    assert score.disrupted_precision >= min_precision
    assert score.disrupted_recall >= min_recall


@pytest.mark.slow
@needs_relation_data
def test_matched_flank_sequences_do_not_change(relation_benchmark):
    name, score = relation_benchmark
    if name != "complete":
        pytest.skip("the nucleotide check is ratcheted on the complete genome only")
    assert_unchanged(score.nucleotide_mismatches, KNOWN_NUCLEOTIDE_MISMATCHES, "KNOWN_NUCLEOTIDE_MISMATCHES")
