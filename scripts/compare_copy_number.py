"""
Compares collapse-mode copy-number estimators on the EC958 benchmarks (issue #30, a spike: eris output is unchanged).

Copy number counts each replicon once (see CONTEXT.md and docs/adr/0003). The estimators:

1. old: the deepest contig on the locus's path over the genome's median contig depth, as eris reports now.
2. link: the number of links from the repeat hub to its anchors, averaged over the hub's two sides.
3. anchor: hub depth over anchor depth, with anchors grouped into depth tiers.
4. combined: (3) as the estimate and (2) as the check, giving a min/max range when they disagree. A hub with no
   anchors falls back to hub depth over the length-weighted depth of contigs of at least 1 kb, and is flagged.

A hub is the set of contigs a locus target covers end to end; a target with sequence beyond it on its contig sits in
unique sequence and counts 1 under estimators 2-4. Loci of one family that share a hub are scored once, since until
#14 is fixed several near-identical names walk through the same hub. Families are scored with the copy-aware P/R/F1
of tests/test_integration.py.

Prints, for the simulated and the real-read assembly: every locus target with its estimates, every hub where link
count and anchor depth disagree with the walk results that explain it, family-level P/R/F1 per estimator, and (real
reads only) how well depth tiers separate chromosome from plasmid anchors at a range of tolerances, by mapping each
anchor to the complete EC958 genome.

Usage:
    uv run python scripts/compare_copy_number.py [--data DIR]
"""
import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import mappy

from eris.constants import FeatureType
from eris.graph import Edge, Graph
from eris.interval import Strand
from eris.io import GenomeAssembly, TargetDatabase
from eris.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from test_integration import compute_metrics  # noqa: E402

ASSEMBLIES = [
    # (label, assembly, truth BED); the simulated reads come from the chromosome only
    ("simulated", "contigs_shovill_simulated.gfa", "ec958_IS.reannotated.sorted.bed"),
    ("real reads", "contigs_shovill_realfastq.gfa", "complete_ec958_IS.reanotated.sorted.bed"),
]
REFERENCE = "EC958_complete_genome.fasta"
REPLICONS = {"HG941718.1": "chr", "HG941719.1": "pEC958", "HG941720.1": "pEC958B"}
TOLERANCES = (1.2, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.8)
ESTIMATORS = ("old", "link", "anchor", "low", "high")

MIN_ANCHOR_LENGTH = 1000  # bp; a shorter contig next to a hub is a stub that the walk crosses
STUB_BUDGET = 500  # bp of stubs a walk may cross before it gives up (ISEc12's flanks break into 200-400 bp stubs)
TIER_TOLERANCE = 1.4  # largest depth ratio between neighbouring anchors in one depth tier
END_SLOP = 10  # bp a target alignment may stop short of a contig's ends, beyond the link overlap, and still cover it


def depth_tiers(depths: list[float], tolerance: float = TIER_TOLERANCE) -> list[list[float]]:
    """
    Groups depths into depth tiers: sorted, a new tier starts where a depth exceeds the one before it by more than
    `tolerance`. Comparing neighbours, not each tier's extremes, keeps a gradient such as the chromosome's in one tier.
    """
    tiers: list[list[float]] = []
    for depth in sorted(depths):
        if tiers and depth <= tiers[-1][-1] * tolerance:
            tiers[-1].append(depth)
        else:
            tiers.append([depth])
    return tiers


def hub_contigs(segments: list[tuple[str, int, int, Strand]], lengths: dict[str, int],
                overlap: int) -> list[tuple[str, Strand]]:
    """
    The contigs of a locus target's segments that the target covers end to end, each once with the strand it is
    first read on. Together they are the target's repeat hub; a contig with sequence beyond the target is not in it.
    """
    hub: dict[str, Strand] = {}
    for contig, start, end, strand in segments:
        if lengths[contig] - (end - start) <= overlap + END_SLOP:
            hub.setdefault(contig, strand)
    return list(hub.items())


class WalkEnd(NamedTuple):
    """Where one walk out of a repeat hub ends: an anchor, a repeat (another target's hub), a dead end or a tip."""
    contig: str
    strand: Strand
    kind: str


def link_graph(edges: list[Edge]) -> Graph:
    """The assembly graph with each link on both strands, as `TopologyEngine` builds it."""
    return Graph(edges + [e.complement() for e in edges])


def walk_out(graph: Graph, hub: list[tuple[str, Strand]], lengths: dict[str, int], repeats: set[str],
             min_anchor: int = MIN_ANCHOR_LENGTH, stub_budget: int = STUB_BUDGET) -> dict[str, list[WalkEnd]]:
    """
    Walks out of a repeat hub on each side, following edge orientation so a walk never turns back towards the hub's
    side of a contig (an undirected walk out of the IS26 hub reaches the rRNA repeat). `hub` holds each hub contig
    with the strand the target reads it on. A walk ends at the first contig of at least `min_anchor` bp (an anchor),
    at a contig in `repeats`, or when the stubs it crossed exceed `stub_budget`. Each distinct oriented end is one
    link, so an anchor between two inverted copies counts twice.
    """
    in_hub = {contig for contig, _ in hub}
    ends = {}
    for side, sign in (("upstream", -1), ("downstream", 1)):
        found: dict[tuple[str, Strand], WalkEnd] = {}
        stack = [(edge, 0) for contig, strand in hub for edge in graph.get_neighbors(contig)
                 if edge.u_strand == strand * sign]
        seen = set()
        while stack:
            edge, used = stack.pop()
            node = (edge.v, edge.v_strand)
            if node in seen or edge.v in in_hub:
                continue
            seen.add(node)
            used += lengths[edge.v] - edge.overlap
            onward = [e for e in graph.get_neighbors(edge.v) if e.u_strand == edge.v_strand]
            if edge.v in repeats:
                kind = "repeat"
            elif lengths[edge.v] >= min_anchor:
                kind = "anchor"
            elif used > stub_budget:
                kind = "dead_end"
            elif not onward:
                kind = "tip"
            else:
                stack.extend((e, used) for e in onward)
                continue
            found.setdefault(node, WalkEnd(edge.v, edge.v_strand, kind))
        ends[side] = list(found.values())
    return ends


class Estimates(NamedTuple):
    """Estimators 2-4 for one repeat hub. `tiers` holds (copies, mean anchor depth) per depth tier."""
    link: int
    anchor: int
    low: int
    high: int
    tiers: list[tuple[int, float]]
    fallback: bool


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def estimate(hub_depth: float, ends: dict[str, list[WalkEnd]], depths: dict[str, float], baseline: float,
             tolerance: float = TIER_TOLERANCE) -> Estimates:
    """
    Estimates a repeat hub's copy number from where the walks out of its two sides end.

    link: the links to anchors, averaged over the two sides. anchor: the hub depth shared out over the depth tiers of
    its anchors, each tier weighted by the anchors it has on its better-linked side, so with one tier it is hub depth
    over mean anchor depth, and a tier whose anchor one side misses still gets its copy. With no anchors, hub depth
    over `baseline`, flagged as a fallback.
    """
    anchors = {side: [depths[e.contig] for e in side_ends if e.kind == "anchor"] for side, side_ends in ends.items()}
    link = max(1, round_half_up(sum(len(a) for a in anchors.values()) / 2))
    if not any(anchors.values()):
        anchor = max(1, round_half_up(hub_depth / baseline))
        return Estimates(link, anchor, min(link, anchor), max(link, anchor), [(anchor, baseline)], True)

    tiers = depth_tiers([d for side in anchors.values() for d in side], tolerance)
    # Anchors per tier on the side that reached more of them, scaled so the tiers' summed depth equals the hub's
    per_side = [(max(sum(d in t for d in side) for side in anchors.values()), sum(t) / len(t)) for t in tiers]
    scale = hub_depth / sum(n * depth for n, depth in per_side)
    tier_copies = [(max(1, round_half_up(n * scale)), depth) for n, depth in per_side]
    anchor = max(1, sum(copies for copies, _ in tier_copies))
    return Estimates(link, anchor, min(link, anchor), max(link, anchor), tier_copies, False)


class LocusHub(NamedTuple):
    """One locus target with the hub contigs it covers end to end (none for a target in unique sequence)."""
    locus_id: str
    family: str
    name: str
    hub: list[tuple[str, Strand]]
    old: float  # copy number as eris reports it now


@dataclass
class HubGroup:
    """The loci of one family that share a repeat hub, scored once (issue #14 reports a hub under several names)."""
    family: str
    names: set[str]
    locus_ids: list[str]
    hub: list[tuple[str, Strand]]
    old: float
    hub_depth: float = 0.0
    ends: dict[str, list[WalkEnd]] = field(default_factory=dict)
    estimates: Estimates | None = None

    @property
    def contigs(self) -> set[str]:
        return {contig for contig, _ in self.hub}


def group_by_hub(loci: list[LocusHub]) -> list[HubGroup]:
    """Merges loci of the same family whose hubs share a contig. A shared contig keeps the strand first read on it."""
    groups: list[HubGroup] = []
    for locus in loci:
        contigs = {contig for contig, _ in locus.hub}
        group = next((g for g in groups if g.family == locus.family and contigs & g.contigs), None)
        if group is None:
            groups.append(HubGroup(locus.family, {locus.name}, [locus.locus_id], list(locus.hub), locus.old))
            continue
        group.names.add(locus.name)
        group.locus_ids.append(locus.locus_id)
        group.hub += [(c, s) for c, s in locus.hub if c not in group.contigs]
        group.old = max(group.old, locus.old)
    return groups


def baseline_depth(genome: GenomeAssembly, repeats: set[str], min_length: int = MIN_ANCHOR_LENGTH) -> float:
    """Length-weighted depth of the contigs of at least `min_length` bp that are not a target's hub."""
    contigs = [c for c, n in genome.contig_lengths.items() if n >= min_length and c not in repeats]
    total = sum(genome.contig_lengths[c] for c in contigs)
    return float(sum(genome.contig_depths[c] * genome.contig_lengths[c] for c in contigs) / total)


def family_scores(truth: Counter[str], called: Counter[str]) -> tuple[float, float, float, dict[str, tuple[int, int]]]:
    """
    Copy-aware precision, recall and F1 over families, as `compute_metrics`, and the families that are wrong. It takes
    counts rather than loci, so a hub shared by several loci can be scored once.
    """
    families = sorted(set(truth) | set(called))
    tp = sum(min(truth[f], called[f]) for f in families)
    precision = tp / sum(called.values())
    recall = tp / sum(truth.values())
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, {f: (truth[f], called[f]) for f in families if truth[f] != called[f]}


def group_copies(group: HubGroup, estimator: str) -> int:
    if estimator == "old":
        return int(group.old)
    return getattr(group.estimates, estimator) if group.estimates else 1


def anchor_contigs(group: HubGroup) -> set[str]:
    return {e.contig for side in group.ends.values() for e in side if e.kind == "anchor"}


def disagreement_reason(group: HubGroup, depths: dict[str, float]) -> str:
    """Why link count and anchor depth disagree on a hub, read from where the walks out of each side ended."""
    est = group.estimates
    assert est is not None
    counts = {side: Counter(e.kind for e in ends) for side, ends in group.ends.items()}
    summed = {side: sum(depths[e.contig] for e in ends if e.kind == "anchor") for side, ends in group.ends.items()}
    reasons = []
    if counts["upstream"]["anchor"] != counts["downstream"]["anchor"]:
        reasons.append(f"the sides reach {counts['upstream']['anchor']} and {counts['downstream']['anchor']} "
                       f"anchors")
    for side, kinds in counts.items():
        if kinds["dead_end"] or kinds["tip"]:
            reasons.append(f"{kinds['dead_end'] + kinds['tip']} {side} walk(s) end in stubs before an anchor "
                           f"(links lost)")
        if kinds["repeat"]:
            reasons.append(f"{kinds['repeat']} {side} walk(s) run into another target's hub (not counted)")
    over = [side for side, s in summed.items() if s > group.hub_depth * 1.15]
    under = [side for side, s in summed.items() if s < group.hub_depth / 1.15]
    if over:
        reasons.append(f"{'/'.join(over)} anchors sum past the hub depth (extra links)")
    if under:
        reasons.append(f"{'/'.join(under)} anchors sum short of the hub depth (missing links)")
    if est.link < est.anchor and not under:
        reasons.append("hub depth over anchor depth rounds up (the hub reads deeper than its copies)")
    depth_sums = ", ".join(f"{side} {s:.0f}" for side, s in summed.items())
    reasons[0] = reasons[0][0].upper() + reasons[0][1:]
    return "; ".join(reasons) + f". Hub {group.hub_depth:.0f}x, summed anchors {depth_sums}."


def tier_check(groups: list[HubGroup], depths: dict[str, float], replicon_of: dict[str, str]) -> None:
    """At each tolerance, counts the hubs whose depth tiers put each replicon's anchors in a tier of their own."""
    hubs = []
    for group in groups:
        anchors = {c for c in anchor_contigs(group) if c in replicon_of}
        if len(anchors) >= 2:
            hubs.append(anchors)
    print(f"\nDepth tiers against replicons ({len(hubs)} hubs with 2+ anchors that map to one replicon; "
          f"pEC958 and pEC958B count as one, as their depths can't be told apart):")
    for tolerance in TOLERANCES:
        wrong = []
        for anchors in hubs:
            tiers = depth_tiers([depths[c] for c in anchors], tolerance)
            per_tier = [{replicon_of[c] for c in anchors if depths[c] in t} for t in tiers]
            if not (all(len(r) == 1 for r in per_tier) and len(tiers) == len({replicon_of[c] for c in anchors})):
                wrong.append(sorted((round(depths[c]), replicon_of[c]) for c in anchors))
        print(f"  tolerance {tolerance:<5} right {len(hubs) - len(wrong):>2}  wrong {len(wrong):>2}  {wrong}")


def map_to_replicons(genome: GenomeAssembly, contigs: set[str], reference: Path) -> dict[str, str]:
    """The replicon each contig maps to, for contigs that map to exactly one."""
    aligner = mappy.Aligner(str(reference), preset="asm5")
    replicon_of = {}
    for contig in contigs:
        hits = {REPLICONS[h.ctg] for h in aligner.map(str(genome[contig])) if h.mlen >= 0.9 * h.blen}
        if len(hits) == 1:
            replicon_of[contig] = hits.pop().replace("pEC958B", "pEC958")
    return replicon_of


def compare(label: str, genome: GenomeAssembly, truth_file: Path, db: TargetDatabase, families: dict[str, str],
            reference: Path | None) -> None:
    with Pipeline(db, max_feature_hops=0, mode="collapse", min_coverage=0.30) as pipeline:
        loci = [locus for locus in pipeline(genome) if locus.targets]
    lengths, depths = genome.contig_lengths, genome.contig_depths
    overlap = max((e.overlap for e in genome.edges), default=0)

    targets = []
    for locus in loci:
        for target in locus.targets:
            name = target.id.split(",")[0].split("_")[0]
            segments = [(s.contig, s.start, s.end, s.strand) for s in target.segments]
            targets.append((locus, target, LocusHub(locus.id, families.get(name, name), name,
                                            hub_contigs(segments, lengths, overlap), locus.fractional_depth)))
    repeats = {contig for _, _, hub in targets for contig, _ in hub.hub}
    baseline = baseline_depth(genome, repeats)
    graph = link_graph(list(genome.edges))

    groups = group_by_hub([hub for _, _, hub in targets])
    for group in groups:
        if group.hub:
            group.ends = walk_out(graph, group.hub, lengths, repeats - group.contigs)
            group.hub_depth = max(depths[c] for c in group.contigs)
            group.estimates = estimate(group.hub_depth, group.ends, depths, baseline)

    print(f"\n=== {label}: {genome.id}, truth {truth_file.name}; {len(loci)} loci, {len(groups)} scoring groups "
          f"(hubs or loci in unique sequence); baseline (contigs >= {MIN_ANCHOR_LENGTH} bp, not hubs) {baseline:.1f}x ===")
    print(f"{'family':<12}{'target':<10}{'kind':<9}{'old':>4}{'link':>5}{'anchor':>7}  combined   tiers (copies@depth)")
    # A locus can hold targets of several families, each in its own group
    group_of = {(lid, g.family): g for g in groups for lid in g.locus_ids}
    for locus, target, hub in sorted(targets, key=lambda x: (x[2].family, x[2].name)):
        est = group_of[(locus.id, hub.family)].estimates
        kind = "stitched" if len(target.segments) > 1 else "local"
        row = f"{hub.family:<12}{hub.name:<10}{kind:<9}{hub.old:>4.0f}"
        if est is None:
            print(f"{row}{1:>5}{1:>7}  1          unique sequence")
            continue
        combined = f"{est.anchor}" + (f" [{est.low}-{est.high}]" if est.low != est.high else "")
        tiers = " + ".join(f"{c}@{d:.0f}" for c, d in est.tiers) + (" FALLBACK" if est.fallback else "")
        print(f"{row}{est.link:>5}{est.anchor:>7}  {combined:<10} {tiers}")

    print("\nHubs where link count and anchor depth disagree:")
    for group in groups:
        if group.estimates and group.estimates.link != group.estimates.anchor:
            print(f"  {group.family} ({'/'.join(sorted(group.names))}): link {group.estimates.link}, anchor "
                  f"{group.estimates.anchor}. {disagreement_reason(group, depths)}")

    truth: Counter[str] = Counter()
    with open(truth_file) as bed:
        for line in bed:
            if line.strip():
                name = line.split("\t")[3].split("_")[0]
                truth[families.get(name, name)] += 1
    print("\nFamily-level scores, each hub once (low/high: ends of the combined range):")
    for estimator in ESTIMATORS:
        called: Counter[str] = Counter()
        for group in groups:
            called[group.family] += group_copies(group, estimator)
        p, r, f1, wrong = family_scores(truth, called)
        print(f"  {estimator:<7} P={p:.3f} R={r:.3f} F1={f1:.3f}  wrong (truth, eris): {wrong}")
    p, r, f1 = compute_metrics(truth_file, loci, families, count="sum")
    print(f"  old, as tests/test_integration.py scores it (every locus, summed): P={p:.3f} R={r:.3f} F1={f1:.3f}")

    if reference:
        anchors = set().union(*(anchor_contigs(g) for g in groups))
        tier_check(groups, depths, map_to_replicons(genome, anchors, reference))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    db = TargetDatabase(args.data / "IS.mmi", FeatureType.CDS, indexing_threads=1)
    # Target names are '<name>_<family>_<group>', e.g. 'IS600_IS3_IS3'
    families = {n.rsplit("_", 2)[0]: n.rsplit("_", 2)[1] for n in db.aligner.seq_names if n.count("_") >= 2}
    for label, assembly, truth in ASSEMBLIES:
        reference = args.data / REFERENCE if label == "real reads" else None
        compare(label, GenomeAssembly.from_file(args.data / assembly), args.data / truth, db, families, reference)


if __name__ == "__main__":
    main()
