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


class Exit(NamedTuple):
    """Where one walk out of a repeat hub ends: an anchor, a repeat (another target's hub), a dead end or a tip."""
    contig: str
    strand: Strand
    kind: str


def link_graph(edges: list[Edge]) -> Graph:
    """The assembly graph with each link on both strands, as `TopologyEngine` builds it."""
    return Graph(edges + [e.complement() for e in edges])


def find_exits(graph: Graph, hub: list[tuple[str, Strand]], lengths: dict[str, int], repeats: set[str],
               min_anchor: int = MIN_ANCHOR_LENGTH, stub_budget: int = STUB_BUDGET) -> dict[str, list[Exit]]:
    """
    Walks out of a repeat hub on each side, following edge orientation so a walk never turns back towards the hub's
    side of a contig (an undirected walk out of the IS26 hub reaches the rRNA repeat). `hub` holds each hub contig
    with the strand the target reads it on. A walk ends at the first contig of at least `min_anchor` bp (an anchor),
    at a contig in `repeats`, or when the stubs it crossed exceed `stub_budget`. Each distinct oriented end is one
    link, so an anchor between two inverted copies counts twice.
    """
    hub_contigs = {contig for contig, _ in hub}
    exits = {}
    for side, sign in (("upstream", -1), ("downstream", 1)):
        found: dict[tuple[str, Strand], Exit] = {}
        stack = [(edge, 0) for contig, strand in hub for edge in graph.get_neighbors(contig)
                 if edge.u_strand == strand * sign]
        seen = set()
        while stack:
            edge, used = stack.pop()
            node = (edge.v, edge.v_strand)
            if node in seen or edge.v in hub_contigs:
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
            found.setdefault(node, Exit(edge.v, edge.v_strand, kind))
        exits[side] = list(found.values())
    return exits


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


def estimate(hub_depth: float, exits: dict[str, list[Exit]], depths: dict[str, float], baseline: float,
             tolerance: float = TIER_TOLERANCE) -> Estimates:
    """
    Estimates a repeat hub's copy number from the exits on its two sides.

    link: the links to anchors and repeats, averaged over the two sides. anchor: the hub depth shared out over the depth
    tiers of its anchors, each tier's share of the copies set by its share of the links, so with one tier it is hub
    depth over mean anchor depth. With no anchors, hub depth over `baseline`, flagged as a fallback.
    """
    links = [e for side in exits.values() for e in side if e.kind in ("anchor", "repeat")]
    link = max(1, round_half_up(len(links) / 2))
    anchor_depths = [depths[e.contig] for e in links if e.kind == "anchor"]
    if not anchor_depths:
        anchor = max(1, round_half_up(hub_depth / baseline))
        return Estimates(link, anchor, min(link, anchor), max(link, anchor), [(anchor, baseline)], True)

    tiers = depth_tiers(anchor_depths, tolerance)
    # Copies per side in each tier, scaled so the tiers' summed depth equals the hub's
    per_side = [(len(t) / 2, sum(t) / len(t)) for t in tiers]
    scale = hub_depth / sum(n * depth for n, depth in per_side)
    tier_copies = [(round_half_up(n * scale), depth) for n, depth in per_side]
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
    estimates: Estimates | None = field(default=None, compare=False)


def group_by_hub(loci: list[LocusHub]) -> list[HubGroup]:
    """Merges loci of the same family whose hubs share a contig. A shared contig keeps the strand first read on it."""
    groups: list[HubGroup] = []
    for locus in loci:
        contigs = {contig for contig, _ in locus.hub}
        group = next((g for g in groups if g.family == locus.family and contigs & {c for c, _ in g.hub}), None)
        if group is None:
            groups.append(HubGroup(locus.family, {locus.name}, [locus.locus_id], list(locus.hub), locus.old))
            continue
        group.names.add(locus.name)
        group.locus_ids.append(locus.locus_id)
        group.hub += [(c, s) for c, s in locus.hub if c not in {h for h, _ in group.hub}]
        group.old = max(group.old, locus.old)
    return groups


def baseline_depth(genome: GenomeAssembly, repeats: set[str], min_length: int = MIN_ANCHOR_LENGTH) -> float:
    """Length-weighted depth of the contigs of at least `min_length` bp that are not a target's hub."""
    contigs = [c for c, n in genome.contig_lengths.items() if n >= min_length and c not in repeats]
    total = sum(genome.contig_lengths[c] for c in contigs)
    return float(sum(genome.contig_depths[c] * genome.contig_lengths[c] for c in contigs) / total)


def family_scores(truth: Counter[str], eris: Counter[str]) -> tuple[float, float, float, dict[str, tuple[int, int]]]:
    """Copy-aware precision, recall and F1 over families, as `compute_metrics`, and the families that are wrong."""
    families = sorted(set(truth) | set(eris))
    tp = sum(min(truth[f], eris[f]) for f in families)
    precision = tp / sum(eris.values())
    recall = tp / sum(truth.values())
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, {f: (truth[f], eris[f]) for f in families if truth[f] != eris[f]}


def group_copies(group: HubGroup, estimator: str) -> int:
    if estimator == "old":
        return int(group.old)
    return getattr(group.estimates, estimator) if group.estimates else 1


def disagreement_reason(est: Estimates, exits: dict[str, list[Exit]], hub_depth: float,
                        depths: dict[str, float]) -> str:
    """What the walks out of a hub found on each side, for a hub where link count and anchor depth disagree."""
    parts = []
    for side, side_exits in exits.items():
        kinds = Counter(e.kind for e in side_exits)
        summed = sum(depths[e.contig] for e in side_exits if e.kind == "anchor")
        doubled = sum(n - 1 for n in Counter(e.contig for e in side_exits if e.kind == "anchor").values())
        note = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
        note += f"; hub/anchors {hub_depth:.0f}/{summed:.0f}" if summed else ""
        note += f"; {doubled} anchor(s) reached twice" if doubled else ""
        parts.append(f"{side}: {note}")
    return " | ".join(parts)


def tier_check(groups: list[HubGroup], group_exits: list[dict], depths: dict[str, float],
               replicon_of: dict[str, str]) -> None:
    """At each tolerance, counts the hubs whose depth tiers put each replicon's anchors in a tier of their own."""
    hubs = []
    for exits in group_exits:
        anchors = {e.contig for side in exits.values() for e in side if e.kind == "anchor" and e.contig in replicon_of}
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
            targets.append((locus, LocusHub(locus.id, families.get(name, name), name,
                                            hub_contigs(segments, lengths, overlap), locus.fractional_depth)))
    repeats = {contig for _, t in targets for contig, _ in t.hub}
    baseline = baseline_depth(genome, repeats)
    graph = link_graph(list(genome.edges))

    groups = group_by_hub([t for _, t in targets])
    group_exits = []
    for group in groups:
        exits = find_exits(graph, group.hub, lengths, repeats - {c for c, _ in group.hub}) if group.hub else {}
        group_exits.append(exits)
        if group.hub:
            hub_depth = max(depths[c] for c, _ in group.hub)
            group.estimates = estimate(hub_depth, exits, depths, baseline)

    print(f"\n=== {label}: {genome.id}, truth {truth_file.name}; {len(loci)} loci, {len(groups)} scoring groups "
          f"(hubs or loci in unique sequence); baseline (contigs >= {MIN_ANCHOR_LENGTH} bp, not hubs) {baseline:.1f}x ===")
    print(f"{'family':<12}{'target':<10}{'kind':<9}{'old':>4}{'link':>5}{'anchor':>7}  combined   tiers (copies@depth)")
    group_of = {lid: g for g in groups for lid in g.locus_ids}
    for locus, t in sorted(targets, key=lambda x: (x[1].family, x[1].name)):
        g = group_of[locus.id]
        kind = "stitched" if len(locus.targets[0].segments) > 1 else "local"
        e = g.estimates
        if e is None:
            print(f"{t.family:<12}{t.name:<10}{kind:<9}{t.old:>4.0f}{1:>5}{1:>7}  1          unique sequence")
            continue
        combined = f"{e.anchor}" + (f" [{e.low}-{e.high}]" if e.low != e.high else "")
        tiers = " + ".join(f"{c}@{d:.0f}" for c, d in e.tiers) + (" FALLBACK" if e.fallback else "")
        print(f"{t.family:<12}{t.name:<10}{kind:<9}{t.old:>4.0f}{e.link:>5}{e.anchor:>7}  {combined:<10} {tiers}")

    print("\nHubs where link count and anchor depth disagree:")
    for group, exits in zip(groups, group_exits):
        e = group.estimates
        if e and e.link != e.anchor:
            hub_depth = max(depths[c] for c, _ in group.hub)
            print(f"  {group.family} ({'/'.join(sorted(group.names))}), hub {hub_depth:.0f}x: link {e.link}, "
                  f"anchor {e.anchor}. {disagreement_reason(e, exits, hub_depth, depths)}")

    truth: Counter[str] = Counter()
    for line in open(truth_file):
        if line.strip():
            name = line.split("\t")[3].split("_")[0]
            truth[families.get(name, name)] += 1
    print("\nFamily-level scores, each hub once (low/high: ends of the combined range):")
    for estimator in ESTIMATORS:
        eris: Counter[str] = Counter()
        for group in groups:
            eris[group.family] += group_copies(group, estimator)
        p, r, f1, wrong = family_scores(truth, eris)
        print(f"  {estimator:<7} P={p:.3f} R={r:.3f} F1={f1:.3f}  wrong (truth, eris): {wrong}")
    p, r, f1 = compute_metrics(truth_file, loci, families, count="sum")
    print(f"  old, as tests/test_integration.py scores it (every locus, summed): P={p:.3f} R={r:.3f} F1={f1:.3f}")

    if reference:
        anchors = {e.contig for exits in group_exits for side in exits.values() for e in side if e.kind == "anchor"}
        tier_check(groups, group_exits, depths, map_to_replicons(genome, anchors, reference))


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
