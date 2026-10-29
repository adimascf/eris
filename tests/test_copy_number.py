from Bio.Seq import Seq

from eris.constants import FeatureType, Strand
from eris.copy_number import assign_copy_numbers
from eris.graph import Edge
from eris.io import GenomeAssembly
from eris.pipeline import GenomicFeature, LocationSegment, Locus

F, R = Strand.FORWARD, Strand.REVERSE


def genome(contigs: dict[str, tuple[int, float]], edges: list[Edge]) -> GenomeAssembly:
    """An assembly from {contig: (length, depth)} and its links."""
    lengths = {c: n for c, (n, _) in contigs.items()}
    depths = {c: d for c, (_, d) in contigs.items()}
    return GenomeAssembly("test", {c: Seq("A" * n) for c, n in lengths.items()}, edges, depths, lengths)


def locus(name: str, *segments: tuple[str, int, int, Strand]) -> Locus:
    target = GenomicFeature(name, FeatureType.CDS, [LocationSegment(*s) for s in segments])
    first, last = segments[0], segments[-1]
    return Locus(f"locus_{name}", "|".join(s[0] for s in segments), first[1], last[2], [target], [], [], [])


def hub_with_anchors(hub: str, upstream: list[float], downstream: list[float], prefix: str = "") -> tuple[dict, list]:
    """A hub contig with one 5 kb anchor linked into it per upstream depth, and one linked out per downstream depth."""
    contigs, edges = {}, []
    for i, depth in enumerate(upstream):
        contigs[f"{prefix}u{i}"] = (5000, depth)
        edges.append(Edge(f"{prefix}u{i}", F, hub, F))
    for i, depth in enumerate(downstream):
        contigs[f"{prefix}d{i}"] = (5000, depth)
        edges.append(Edge(hub, F, f"{prefix}d{i}", F))
    return contigs, edges


def test_a_hub_shares_its_depth_out_over_the_depth_tiers_of_its_anchors():
    # IS1 on the real-read EC958 assembly: 3 chromosome copies at ~37x and 2 pEC958 copies at ~76x in one 272x hub
    contigs, edges = hub_with_anchors("hub", [30, 42, 44, 75, 77], [32, 44, 47, 78, 79])
    loci = [locus("IS1", ("hub", 0, 768, F))]

    assign_copy_numbers(loci, genome({"hub": (768, 272), **contigs}, edges))

    copies = loci[0].copy_number
    assert (copies.estimate, copies.low, copies.high, copies.fallback) == (5, 5, 5, False)
    assert [n for n, _ in copies.tiers] == [3, 2]
    assert loci[0].fractional_depth == 5


def test_a_hub_without_anchors_is_measured_against_the_unique_contigs_and_flagged():
    # Both sides break into stubs; the genome's unique sequence is two contigs at 30x and 45x, length-weighted 40x
    contigs = {"hub": (1300, 160), "s1": (300, 20), "s2": (300, 20), "s3": (300, 20),
               "chrA": (20000, 30), "chrB": (40000, 45)}
    edges = [Edge("s1", F, "hub", F), Edge("s2", F, "s1", F), Edge("s3", F, "s2", F)]
    loci = [locus("ISEc12", ("hub", 0, 1300, F))]

    assign_copy_numbers(loci, genome(contigs, edges))

    copies = loci[0].copy_number
    assert (copies.estimate, copies.fallback, copies.tiers) == (4, True, ((4, 40.0),))


def test_a_hub_reports_a_range_when_stubs_hide_links_from_the_count():
    # ISEc12: 4 chromosome copies at 153x, but two upstream flanks break into stubs, so only 2 + 4 links reach anchors
    contigs, edges = hub_with_anchors("hub", [30, 39], [30, 29, 41, 45])
    contigs.update({"s1": (300, 20), "s2": (300, 20), "s1b": (400, 20), "s2b": (400, 20)})
    edges += [Edge("s1", F, "hub", F), Edge("s1b", F, "s1", F), Edge("s2", F, "hub", F), Edge("s2b", F, "s2", F)]
    loci = [locus("ISEc12", ("hub", 0, 1300, F))]

    assign_copy_numbers(loci, genome({"hub": (1300, 153), **contigs}, edges))

    copies = loci[0].copy_number
    assert (copies.estimate, copies.low, copies.high) == (4, 3, 4)


def test_a_target_in_unique_sequence_is_one_copy():
    # A plasmid IS with flanks on its own contig: the contig's 76x depth is the plasmid's, not extra copies
    loci = [locus("IS1", ("plasmid", 2000, 2768, F))]

    assign_copy_numbers(loci, genome({"plasmid": (60000, 76)}, []))

    assert loci[0].copy_number.estimate == 1
    assert loci[0].copy_number.hub == frozenset()
    assert loci[0].fractional_depth == 1


def test_an_anchor_between_two_inverted_copies_counts_once_per_link():
    # IS26 around 2419936: two copies collapsed into one hub, inverted around anchor A, so A is reached twice
    contigs = {"hub": (820, 150), "A": (11000, 75), "B": (3000, 75), "C": (3000, 75)}
    edges = [Edge("hub", F, "A", F), Edge("A", F, "hub", R), Edge("B", F, "hub", F), Edge("C", F, "hub", F)]
    loci = [locus("IS26", ("hub", 0, 820, F))]

    assign_copy_numbers(loci, genome(contigs, edges))

    copies = loci[0].copy_number
    assert (copies.estimate, copies.low, copies.high) == (2, 2, 2)


def test_a_walk_stops_at_another_targets_hub():
    # One IS26 flank runs into the ISEc9 hub after a 140 bp stub, so that link reaches no anchor
    contigs, edges = hub_with_anchors("IS26hub", [75, 75], [75])
    contigs.update({"stub": (140, 150), "ISEc9hub": (1100, 115), "far": (5000, 75)})
    edges += [Edge("IS26hub", F, "stub", F), Edge("stub", F, "ISEc9hub", F), Edge("ISEc9hub", F, "far", F)]
    loci = [locus("IS26", ("IS26hub", 0, 820, F)), locus("ISEc9", ("ISEc9hub", 0, 1100, F))]

    assign_copy_numbers(loci, genome({"IS26hub": (820, 150), **contigs}, edges))

    copies = loci[0].copy_number
    assert (copies.estimate, copies.low, copies.high) == (2, 2, 2)  # 3 anchor links round half up to 2
    assert loci[1].copy_number.hub == frozenset({"ISEc9hub"})


def test_loci_that_share_a_hub_share_its_copy_number():
    # Issue #14: IS26 and IS15 both walk through the IS6 hub; IS15 also covers a second hub contig
    contigs, edges = hub_with_anchors("h1", [75] * 8, [75] * 8)
    contigs.update({"h2": (500, 600)})
    edges.append(Edge("h2", F, "h1", F))
    loci = [locus("IS26", ("h1", 0, 820, F)), locus("IS15", ("h2", 0, 500, F), ("h1", 0, 820, F))]

    assign_copy_numbers(loci, genome({"h1": (820, 600), **contigs}, edges))

    assert loci[0].copy_number is loci[1].copy_number
    assert loci[0].copy_number.estimate == 8
    assert loci[0].copy_number.hub == frozenset({"h1", "h2"})
