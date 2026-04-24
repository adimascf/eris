from collections import defaultdict
from typing import Iterable, Generator, Self
from dataclasses import dataclass
from re import compile as re_compile

import numpy as np

from mappy import Alignment

from eris.constants import Strand
from eris.interval import IntervalBatch


# Classes --------------------------------------------------------------------------------------------------------------
class Cigar:
    """
    Represents a CIGAR (Compact Idiosyncratic Gapped Alignment Report) string.

    This class provides tools for parsing and iterating over CIGAR operations,
    calculating consumed lengths on query and target sequences.

    Example:
        >>> cigar = Cigar("10M2I5M")
        >>> list(cigar)
        [('M', 10, 10, 10, 10), ('I', 2, 12, 10, 10), ('M', 5, 17, 15, 15)]
    """
    _OPS = {
        'M': (True, True),
        'I': (True, False),
        'D': (False, True),
        'N': (False, True),
        'S': (True, False),
        'H': (False, False),
        'P': (False, False),
        '=': (True, True),
        'X': (True, True),
        'B': (False, False),
    }
    _REGEX = re_compile(r'(\d+)([MIDNSHP=XB])')
    _SWAP_MAP = str.maketrans('ID', 'DI')

    __slots__ = ('_data',)

    def __init__(self, data: str):
        """
        Initialize a Cigar object.

        Args:
            data: The CIGAR string (e.g., "100M2I10D").
        """
        self._data = data

    def __str__(self) -> str:
        return self._data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"

    def __iter__(self) -> Generator[tuple[str, int, int, int, int], None, None]:
        """
        Iterates over the CIGAR string and yields parsed operations.

        Yields:
            tuple: (op, count, query_consumed, target_consumed, aln_consumed)
                - op: The CIGAR operation character (e.g., 'M', 'I', 'D').
                - count: The number of bases for this operation.
                - query_consumed: Cumulative bases consumed in the query.
                - target_consumed: Cumulative bases consumed in the target.
                - aln_consumed: Cumulative aligned (matching/mismatching) bases.
        """
        q_len, t_len, aln_len = 0, 0, 0
        for match in self._REGEX.finditer(self._data):
            n = int(match.group(1))
            op = match.group(2)
            consume_query, consume_target = self._OPS[op]

            if consume_query: q_len += n
            if consume_target: t_len += n
            if consume_query and consume_target: aln_len += n

            yield op, n, q_len, t_len, aln_len


@dataclass(slots=True, frozen=True)
class AlignmentRecord:
    """
    A lightweight, self-aware view of a single alignment record.

    Attributes:
        idx: Original index within the source AlignmentBatch.
        q_name: Query sequence name.
        q_length: Total length of the query sequence.
        q_start: Start position on the query.
        q_end: End position on the query.
        t_name: Target (reference) sequence name.
        t_length: Total length of the target sequence.
        t_start: Start position on the target.
        t_end: End position on the target.
        strand: Alignment orientation (Strand.FORWARD or Strand.REVERSE).
        length: Total alignment block length.
        match: Number of matching bases.
        mismatch: Number of mismatches (NM tag).
        quality: Mapping quality (MAPQ).
        cigar: CIGAR string.
    """
    idx: int
    q_name: str
    q_length: int
    q_start: int
    q_end: int
    t_name: str
    t_length: int
    t_start: int
    t_end: int
    strand: Strand
    length: int
    match: int
    mismatch: int
    quality: int
    cigar: str  # Kept as str to match the array type, can be wrapped in Cigar() if needed

    @property
    def is_partial(self) -> bool:
        """True if the alignment covers less than 90% of the query sequence."""
        return self.length < (self.q_length * 0.9)

    @property
    def hangs_5p(self) -> bool:
        """True if the alignment starts within 15bp of the target start."""
        return self.t_start <= 15

    @property
    def hangs_3p(self) -> bool:
        """True if the alignment ends within 15bp of the target end."""
        return (self.t_length - self.t_end) <= 15


@dataclass(frozen=True, slots=True)
class AlignmentBatch:
    """
    A high-performance batch of alignments stored in NumPy arrays.

    Uses a Structure-of-Arrays (SoA) layout to enable fast vectorized operations
    and efficient filtering of large alignment sets.

    Example:
        >>> batch = AlignmentBatch.from_mappy("read1", 100, alignments)
        >>> filtered = batch.filter(batch.qualities > 30)
        >>> scores = batch.scores
    """
    q_names: np.ndarray
    q_lengths: np.ndarray
    q_starts: np.ndarray
    q_ends: np.ndarray
    t_names: np.ndarray
    t_lengths: np.ndarray
    t_starts: np.ndarray
    t_ends: np.ndarray
    strands: np.ndarray
    lengths: np.ndarray
    matches: np.ndarray
    mismatches: np.ndarray
    qualities: np.ndarray
    cigars: np.ndarray

    def __len__(self) -> int:
        return len(self.q_starts)

    @property
    def scores(self) -> np.ndarray:
        """Vectorized alignment score (matches - mismatches)."""
        return self.matches - self.mismatches

    @classmethod
    def from_mappy(cls, q_name: str, q_length: int, alignments: Iterable[Alignment]) -> Self:
        """
        Create an AlignmentBatch from mappy Alignment objects.

        Args:
            q_name: Name of the query sequence.
            q_length: Length of the query sequence.
            alignments: Iterable of mappy.Alignment objects.

        Returns:
            AlignmentBatch: A new batch containing the alignments.
        """
        # OPTIMIZATION: List comprehension + zip is significantly faster than 14 .append() calls per hit
        data = [
            (q_name, q_length, h.q_st, h.q_en, h.ctg, h.ctg_len, h.r_st, h.r_en,
             h.strand, h.blen, h.mlen, h.NM, h.mapq, h.cigar_str)
            for h in alignments
        ]

        if not data:
            raise ValueError("Cannot initialize AlignmentBatch with empty alignments")

        qn, ql, qs, qe, tn, tl, ts, te, st, bl, ml, nm, mq, cg = zip(*data)

        return cls(
            q_names=np.array(qn, dtype=object),
            q_lengths=np.array(ql, dtype=np.int32),
            q_starts=np.array(qs, dtype=np.int32),
            q_ends=np.array(qe, dtype=np.int32),
            t_names=np.array(tn, dtype=object),
            t_lengths=np.array(tl, dtype=np.int32),
            t_starts=np.array(ts, dtype=np.int32),
            t_ends=np.array(te, dtype=np.int32),
            strands=np.array(st, dtype=np.int8),
            lengths=np.array(bl, dtype=np.int32),
            matches=np.array(ml, dtype=np.int32),
            mismatches=np.array(nm, dtype=np.int32),
            qualities=np.array(mq, dtype=np.int32),
            cigars=np.array(cg, dtype=object)
        )

    @classmethod
    def concat(cls, batches: Iterable['AlignmentBatch']) -> Self:
        """Concatenate multiple AlignmentBatch objects into one."""
        batches = list(batches)
        if not batches:
            raise ValueError("Cannot concatenate an empty iterable of batches")

        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            first_val = getattr(batches[0], field_name)
            if isinstance(first_val, np.ndarray):
                kwargs[field_name] = np.concatenate([getattr(b, field_name) for b in batches])
            else:
                if any(getattr(b, field_name) != first_val for b in batches):
                    raise ValueError(f"Cannot concatenate batches with mismatched '{field_name}' values")
                kwargs[field_name] = first_val

        return cls(**kwargs)

    def filter(self, mask: np.ndarray) -> 'AlignmentBatch':
        """Return a new batch containing only elements where mask is True."""
        return AlignmentBatch(
            q_names=self.q_names[mask],
            q_lengths=self.q_lengths[mask],
            q_starts=self.q_starts[mask],
            q_ends=self.q_ends[mask],
            t_names=self.t_names[mask],
            t_lengths=self.t_lengths[mask],
            t_starts=self.t_starts[mask],
            t_ends=self.t_ends[mask],
            strands=self.strands[mask],
            lengths=self.lengths[mask],
            matches=self.matches[mask],
            mismatches=self.mismatches[mask],
            qualities=self.qualities[mask],
            cigars=self.cigars[mask]
        )

    def filter_out(self, mask: np.ndarray) -> 'AlignmentBatch':
        """Returns a new batch excluding the masked items."""
        return self.filter(~mask)

    def cull_overlaps(self, max_overlap_fraction: float = 0.1) -> 'AlignmentBatch':
        """
        Greedily culls overlapping alignments on the target.

        Prioritizes alignments with higher match scores. Useful for resolving
        competing alignments for the same query region.

        Args:
            max_overlap_fraction: Maximum allowed overlap fraction before culling.
        """
        n = len(self)
        if n < 2: return self
        kept_mask = np.zeros(n, dtype=bool)
        kept_intervals = defaultdict(list)
        for idx in np.argsort(self.scores, kind='stable')[::-1]:
            t_name = self.t_names[idx]
            s, e = self.t_starts[idx], self.t_ends[idx]
            length = e - s
            if length <= 0: continue
            kept = kept_intervals[t_name]

            overlap_found = False
            for ks, ke in kept:
                overlap = min(e, ke) - max(s, ks)
                if overlap > 0 and (overlap / length) > max_overlap_fraction:
                    overlap_found = True
                    break
            if overlap_found: continue

            kept.append((s, e))
            kept_mask[idx] = True
        return self.filter(kept_mask)

    def swap_sides(self) -> 'AlignmentBatch':
        """
        Swaps query and target roles in the alignment records.

        This is used when query sequences (contigs) are being treated as targets
        and vice-versa, which is common in reciprocal mapping.
        """
        return AlignmentBatch(
            q_names=self.t_names,
            q_lengths=self.t_lengths,
            q_starts=self.t_starts,
            q_ends=self.t_ends,
            t_names=self.q_names,
            t_lengths=self.q_lengths,
            t_starts=self.q_starts,
            t_ends=self.q_ends,
            strands=self.strands,
            lengths=self.lengths,
            matches=self.matches,
            mismatches=self.mismatches,
            qualities=self.qualities,
            cigars=np.array([c.translate(Cigar._SWAP_MAP) for c in self.cigars], dtype=object)
        )

    def split(self, by_query: bool = False) -> Iterable[tuple[str, 'AlignmentBatch']]:
        """Splits a batch into separate batches by target or query name."""
        key_array = self.q_names if by_query else self.t_names
        for key in np.unique(key_array):
            yield key, self.filter(key_array == key)

    def to_intervals(self, by_query: bool = False) -> IntervalBatch:
        """
        Converts the batch into a high-performance IntervalBatch.

        Args:
            by_query: If True, use query coordinates. Otherwise, use target coordinates.
        """
        starts = self.q_starts if by_query else self.t_starts
        ends = self.q_ends if by_query else self.t_ends

        return IntervalBatch(
            starts=starts.copy(),
            ends=ends.copy(),
            strands=self.strands.copy(),
            # CRITICAL: Ensures we can map relational queries back to this alignment record!
            original_indices=np.arange(len(self), dtype=np.int32)
        )

    def get_record(self, idx: int) -> AlignmentRecord:
        """Retrieve a single AlignmentRecord by its batch index."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Batch index out of range")

        return AlignmentRecord(
            idx=idx,
            q_name=self.q_names[idx],
            q_length=self.q_lengths[idx],
            q_start=self.q_starts[idx],
            q_end=self.q_ends[idx],
            t_name=self.t_names[idx],
            t_length=self.t_lengths[idx],
            t_start=self.t_starts[idx],
            t_end=self.t_ends[idx],
            strand=self.strands[idx],
            length=self.lengths[idx],
            match=self.matches[idx],
            mismatch=self.mismatches[idx],
            quality=self.qualities[idx],
            cigar=self.cigars[idx]
        )