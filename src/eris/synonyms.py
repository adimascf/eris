"""
Module for collapsing synonymous IS target names (e.g. IS1A / IS1B, IS30 / IS30D) onto a single name.

Names are only ever linked when the ISfinder table (IS.csv) says so: similar-looking names are NOT assumed to be
synonyms (e.g. IS1A and IS1F are distinct IS1-family elements, ~90% identical).

Without this, two alignments of the same physical insertion can be reported under different names
(minimap2 picks whichever isoform scores best on each contig), which stops them being stitched together
and inflates the per-name counts.
"""
import csv
from dataclasses import replace
from os import environ
from pathlib import Path
import ssl
from typing import Union
from urllib.error import URLError
from urllib.request import urlopen

import certifi
import numpy as np

from eris.alignment import AlignmentBatch


# Constants ------------------------------------------------------------------------------------------------------------
ISFINDER_TABLE_URL = "https://raw.githubusercontent.com/thanhleviet/ISfinder-sequences/master/IS.csv"


# Functions ------------------------------------------------------------------------------------------------------------
def fetch_isfinder_table(cache_dir: Union[str, Path, None] = None, url: str = ISFINDER_TABLE_URL) -> Path:
    """
    Returns a local copy of ISfinder's IS.csv, downloading it on first use.

    The table is cached (default: $XDG_CACHE_HOME/eris or ~/.cache/eris) so later runs work offline;
    delete the cached file to pick up a newer ISfinder release.
    """
    cache_dir = Path(cache_dir) if cache_dir else Path(environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'eris'
    path = cache_dir / 'IS.csv'
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        # certifi's CA bundle: python.org macOS builds ship without system certificates
        context = ssl.create_default_context(cafile=certifi.where())
        try:
            with urlopen(url, timeout=60, context=context) as resp:
                data = resp.read()
        except URLError as e:
            raise RuntimeError(f"Could not download the ISfinder table from {url} ({e.reason}). "
                               f"Download it manually and pass it as --merge-synonyms IS.csv") from e
        tmp = path.with_suffix('.part')
        tmp.write_bytes(data)
        tmp.replace(path)  # never leave a half-written table in the cache
    return path


# Classes --------------------------------------------------------------------------------------------------------------
class _UnionFind:
    __slots__ = ('parent',)

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


class SynonymResolver:
    """
    Maps every IS target name to the FIRST name of its synonym group seen during a run.

    Synonym groups come only from the 'Synomyns' and 'Iso' columns of the ISfinder IS.csv table, matched on the
    element name (the first '_'-delimited field of the target name). Names absent from the table stay as they are.

    Example:
        >>> r = SynonymResolver('IS.csv')  # IS.csv row: IS1B ... Iso=IS1A
        >>> r.canonical('IS1B_IS1_unknown')  # seen first, so it names the group
        'IS1B_IS1_unknown'
        >>> r.canonical('IS1A_IS1_unknown')
        'IS1B_IS1_unknown'
        >>> r.canonical('IS1F_IS1_unknown')  # not linked to IS1A/IS1B in ISfinder
        'IS1F_IS1_unknown'
    """
    __slots__ = ('_uf', '_first_seen')

    def __init__(self, table: Union[str, Path]):
        self._uf = _UnionFind()
        self._first_seen: dict[str, str] = {}  # group root -> first target name observed
        self._load_isfinder_table(Path(table))

    def _load_isfinder_table(self, path: Path):
        """Links names via the 'Synomyns' and 'Iso' columns of ISfinder's IS.csv."""
        with open(path, newline='') as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            cols = {h.strip().lower(): i for i, h in enumerate(header)}
            name_i = cols.get('name', 1)
            syn_i = cols.get('synomyns', cols.get('synonyms', 4))
            iso_i = cols.get('iso', 5)
            for row in reader:
                if len(row) <= max(name_i, syn_i, iso_i) or not (name := row[name_i].strip()):
                    continue
                for field in (row[syn_i], row[iso_i]):
                    for other in field.split(','):
                        if (other := other.strip()) and other.upper() != 'NA':
                            self._uf.union(name, other)

    def canonical(self, target_name: str) -> str:
        """Returns the first-seen target name of this name's synonym group (registering it if new)."""
        root = self._uf.find(target_name.split('_')[0])
        return self._first_seen.setdefault(root, target_name)

    def rename(self, batch: AlignmentBatch) -> AlignmentBatch:
        """Renames the IS (query) side of a swapped batch, registering names in contig-position order."""
        for idx in batch.t_starts.argsort(kind='stable'):
            self.canonical(batch.q_names[idx])
        return replace(batch, q_names=np.array([self.canonical(n) for n in batch.q_names], dtype=object))
