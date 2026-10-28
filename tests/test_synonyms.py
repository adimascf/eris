import numpy as np
import pytest

from eris.alignment import AlignmentBatch
from eris.synonyms import SynonymResolver, fetch_isfinder_table

# Rows copied from ISfinder's IS.csv
ISFINDER_ROWS = (
    'N°,Name,Family,Group,Synomyns,Iso,Origin\n'
    '44,IS1A,IS1,,"IS1E, IS1K",,Escherichia coli\n'
    '45,IS1B,IS1,,IS1C,IS1A,Escherichia coli\n'
    '49,IS1F,IS1,,IS1T,IS1X1,Escherichia coli\n'
    '79,IS30D,IS30,,IS30,IS30,Escherichia coli\n'
    '1,ISEc52,IS3,,NA,ISEc53,Escherichia coli\n'
)


@pytest.fixture
def resolver(tmp_path):
    table = tmp_path / 'IS.csv'
    table.write_text(ISFINDER_ROWS)
    return SynonymResolver(table)


def _batch(q_names, t_starts):
    n = len(q_names)
    ints = lambda v: np.full(n, v, dtype=np.int32)
    return AlignmentBatch(
        q_names=np.array(q_names, dtype=object), q_lengths=ints(768), q_starts=ints(0), q_ends=ints(700),
        t_names=np.array(['ctg1'] * n, dtype=object), t_lengths=ints(10_000),
        t_starts=np.array(t_starts, dtype=np.int32), t_ends=np.array(t_starts, dtype=np.int32) + 700,
        strands=np.ones(n, dtype=np.int8), lengths=ints(700), matches=ints(690), mismatches=ints(10),
        qualities=ints(60), cigars=np.array(['700M'] * n, dtype=object),
    )


def test_first_seen_name_wins(resolver):
    assert resolver.canonical('IS1B_IS1_unknown') == 'IS1B_IS1_unknown'
    assert resolver.canonical('IS1A_IS1_unknown') == 'IS1B_IS1_unknown'
    assert resolver.canonical('IS30D_IS30_unknown') == 'IS30D_IS30_unknown'
    assert resolver.canonical('IS30_IS30_unknown') == 'IS30D_IS30_unknown'
    assert resolver.canonical('ISEc53_IS3_unknown') == 'ISEc53_IS3_unknown'
    assert resolver.canonical('ISEc52_IS3_unknown') == 'ISEc53_IS3_unknown'


def test_lookalike_names_not_merged(resolver):
    # IS1A and IS1F share the IS1 family but are distinct elements in ISfinder
    assert resolver.canonical('IS1A_IS1_unknown') == 'IS1A_IS1_unknown'
    assert resolver.canonical('IS1F_IS1_unknown') == 'IS1F_IS1_unknown'
    # Names missing from the table are left alone
    assert resolver.canonical('IS26_IS6_unknown') == 'IS26_IS6_unknown'


def test_rename_uses_contig_position_order(resolver):
    # IS1B listed first in the batch, but IS1A sits earlier on the contig, so IS1A names the group
    renamed = resolver.rename(_batch(['IS1B_IS1_unknown', 'IS1A_IS1_unknown', 'IS1F_IS1_unknown'], [5000, 100, 8000]))
    assert list(renamed.q_names) == ['IS1A_IS1_unknown', 'IS1A_IS1_unknown', 'IS1F_IS1_unknown']
    # The choice sticks for later batches (e.g. the next contig)
    assert list(resolver.rename(_batch(['IS1B_IS1_unknown'], [0])).q_names) == ['IS1A_IS1_unknown']


def test_fetch_isfinder_table_downloads_once(tmp_path):
    source = tmp_path / 'remote.csv'
    source.write_text(ISFINDER_ROWS)
    cache = tmp_path / 'cache'

    path = fetch_isfinder_table(cache, url=source.as_uri())
    assert path == cache / 'IS.csv' and path.read_text() == ISFINDER_ROWS

    source.unlink()  # second call must come from the cache, not the URL
    assert fetch_isfinder_table(cache, url=source.as_uri()) == path
    assert SynonymResolver(path).canonical('IS1F_IS1_unknown') == 'IS1F_IS1_unknown'
