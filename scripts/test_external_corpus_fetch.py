import tempfile
from pathlib import Path
import unittest
from external_corpus_fetch import obtain,checked
from external_corpus_inventory import digest
from test_external_corpus_inventory import row

class FetchTests(unittest.TestCase):
    def test_immutable_url_cache_and_offline(self):
        r=row();r.update(bytes=3,sha256=digest(b'abc'));seen=[]
        def fetch(url,limit):seen.append((url,limit));return b'abc'
        with tempfile.TemporaryDirectory() as t:
            c=Path(t);self.assertEqual(obtain(r,c,fetch=fetch),b'abc')
            self.assertEqual(obtain(r,c,offline=True),b'abc')
            self.assertEqual(seen,[(f"https://raw.githubusercontent.com/owner/repo/{'a'*40}/tests/a.rs",3)])
    def test_corrupt_truncated_missing_and_symlink_refused(self):
        r=row()
        for data in [b'',b'x'*21]:
            with self.assertRaises(ValueError):checked(data,r['sha256'],r['bytes'])
        with tempfile.TemporaryDirectory() as t:
            c=Path(t)
            with self.assertRaises(ValueError):obtain(r,c,offline=True)
            (c/r['sha256']).symlink_to(c/'missing')
            with self.assertRaises(ValueError):obtain(r,c)
