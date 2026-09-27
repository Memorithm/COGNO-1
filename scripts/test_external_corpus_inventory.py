import copy
import json
import unittest
from external_corpus_inventory import validate, digest


def row():
    return dict(id='sample', repository='owner/repo', revision='a'*40, path='tests/a.rs',
                sha256='b'*64, bytes=20, license='MIT', license_path='LICENSE',
                license_sha256='c'*64, license_bytes=1000, expected_label=1, expected_diagnostics=[])


def check(rows):
    data=json.dumps(dict(schema=1,sources=rows)).encode()
    return validate(data,digest(data))


class InventoryTests(unittest.TestCase):
    def test_valid_and_identity(self):
        self.assertEqual(check([row()])['sources'][0],row())
        with self.assertRaises(ValueError):check([row(),row()])
    def test_immutable_paths_labels_and_types(self):
        for key,value in [('revision','main'),('path','../secret'),('repository','Owner/Repo'),('bytes',True),('expected_label',0)]:
            r=row();r[key]=value
            with self.assertRaises(ValueError):check([r])
    def test_digest_and_duplicate_keys(self):
        with self.assertRaises(ValueError):validate(b'{}','0'*64)
        data=b'{"schema":1,"schema":1,"sources":[]}'
        with self.assertRaises(ValueError):validate(data,digest(data))
