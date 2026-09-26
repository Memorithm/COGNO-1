import hashlib
import json
from pathlib import Path
import unittest

PANEL = Path(__file__).resolve().parents[1] / 'experiments/external-rust-panel'


class PanelTests(unittest.TestCase):
    def test_pinned_sources_and_reviewed_transformation(self):
        panel = json.loads((PANEL / 'panel.json').read_text())
        self.assertEqual(panel['partition'], 'test')
        self.assertEqual(len(panel['cases']), 8)
        self.assertEqual(sum(c['expected_label'] for c in panel['cases']), 3)
        for c in panel['cases']:
            raw = (PANEL / 'upstream' / (c['id'] + '.rs')).read_bytes()
            source = (PANEL / 'source' / (c['id'] + '.rs')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), c['raw_sha256'])
            self.assertEqual(hashlib.sha256(source).hexdigest(), c['source_sha256'])
            lines = [line.split('//')[0].rstrip() for line in raw.decode().split('\n')]
            cleaned = '\n'.join(line for line in lines if line.strip()) + '\n'
            self.assertEqual(source, cleaned.encode())
            self.assertNotIn(b'ERROR', source)
            self.assertNotIn(b'run-pass', source)


if __name__ == '__main__':
    unittest.main()
