import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import prepare_external_rust as external
from evaluate_external_rust import summarize

PANEL = Path(__file__).resolve().parents[1] / 'experiments/external-rust-panel'


class PanelTests(unittest.TestCase):
    def test_refusals_remain_in_denominator_and_bad_evidence_fails(self):
        rows = [dict(source_sha256='a', target='1', status='accepted', prediction='1', p_compile='0.8', tokens='5'),
                dict(source_sha256='b', target='0', status='capacity', prediction='', p_compile='', tokens='')]
        expected = {'a': 1, 'b': 0}
        report = summarize(rows, expected)
        self.assertEqual((report['total'], report['accepted'], report['refused'], report['correct']), (2, 1, 1, 1))
        for bad in [rows[:1], rows + [rows[0]], [dict(rows[0], p_compile='nan'), rows[1]],
                    [rows[0], dict(rows[1], prediction='0')]]:
            with self.assertRaises(ValueError):
                summarize(bad, expected)
    def test_compiler_inputs_must_match_reviewed_inventory(self):
        self.assertEqual(len(external.checked_panel()['cases']), 8)
        with patch.object(external, 'PANEL_HASH', '0'*64):
            with self.assertRaises(ValueError):
                external.checked_panel()
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
