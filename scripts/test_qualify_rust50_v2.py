import json
import unittest
from unittest.mock import patch
import qualify_rust50_v2 as q


class QualificationTests(unittest.TestCase):
    def setUp(self):
        source = b'fn main() {}\n'
        self.record = dict(source=source.decode(), sha256=q.sha(source), project='upstream',
                           label=1, split='train', license='MIT')
        self.corpus = (f'CRUST001\nvalidation\tupstream\t1\t{q.sha(source)}\t{source.hex()}\n').encode()

    def original(self, rows):
        return ''.join(json.dumps(row) + '\n' for row in rows).encode()

    def test_repartition_binds_actual_source_and_preserves_metadata(self):
        with patch.object(q, 'CORPUS_SHA', q.sha(self.corpus)):
            result = json.loads(q.regroup_provenance(self.corpus, self.original([self.record])))
        self.assertEqual(result, dict(self.record, split='validation'))

    def test_missing_duplicate_or_changed_provenance_refused(self):
        with patch.object(q, 'CORPUS_SHA', q.sha(self.corpus)):
            for records in [[], [self.record, self.record], [dict(self.record, label=0)],
                            [dict(self.record, source='changed')], [dict(self.record, project='alias')]]:
                with self.subTest(records=records), self.assertRaises(ValueError):
                    q.regroup_provenance(self.corpus, self.original(records))

    def test_fixed_corpus_identity_required(self):
        with self.assertRaises(ValueError):
            q.regroup_provenance(self.corpus, self.original([self.record]))

    def test_protocol_pins_budget_and_provenance(self):
        fields = dict(line.split('\t') for line in q.trainer_protocol(b'provenance').decode().splitlines())
        self.assertEqual(fields['provenance_sha256'], q.sha(b'provenance'))
        self.assertEqual((fields['seeds'], fields['arms'], fields['epochs'], fields['batch']),
                         ('1,7,42', 'full,cycle_mix', '24', '1'))
        self.assertEqual(q.sha(q.PROTOCOL.read_bytes()),
                         'e9f991f4332706d2b69f60762991f3a019b6488bcf34ccb62bc49f24d915f94d')
