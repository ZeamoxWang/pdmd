import json
import tempfile
import unittest
from pathlib import Path
from pdmd_training.run_log import recover_metrics


class RecoveryLogTests(unittest.TestCase):
    def test_interrupted_tail_is_preserved_and_recovery_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'metrics.jsonl'
            prefix = ''.join(json.dumps({'iteration': i})+'\n' for i in (1, 2))
            original = prefix + '{"iteration":3}\n{"iteration":'
            p.write_text(original)
            recover_metrics(p, 2)
            self.assertEqual(p.read_text(), prefix)
            backups = list(Path(d).glob('*.before-resume-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), original)
            recover_metrics(p, 2)
            self.assertEqual(len(list(Path(d).glob('*.before-resume-*'))), 1)

    def test_missing_or_inconsistent_saved_steps_are_not_silently_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'metrics.jsonl'
            for original in ('{"iteration":1}\n', '{"iteration":1}\n{"iteration":3}\n'):
                p.write_text(original)
                with self.assertRaises(ValueError):
                    recover_metrics(p, 2)
                self.assertEqual(p.read_text(), original)


if __name__ == '__main__':
    unittest.main()
