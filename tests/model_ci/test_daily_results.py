"""CPU-only checks for missing scores observed in run 37836477318."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2] / '.github/scripts/model_ci/validate_daily_results.py'
spec = importlib.util.spec_from_file_location('daily_results', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DailyResultsTests(unittest.TestCase):
    def setUp(self):
        self.cases = {'model': {'type': 'precision', 'evalscope': {'datasets': ['gsm8k', 'mcp_atlas']}}}

    def test_complete_and_legal_zero(self):
        result = module.validate_results(self.cases, {'test_result': [{'model_name': 'model', 'gsm8k': 0, 'mcp-atlas': 80}]})
        self.assertTrue(result['complete'])
        self.assertEqual(result['valid_result_count'], 2)

    def test_skipped_mcp_is_failure(self):
        result = module.validate_results(self.cases, {'test_result': [{'model_name': 'model', 'gsm8k': 90}]})
        self.assertFalse(result['complete'])
        self.assertIn('model/mcp_atlas', result['errors'][0])

    def test_empty_model_and_missing_model_are_failures(self):
        for entries in ([], [{'model_name': 'model'}]):
            self.assertFalse(module.validate_results(self.cases, {'test_result': entries})['complete'])

    def test_invalid_scores_are_failures(self):
        for score in (None, True, '90', float('nan'), float('inf'), -1, 101):
            result = module.validate_results(self.cases, {'test_result': [{'model_name': 'model', 'gsm8k': score, 'mcp-atlas': 80}]})
            self.assertFalse(result['complete'])

    def test_duplicate_and_unexpected_models_are_failures(self):
        entry = {'model_name': 'model', 'gsm8k': 90, 'mcp-atlas': 80}
        for entries in ([entry, entry], [entry, {'model_name': 'other'}]):
            self.assertFalse(module.validate_results(self.cases, {'test_result': entries})['complete'])

    def test_malformed_summary_is_failure(self):
        for notice in ({}, [], {'test_result': {}}):
            with self.assertRaises(ValueError):
                module.validate_results(self.cases, notice)

    def test_failed_command_cannot_pass_with_complete_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'cases.yaml'
            config.write_text('model:\n  type: precision\n  evalscope:\n    datasets: [gsm8k, mcp_atlas]\n')
            (root / 'notificate.json').write_text(json.dumps({'test_result': [
                {'model_name': 'model', 'gsm8k': 90, 'mcp-atlas': 80}]}))
            argv = ['gate', '--case-conf', str(config), '--output-dir', str(root), '--evaluation-outcome', 'failure']
            with patch('sys.argv', argv):
                self.assertEqual(module.main(), 1)
            result = json.loads((root / 'completeness.json').read_text())
            self.assertFalse(result['complete'])
            self.assertIn('CItest step outcome: failure', result['errors'])


if __name__ == '__main__':
    unittest.main()
