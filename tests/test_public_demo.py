"""Standard-library-only public checks; no private data, model or network."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# ==================== CONFIGURATION ====================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
# =======================================================
import run_pipeline
import generate_answer
import linkedin_adapter
import retrieve_knowledge
from generate_answer import generate


class PublicDemoTests(unittest.TestCase):
    def test_legacy_self_tests_use_only_fictional_profile(self):
        original_read = Path.read_text

        def public_read(path, *args, **kwargs):
            if path.name == 'profile_v1.json' or 'data' in path.parts:
                raise AssertionError('Private profile access forbidden')
            return original_read(path, *args, **kwargs)

        with patch.object(Path, 'read_text', public_read), \
                patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')), \
                patch.object(generate_answer, 'api_key', side_effect=AssertionError('Credentials forbidden')), \
                patch.object(linkedin_adapter, 'LayaClassifier', side_effect=AssertionError('Model forbidden')):
            with patch.object(sys, 'argv', ['generate_answer.py', '--self-test']):
                generate_answer.main()
            with patch.object(sys, 'argv', ['retrieve_knowledge.py', '--self-test']):
                retrieve_knowledge.main()
            linkedin_adapter.self_test()

    def test_demo_never_loads_model_or_calls_network(self):
        with tempfile.TemporaryDirectory() as output, \
                patch.object(run_pipeline, 'LayaClassifier', side_effect=AssertionError('Model forbidden')), \
                patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')), \
                patch('retrieve_openrouter.api_key', side_effect=AssertionError('Credentials forbidden')):
            report = run_pipeline.public_demo(output)
            self.assertEqual(len(report['results']), 9)
            self.assertEqual(sum(r['result']['status'] == 'answer_ready' for r in report['results']), 4)
            self.assertTrue(all(r['passed'] for r in report['results']))
            self.assertFalse(report['laya_loaded'])
            saved = list(Path(output).glob('*/report.json'))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text(encoding='utf-8')), report)

    def test_unconfirmed_fact_is_not_used(self):
        profile = json.loads(run_pipeline.PUBLIC_PROFILE.read_text(encoding='utf-8'))
        for fact in profile['facts']:
            fact['status'] = 'needs_clarification'
        result = generate('What is your English proficiency level?', 'text', [], profile)
        self.assertEqual(result['status'], 'needs_information')

    def test_duplicate_equivalent_options_are_not_guessed(self):
        profile = json.loads(run_pipeline.PUBLIC_PROFILE.read_text(encoding='utf-8'))
        result = generate('Can you start immediately?', 'choice', ['No', 'Não'], profile)
        self.assertEqual(result['status'], 'needs_information')

    def test_public_mode_rejects_paid_or_private_modes(self):
        for flags in [['--use-api'], ['--hybrid'], ['--resume', 'private.json'],
                      ['--input', 'private.json'], ['--self-test']]:
            process = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/run_pipeline.py'),
                                      '--public-demo', *flags], capture_output=True, text=True)
            self.assertEqual(process.returncode, 2, flags)
            self.assertIn('cannot be combined', process.stderr)


if __name__ == '__main__':
    unittest.main()
