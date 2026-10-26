"""Batch MVP: --input examples/questions.json [--use-api]. No training or submission."""
import argparse
import hashlib
import json
import logging
import math
import os
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

from generate_answer import generate, render_candidates, generate_retrieved, PROFILE
from retrieve_openrouter import PacedClient
from retrieve_knowledge import validate_candidates

# ==================== CONFIGURATION ====================
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT/'runs/full_20261020_185211_497996/checkpoint'
SCHEMA = ROOT/'data/real_topics_v2/train.jsonl'
THREADS = 6  # CPU inference only; no GPU/training/downloads.
DEFAULT_ATTEMPTS = 3  # Total HTTP attempts, including retries, per run.
DEFAULT_COST = 0.01  # Stop after reported spend; NOT a hard billing cap.
MAX_ITEMS = 100
PUBLIC_PROFILE = ROOT/'examples/public/profile.json'
PUBLIC_QUESTIONS = ROOT/'examples/public/questions.json'
# =======================================================


class BudgetExceeded(RuntimeError):
    pass


class BudgetClient(PacedClient):
    def __init__(self, state, max_attempts, max_cost, opener=urllib.request.urlopen, save=lambda: None):
        self.state, self.max_attempts, self.max_cost = state, max_attempts, max_cost
        self.save = save
        def bounded(*args, **kwargs):
            if state['http_attempts'] >= max_attempts or state['reported_cost_usd'] >= max_cost:
                raise BudgetExceeded('API budget reached')
            state['http_attempts'] += 1  # Persist BEFORE dispatch, including uncertain failures.
            save()
            return opener(*args, **kwargs)
        super().__init__(opener=bounded)

    def request(self, request):
        data, transport = super().request(request)
        cost = (data.get('usage') or {}).get('cost')
        if isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0:
            self.state['reported_cost_usd'] += cost
        else:
            self.state['unknown_cost_responses'] += 1
        self.save()
        return data, transport


class LayaClassifier:
    def __init__(self):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        if not CHECKPOINT.is_dir():
            raise FileNotFoundError('Local trained checkpoint missing')
        import torch
        import laya
        torch.set_num_threads(THREADS)
        self.questions = json.loads(SCHEMA.read_text(encoding='utf-8').splitlines()[0])['questions']
        self.agent = laya.load(str(CHECKPOINT), device='cpu', fast=False, compile=False)

    def classify(self, question):
        result = self.agent.predict_batch([question], self.questions, batch_size=1)[0]
        answer = result['answers']['intent']
        return {'topic': answer['choice'], 'confidence': answer.get('confidence'),
                'calibrated': False, 'role': 'hint_only_not_evidence'}


def read_cases(path):
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ITEMS:
        raise ValueError('Input must contain 1..100 questions')
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen:
            raise ValueError('Every question needs a unique string id')
        seen.add(row['id'])
        if 'job_country' in row and not isinstance(row['job_country'], str):
            raise ValueError('Job country must be explicit text')
        context = row.get('job_context', {})
        if not isinstance(context, dict) or set(context) - {'salary_currency', 'salary_period', 'work_mode'} or any(not isinstance(v, str) or not v.strip() for v in context.values()):
            raise ValueError('Invalid job context; expected currency, period and work mode strings')
        if 'source_dataset' in row:
            source_path = (ROOT/row['source_dataset']).resolve()
            if not source_path.is_relative_to(ROOT/'data') or source_path.suffix != '.jsonl':
                raise ValueError('Invalid diagnostic source path')
            source_rows = [json.loads(line) for line in source_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            if not any(item.get('state') == row.get('question') and item.get('source') == row.get('source_capture') for item in source_rows):
                raise ValueError('Diagnostic question or capture reference does not match source dataset')
        if not isinstance(row.get('question'), str) or not 0 < len(row['question']) <= 4000:
            raise ValueError('Invalid question')
        if row.get('kind') not in {'text', 'number', 'choice', 'open', 'auto'}:
            raise ValueError('Invalid kind')
        options = row.get('options', [])
        if not isinstance(options, list) or any(not isinstance(o, str) or not o for o in options):
            raise ValueError('Invalid options')
        if row['kind'] == 'choice' and not options:
            raise ValueError('Choice fields require actual options')
        if 'expected_applicable_ids' in row and (not isinstance(row['expected_applicable_ids'], list) or any(not isinstance(i, str) for i in row['expected_applicable_ids'])):
            raise ValueError('Invalid applicability expectation')
        if 'expected_status' in row and row['expected_status'] not in {
                'answer_ready', 'needs_information', 'draft_needs_review', 'pending_api', 'pending_budget', 'error'}:
            raise ValueError('Invalid diagnostic expectation')
    return rows


def fingerprint(rows, profile, hybrid=False):
    digest = hashlib.sha256(json.dumps([rows, profile, hybrid], sort_keys=True).encode())
    for name in ('run_pipeline.py', 'generate_answer.py', 'retrieve_knowledge.py', 'retrieve_openrouter.py', 'retrieve_hybrid.py'):
        digest.update((ROOT/'scripts'/name).read_bytes())
    digest.update((CHECKPOINT/'rl_agent_config.json').read_bytes())
    digest.update(SCHEMA.read_bytes())
    return digest.hexdigest()


def save_state(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def attach_candidates(result, question, profile, use_api, client, search=None, job_country=None, kind='text', options=None):
    if result.get('status') != 'needs_information':
        return result
    if search is None:
        from retrieve_hybrid import search_candidates
        search = search_candidates
    retrieval = {}
    try:
        retrieval = search(question, profile, use_api, client)
        if 'candidates' in retrieval:
            retrieval['applicability'] = validate_candidates(question, retrieval['candidates'], profile, job_country)
            if kind == 'open' and result.get('api_calls', 0) == 0:
                rendered = generate_retrieved(question, profile, retrieval['candidates'], use_api, client)
            else:
                rendered = render_candidates(question, kind, options or [], profile, retrieval['candidates'], job_country)
            if rendered['status'] in {'answer_ready', 'draft_needs_review', 'pending_api'} or rendered.get('api_calls', 0):
                result = rendered
            else:
                retrieval['render_reason'] = rendered['reason']
    except BudgetExceeded:
        retrieval.update(status='pending_budget', ready_to_submit=False)
        if kind == 'open':
            result = {'status': 'pending_budget', 'answer': None, 'ready_to_submit': False}
    except Exception as exc:
        retrieval.update(status='unavailable', error_type=type(exc).__name__, ready_to_submit=False)
        if kind == 'open':
            result = {'status': 'error', 'answer': None, 'error_type': type(exc).__name__, 'ready_to_submit': False}
    return {**result, 'hybrid_retrieval': retrieval}


def self_test():
    import io
    state = {'http_attempts': 0, 'reported_cost_usd': 0, 'unknown_cost_responses': 0}
    client = BudgetClient(state, 1, 1, opener=lambda *a, **k: io.BytesIO(b'{"usage":{"cost":0.001}}'))
    client.request(None)
    client.completed = None
    try:
        client.request(None)
        raise AssertionError('Budget not enforced')
    except BudgetExceeded:
        pass
    assert state['http_attempts'] == 1 and state['reported_cost_usd'] == 0.001
    assert len(read_cases(ROOT/'examples/questions.json')) == 8
    missing = {'status': 'needs_information', 'answer': None, 'ready_to_submit': False}
    def fake_search(*args):
        assert args[-1] is client
        return {'candidates': [{'id': 'residence'}]}
    enriched = attach_candidates(missing, 'unknown', {}, False, client, fake_search)
    assert enriched['status'] == 'needs_information' and enriched['answer'] is None
    assert enriched['hybrid_retrieval']['candidates'][0]['id'] == 'residence'
    ready = {'status': 'answer_ready', 'answer': '4000'}
    assert attach_candidates(ready, '', {}, False, client, fake_search) is ready
    def exhausted(*args):
        raise BudgetExceeded()
    assert attach_candidates(missing, '', {}, True, client, exhausted)['hybrid_retrieval']['status'] == 'pending_budget'
    logging.info('pipeline_tests=PASSED network_calls=0')


def public_demo(output_root=None):
    """Exercise the real renderer with fictional facts; no classifier or API.

    This is deliberately a scoped, standard-library-only entry point, not a
    simulated replacement for the trained model or hybrid search.
    """
    profile = json.loads(PUBLIC_PROFILE.read_text(encoding='utf-8'))
    if profile.get('fixture_type') != 'fictional_public_demo':
        raise ValueError('Public demo requires an explicitly fictional profile')
    if any(f.get('ai_export') is not False for f in profile['facts']):
        raise ValueError('Public fixture must disable fact export')
    rows = read_cases(PUBLIC_QUESTIONS)
    results = []
    for row in rows:
        result = generate(row['question'], row['kind'], row.get('options', []),
                          profile, use_api=False)
        passed = (result['status'] == row['expected_status']
                  and result.get('answer') == row.get('expected_answer')
                  and result.get('api_calls', 0) == 0
                  and result.get('ready_to_submit') is False)
        results.append({'id': row['id'], 'question': row['question'],
                        'result': result, 'passed': passed})
        logging.info('PUBLIC DEMO | case=%s | status=%s | passed=%s',
                     row['id'], result['status'], passed)
    report = {'mode': 'fictional_rules_demo', 'laya_loaded': False,
              'hybrid_retrieval_enabled': False, 'api_calls': 0,
              'ready_to_submit': False, 'results': results}
    output = (Path(output_root) if output_root is not None else ROOT/'runs') / (
        'public_demo_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    save_state(output/'report.json', report)
    if not all(item['passed'] for item in results):
        raise RuntimeError('Public demo regression; inspect report')
    logging.info('PUBLIC DEMO PASSED | cases=%s | API=0 | Laya=off | RAG=off | report=%s',
                 len(results), output/'report.json')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'examples/questions.json')
    parser.add_argument('--resume', type=Path, help='Prior state.json; resumes only pending/errors')
    parser.add_argument('--use-api', action='store_true')
    parser.add_argument('--hybrid', action='store_true', help='Retrieve candidates and render supported validated facts')
    parser.add_argument('--max-attempts', type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument('--max-reported-cost', type=float, default=DEFAULT_COST)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--public-demo', action='store_true',
                        help='Fictional offline rule demo; no model, API, or private files')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    if args.public_demo:
        if args.use_api or args.hybrid or args.resume or args.self_test or args.input != ROOT/'examples/questions.json':
            parser.error('--public-demo cannot be combined with API, hybrid, resume, self-test or custom input')
        public_demo()
        return
    if args.self_test:
        self_test()
        return
    if not 0 <= args.max_attempts <= 100 or not math.isfinite(args.max_reported_cost) or args.max_reported_cost < 0:
        parser.error('Invalid API budget')
    rows = read_cases(args.input)
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    signature = fingerprint(rows, profile, args.hybrid)
    state = {'fingerprint': signature, 'state': 'RUNNING', 'http_attempts': 0,
             'reported_cost_usd': 0, 'unknown_cost_responses': 0, 'results': {},
             'checkpoint': str(CHECKPOINT), 'ready_to_submit': False}
    if args.resume:
        state = json.loads(args.resume.read_text(encoding='utf-8'))
        if state['fingerprint'] != signature:
            raise ValueError('Inputs/profile/code changed; start a new run instead of mixing results')
        state['state'] = 'RUNNING'
    # New output on resume preserves previous audit trail.
    output = ROOT/'runs'/('pipeline_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    path = output/'state.json'
    save = lambda: save_state(path, state)
    save()
    client = BudgetClient(state, args.max_attempts, args.max_reported_cost, save=save)
    logging.info('stage=load_laya checkpoint=%s report=%s', CHECKPOINT.name, path)
    try:
        classifier = LayaClassifier()
    except Exception as exc:
        state.update(state='FAILED_LOADING_LAYA', error_type=type(exc).__name__)
        save()
        raise
    for row in rows:
        old = state['results'].get(row['id'], {})
        if old.get('result', {}).get('status') in {'answer_ready', 'draft_needs_review', 'needs_information'}:
            continue
        started = time.monotonic()
        route = None
        try:
            route = classifier.classify(row['question'])
            kind = row['kind']
            if kind == 'auto':
                kind = 'open' if route['topic'] == 'F' else 'text'
            result = generate(row['question'], kind, row.get('options', []), profile, args.use_api, client, row.get('job_country'), row.get('job_context'))
            if args.hybrid:
                result = attach_candidates(result, row['question'], profile, args.use_api, client, job_country=row.get('job_country'), kind=kind, options=row.get('options', []))
            if kind == 'open' and not args.use_api:
                result = {**result, 'status': 'pending_api'} if result.get('reason') == 'Narrative requires explicit --use-api' else result
        except BudgetExceeded:
            result = {'status': 'pending_budget', 'ready_to_submit': False}
        except Exception as exc:
            result = {'status': 'error', 'error_type': type(exc).__name__, 'ready_to_submit': False}
        state['results'][row['id']] = {'question': row['question'], 'laya': route,
                                      'seconds': time.monotonic()-started, 'result': result}
        if 'source_dataset' in row:
            state['results'][row['id']]['diagnostic_source'] = {k: row[k] for k in
                ('source_dataset', 'source_capture', 'evaluation_note', 'context_note') if k in row}
        save()
        logging.info('item=%s topic=%s status=%s', row['id'], route and route['topic'], result['status'])
    state['counts'] = dict(Counter(r['result']['status'] for r in state['results'].values()))
    checks = [{'id': row['id'], 'expected': row['expected_status'],
               'actual': state['results'][row['id']]['result']['status'],
               'passed': row['expected_status'] == state['results'][row['id']]['result']['status']}
              for row in rows if 'expected_status' in row]
    if checks:
        state['diagnostic_checks'] = checks
        state['diagnostic_note'] = 'Checks output status only, not answer correctness or classifier accuracy; authored regression cases.'
    applicability_checks = []
    for row in rows:
        if 'expected_applicable_ids' not in row:
            continue
        actual = state['results'][row['id']]['result'].get('hybrid_retrieval', {}).get('applicability', {}).get('applicable_ids')
        applicability_checks.append({'id': row['id'], 'expected': row['expected_applicable_ids'], 'actual': actual,
                                     'passed': actual is not None and sorted(actual) == sorted(row['expected_applicable_ids'])})
    if applicability_checks:
        state['applicability_checks'] = applicability_checks
    answer_checks = [{'id': row['id'], 'expected': row['expected_answer'],
                      'actual': state['results'][row['id']]['result'].get('answer'),
                      'passed': state['results'][row['id']]['result'].get('answer') == row['expected_answer']}
                     for row in rows if 'expected_answer' in row]
    if answer_checks:
        state['answer_checks'] = answer_checks
    state['state'] = 'FINISHED_WITH_PENDING' if any(k in state['counts'] for k in ('error', 'pending_api', 'pending_budget', 'needs_information')) else 'FINISHED'
    save()
    logging.info('completed counts=%s attempts=%d cost=%.8f report=%s', state['counts'], state['http_attempts'], state['reported_cost_usd'], path)
    if checks and not all(check['passed'] for check in checks):
        raise RuntimeError('Diagnostic expectations failed; inspect saved report')
    if applicability_checks and not all(check['passed'] for check in applicability_checks):
        raise RuntimeError('Applicability expectations failed; inspect saved report')
    if answer_checks and not all(check['passed'] for check in answer_checks):
        raise RuntimeError('Answer expectations failed; inspect saved report')


if __name__ == '__main__':
    main()
