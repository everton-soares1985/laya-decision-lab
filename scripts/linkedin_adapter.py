"""Preview-only boundary for LinkedIn FormField/JobRecord dictionaries.

Run: python -X utf8 scripts/linkedin_adapter.py --self-test
No browser, submit, training or implicit API calls. Not wired into LinkedIn yet.
"""
import argparse
import json
import logging
import sys
from contextlib import redirect_stdout
from collections.abc import Mapping
from pathlib import Path

from generate_answer import PROFILE, SELF_TEST_PROFILE, generate, withheld
from retrieve_knowledge import normalize
from run_pipeline import attach_candidates, BudgetClient, BudgetExceeded
from run_pipeline import LayaClassifier

# ==================== CONFIGURATION ====================
# Keep browser-owned fields in the existing resolver. No consent inference.
LEGACY_KINDS = {'upload', 'review', 'sensitive', 'work_authorization', 'sponsorship',
                'certification', 'education', 'phone', 'email', 'date', 'checkbox'}
COUNTRIES = {'brazil': 'Brazil', 'brasil': 'Brazil', 'br': 'Brazil',
             'united states': 'United States', 'usa': 'United States',
             'portugal': 'Portugal', 'mexico': 'Mexico', 'canada': 'Canada',
             'argentina': 'Argentina', 'chile': 'Chile', 'peru': 'Peru'}
WORK_MODES = {'remote': 'remote', 'remoto': 'remote', 'remota': 'remote',
              'hybrid': 'hybrid', 'hibrido': 'hybrid', 'hibrida': 'hybrid',
              'on-site': 'onsite', 'onsite': 'onsite', 'presencial': 'onsite'}
MAX_PREVIEW_FIELDS = 30
# =======================================================


def job_metadata(job):
    """Explicit country or recognized terminal location component, never language."""
    country = job.get('job_country') or job.get('country')
    source = 'explicit_country' if country else None
    if not country:
        terminal = str(job.get('location', '')).split(',')[-1].strip()
        country = COUNTRIES.get(normalize(terminal))
        source = 'location_country_suffix' if country else None
    context = dict(job.get('job_context') or {})
    if not set(context) <= {'work_mode', 'salary_currency', 'salary_period'}:
        raise ValueError('Unsupported job context keys')
    if any(not isinstance(value, str) or not value.strip() for value in context.values()):
        raise ValueError('Job context values must be nonempty strings')
    if country is not None and not isinstance(country, str):
        raise ValueError('Invalid job country')
    if 'work_mode' not in context:
        mode = WORK_MODES.get(normalize(str(job.get('workplace', '')).strip()))
        if mode:
            context['work_mode'] = mode
    for key in ('salary_currency', 'salary_period'):
        if key not in context and job.get(key):
            if not isinstance(job[key], str):
                raise ValueError('Invalid salary metadata')
            context[key] = job[key]
    return country, context, source


def preview_field(field, job, profile, *, classifier=None, hybrid=False,
                  use_api=False, client=None):
    """Accept existing .to_dict() output; never authorize filling or submission."""
    if not isinstance(field, Mapping) or not isinstance(job, Mapping):
        raise ValueError('Expected field/job dictionaries')
    question, key = field.get('label'), field.get('field_key')
    if not isinstance(question, str) or not question.strip() or not isinstance(key, str) or not key:
        raise ValueError('Field requires label and field_key')
    options = field.get('options', [])
    if not isinstance(options, (list, tuple)) or any(not isinstance(v, str) or not v.strip() for v in options):
        raise ValueError('Invalid field options')
    if use_api and not isinstance(client, BudgetClient):
        raise ValueError('API preview requires a shared bounded client')
    country, context, country_source = job_metadata(job)
    input_type, semantic = field.get('input_type', ''), field.get('kind', '')
    route = None
    if semantic in LEGACY_KINDS or input_type in {'file', 'checkbox', 'email', 'tel', 'date'} or field.get('contact_role'):
        answer = withheld('Keep this field in existing confirmed/specialized resolver')
    elif input_type in {'radio', 'select', 'select-one'} and not options:
        answer = withheld('Choice control requires extracted options')
    else:
        kind = ('choice' if options else 'open' if input_type == 'textarea' or semantic == 'narrative'
                else 'number' if input_type == 'number' or semantic in {'integer', 'decimal', 'number', 'experience_years'} else 'text')
        if classifier is not None:
            route = classifier.classify(question)
        answer = generate(question, kind, list(options), profile, use_api, client, country, context)
        if hybrid:
            answer = attach_candidates(answer, question, profile, use_api, client,
                                       job_country=country, kind=kind, options=list(options))
    return {'field_key': key, 'selector_hint': field.get('selector_hint', ''),
            'preview_only': True, 'safe_to_fill': False, 'ready_to_submit': False,
            'job_country': country, 'country_source': country_source,
            'job_context': context, 'laya': route, 'result': answer}


def self_test():
    profile = json.loads(SELF_TEST_PROFILE.read_text(encoding='utf-8'))
    field = {'field_key': 'salary', 'label': 'Pretensão salarial CLT?',
             'kind': 'salary', 'input_type': 'text', 'selector_hint': '#salary'}
    job = {'location': 'Cidade Exemplo, Brasil', 'workplace': 'Remoto', 'salary_period': 'month'}
    result = preview_field(field, job, profile)
    assert result['result']['answer'] == '5100'
    assert result['country_source'] == 'location_country_suffix'
    assert result['selector_hint'] == '#salary' and not result['safe_to_fill']
    for location in ['Remote', 'LATAM', 'Example City', 'Georgia']:
        assert job_metadata({'location': location})[0] is None
    city = {**field, 'label': 'What city do you reside in?', 'kind': 'location'}
    assert preview_field(city, job, profile)['result']['answer'] == 'Cidade Exemplo'
    assert preview_field(city, {'country': 'Portugal'}, profile)['result']['answer'] == 'Example City'
    assert preview_field(city, {'location': 'Remote'}, profile)['result']['answer'] is None
    radio = {**field, 'label': 'Do you have experience with Terraform?', 'kind': 'radio',
             'input_type': 'radio', 'options': ('Yes', 'No')}
    assert preview_field(radio, job, profile)['result']['answer'] == 'No'
    assert preview_field({**radio, 'options': []}, job, profile)['result']['answer'] is None
    for special in LEGACY_KINDS:
        assert preview_field({**radio, 'kind': special}, job, profile)['result']['answer'] is None
    try:
        preview_field(field, job, profile, use_api=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Unbounded API call accepted')
    logging.info('adapter_tests=PASSED network_calls=0 browser_actions=0')


def run_request(request_path, output_path, *, use_api=False, max_attempts=3):
    """Single captured page, single model load, persisted shared API budget."""
    request = json.loads(request_path.read_text(encoding='utf-8'))
    fields = request.get('fields')
    if not isinstance(fields, list) or not 1 <= len(fields) <= MAX_PREVIEW_FIELDS:
        raise ValueError('Preview requires 1..30 grouped fields')
    if not 0 <= max_attempts <= 10:
        raise ValueError('Preview API attempts must be 0..10')
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    state = {'preview_only': True, 'safe_to_fill': False, 'ready_to_submit': False,
             'http_attempts': 0, 'reported_cost_usd': 0, 'unknown_cost_responses': 0,
             'results': [], 'state': 'RUNNING'}
    def save():
        temporary = output_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(output_path)
    save()
    client = BudgetClient(state, max_attempts, 0.01, save=save)
    logging.info('preview_load_laya | fields=%d | api=%s', len(fields), use_api)
    classifier = LayaClassifier()
    for field in fields:
        try:
            result = preview_field(field, request.get('job', {}), profile,
                                   classifier=classifier, hybrid=True, use_api=use_api, client=client)
        except BudgetExceeded:
            result = {'field_key': field.get('field_key'), 'preview_only': True,
                      'safe_to_fill': False, 'ready_to_submit': False,
                      'result': {'status': 'pending_budget', 'answer': None}}
        except Exception as exc:
            result = {'field_key': field.get('field_key'), 'preview_only': True,
                      'safe_to_fill': False, 'ready_to_submit': False,
                      'result': {'status': 'error', 'answer': None, 'error_type': type(exc).__name__}}
        state['results'].append(result)
        save()
        logging.info('laya_preview | question=%s | status=%s | answer=%s',
                     json.dumps(field.get('label'), ensure_ascii=False),
                     result['result']['status'], json.dumps(result['result'].get('answer'), ensure_ascii=False))
    state['state'] = 'COMPLETED'
    save()
    logging.info('preview_completed | requests=%d | cost_usd=%.8f', state['http_attempts'], state['reported_cost_usd'])
    return 1 if any(r['result']['status'] == 'error' for r in state['results']) else 0


def serve():
    """Persistent JSON-lines worker. stdout is protocol only, diagnostics use stderr."""
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        profile = json.loads(PROFILE.read_text(encoding='utf-8'))
        classifier = LayaClassifier()
    protocol.write(json.dumps({'ready': True}) + '\n')
    protocol.flush()
    last_completed = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            budget = request.get('remaining_attempts', 0)
            if type(budget) is not int or not 0 <= budget <= 10:
                raise ValueError('Invalid attempt budget')
            state = {'http_attempts': 0, 'reported_cost_usd': 0, 'unknown_cost_responses': 0}
            client = BudgetClient(state, budget, 0.01)
            client.completed = last_completed
            with redirect_stdout(sys.stderr):
                try:
                    result = preview_field(request['field'], request['job'], profile,
                                           classifier=classifier, hybrid=True,
                                           use_api=request.get('use_api') is True, client=client)
                except BudgetExceeded:
                    result = {'result': {'status': 'pending_budget', 'answer': None}}
                except Exception as exc:
                    result = {'result': {'status': 'error', 'answer': None, 'error_type': type(exc).__name__}}
            last_completed = client.completed
            response = {**result, 'budget': state}
        except Exception as exc:
            response = {'result': {'status': 'error', 'answer': None, 'error_type': type(exc).__name__}}
        protocol.write(json.dumps(response, ensure_ascii=False) + '\n')
        protocol.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test', action='store_true')
    group.add_argument('--request', type=Path)
    group.add_argument('--serve', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--use-api', action='store_true')
    parser.add_argument('--max-attempts', type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    if args.self_test:
        self_test()
    elif args.serve:
        serve()
    else:
        if args.output is None:
            parser.error('--request requires --output')
        raise SystemExit(run_request(args.request, args.output, use_api=args.use_api, max_attempts=args.max_attempts))
