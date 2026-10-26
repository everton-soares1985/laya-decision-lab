"""Conservative local evidence retrieval: --question TEXT or --self-test. No LLM/API."""
import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path

# ==================== CONFIGURATION ====================
ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "data" / "knowledge" / "profile_v1.json"
SELF_TEST_PROFILE = ROOT / "examples/public/regression_profile.json"  # Fictional offline checks only.
RULES = {  # Initial routing rules, not embeddings or semantic search.
    "english": r"english|ingles",
    "start": r"start immediately|inicio imediato|inicio inmediato|disponibilidad inmediata",
    "python_years": r"python.*(years|anos)|(?:years|anos).*python",
    "compensation": r"salary|compensation|remuner|salario|pretens|sueldo|\bpay\b|\bearn\b|\bcobrar\b|\bganhar\b|\bganar\b",
    "location": r"where.*(live|based)|current location|resid|mora|vive|vives|cidade|city|ciudad|municip|\btown\b|call home|localidade|ubicacion|donde.*(viviendo|encuentras)",
    "professional_profile": r"current company|current employer|empresa atual|empresa actual|empresa.*(trabalha|trabajas)|currently work for",
    "financial": r"mt5|backtest|mercado financeiro|financial market",
    "finetuning": r"fine.?tun|laya|machine learning|aprendizado de maquina|ajuste fino",
}
# =======================================================


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def retrieve(question, profile):
    text = normalize(question)
    matched = [name for name, pattern in RULES.items() if re.search(pattern, text)]
    result = {"status": "missing_information", "evidence": [], "reasons": [],
              "method": "deterministic_rules_v2_multilingual", "ready_to_submit": False}
    if len(matched) != 1:
        result["reasons"] = ["Unsupported or multi-topic question; needs review"]
        return result
    topic = matched[0]
    target = {"location": "residence", "professional_profile": "current_company_answer",
              "financial": "financial_strategy_experience", "finetuning": "laya_finetuning_project",
              "compensation": "salary_usd_month", "english": "english_level",
              "start": "immediate_start", "python_years": "python_practical_years"}[topic]
    if topic == "english" and (not re.search(r"level|nivel|proficiency", text) or re.search(r"certif|cefr|ielts|toefl|fluen|\bc[12]\b|confiden|confianc|confianz", text)):
        result['reasons'] = ['Self-declared English level does not establish certification or fluency']
        return result
    if topic == "python_years" and (not re.search(r"pratic|practic|hands.on", text) or re.search(r"professional|profissional|profesional|production|produc|work experience", text)):
        result['reasons'] = ['Practical Python duration is not professional or production duration']
        return result
    if topic == "start" and re.search(r"onsite|presencial|hybrid|hibrid|relocat|autoriz|hours|horas", text):
        result['reasons'] = ['Immediate start does not establish other availability conditions']
        return result
    if topic == "compensation":
        if re.search(r"brl|r\$|reais|reales|pesos|mxn|eur|cad|aud|canadian|australian|clt|\bpj\b|anual|annual|year|hour|hora|semana|week|gross|brut|\bnet\b|neto|liquid|contractor|b2b", text):
            result["reasons"] = ["Requested currency, period, regime or gross/net basis is not confirmed"]
            return result
        if not re.search(r"\busd\b|us\$|u\.?s\.? dollars|dolares (americanos|estadounidenses)", text) or not re.search(r"month|mensal|mensual|por mes|ao mes", text):
            result["reasons"] = ["Currency and monthly period must be explicit"]
            return result
    if topic == "location" and re.search(r"relocat|mudar|mudanca|mudarte|desloc|commut|presencial|hybrid|hibrid|work permit|autoriz|prefer|wish|want|gostaria|would.*live", text):
        result["reasons"] = ["Residence does not establish mobility or work authorization"]
        return result
    candidates = [f for f in profile["facts"] if f["id"] == target and f["status"] == "confirmed"]
    if not candidates:
        result["reasons"] = ["No confirmed fact for this topic"]
        return result
    result["evidence"] = [{**{k: fact[k] for k in ("id", "value", "source", "constraints", "status")},
                           "ai_export": fact.get('ai_export', False)} for fact in candidates]
    result["status"] = "evidence_found"
    result["reasons"] = ["Evidence only; review constraints and freshness before generating an answer"]
    if topic == "finetuning" and re.search(r"years|anos|professional|profissional|profesional|production|producao|produccion", text):
        result["status"] = "needs_review"
        result["reasons"] = ["Personal project evidence does not establish years or production employment"]
    return result


def validate_candidates(question, candidates, profile, job_country=None):
    """Conservative applicability audit, not answer/submission authorization.

    Scores are deliberately ignored. Unsupported wording remains under review.
    """
    text = normalize(question)
    facts = {f['id']: f for f in profile.get('facts', [])}
    baseline = retrieve(question, profile)
    supported = {f['id'] for f in baseline['evidence']} if baseline['status'] == 'evidence_found' else set()
    decisions = []
    for candidate in candidates:
        fid = candidate.get('id')
        fact = facts.get(fid)
        status, reason = 'needs_review', 'No applicability rule for this wording and record'
        if not fact or fact.get('status') != 'confirmed':
            status, reason = 'rejected', 'Unknown or unconfirmed record'
        elif fid.startswith('salary_'):
            value = fact['value']
            currency = set()
            for code, pattern in [('USD', r'\busd\b|us\$|us dollars|dolares estadounidenses'), ('BRL', r'\bbrl\b|r\$|reais'), ('MXN', r'\bmxn\b|pesos'), ('EUR', r'\beur\b|euros'), ('CAD', r'\bcad\b|canadian'), ('AUD', r'\baud\b|australian')]:
                if re.search(pattern, text):
                    currency.add(code)
            periods = {p for p, pattern in [('month', r'month|mensal|mensual|por mes|ao mes'), ('hour', r'hour|hora'), ('year', r'annual|anual|year'), ('week', r'week|semana')] if re.search(pattern, text)}
            modes = {m for m, pattern in [('remote', r'remote|remoto'), ('hybrid', r'hybrid|hibrid')] if re.search(pattern, text)}
            if any(re.search(pattern, text) for topic, pattern in RULES.items() if topic != 'compensation'):
                reason = 'Mixed-topic question requires separate answers'
            elif re.search(r'bonus|equity|stock|benefit|beneficio|package|increase|aumento|percent|porcent|tax|impost|convert', text):
                reason = 'Salary reference does not establish package components or calculations'
            elif not re.search(RULES['compensation'], text):
                reason = 'Not an explicit compensation question'
            elif re.search(r'gross|brut|\bnet\b|liquid|neto|clt|\bpj\b|b2b|contractor|current|atual|actual|minimum|minimo|offer|oferta|accept|aceit|acept', text):
                reason = 'Reference amount does not establish earnings, regime, basis, minimum or offer acceptance'
            elif currency and currency != {value['currency']} or periods and periods != {value['period']}:
                status, reason = 'rejected', 'Currency or period mismatch'
            elif currency != {value['currency']} or periods != {value['period']}:
                reason = 'Explicit currency and period required'
            elif value.get('work_mode') and modes != {value['work_mode']}:
                reason = 'Explicit matching work mode required'
            else:
                status, reason = 'applicable', 'Matching salary reference currency, period and work mode'
        elif fid == 'spanish':
            if re.search(r'spanish|espanhol|espanol', text) and re.search(r'level|nivel|proficiency', text) and not re.search(r'certif|cefr|fluen|\bc[12]\b|confiden|confianc|confianz|english|ingles|years|anos|salary|salario|python|portugu|french|franc|german|alema', text):
                status, reason = 'applicable', 'Self-declared Spanish level only'
        elif fid == 'residence':
            if fid in supported:
                country_only = re.search(r'country|pais', text) and not re.search(r'city|cidade|ciudad', text)
                country = normalize(job_country.strip()) if isinstance(job_country, str) else ''
                if country_only or country and country not in {'unknown', 'remote', 'worldwide', 'latam', 'international', 'global', 'desconhecido'}:
                    status, reason = 'applicable', 'Location display rule has required job-country context'
                else:
                    reason = 'Job country required; question language is not country context'
        elif fid in supported:
            status, reason = 'applicable', 'Existing confirmed-evidence rule matches'
        decisions.append({'id': fid, 'status': status, 'reason': reason})
    # Multiple independent records are not silently combined into a single answer.
    applicable = [d['id'] for d in decisions if d['status'] == 'applicable']
    return {'status': 'single_applicable_candidate' if len(applicable) == 1 else 'needs_review',
            'applicable_ids': applicable, 'decisions': decisions,
            'method': 'local_applicability_v1', 'ready_to_submit': False}


def applicability_test(profile):
    cases = [
        ('What hourly salary do you expect in USD?', 'salary_usd_hour', None, 'applicable'),
        ('What monthly salary do you expect in USD?', 'salary_usd_hour', None, 'rejected'),
        ('Pretensão salarial mensal em reais para trabalho remoto?', 'salary_brl_month_remote', None, 'applicable'),
        ('Pretensão salarial mensal em reais para trabalho híbrido?', 'salary_brl_month_hybrid', None, 'applicable'),
        ('Pretensão salarial mensal em reais?', 'salary_brl_month_remote', None, 'needs_review'),
        ('Monthly salary in MXN?', 'salary_usd_month', None, 'rejected'),
        ('Monthly gross salary in USD?', 'salary_usd_month', None, 'needs_review'),
        ('Current monthly salary in USD?', 'salary_usd_month', None, 'needs_review'),
        ('Accept an offer below your monthly salary in USD?', 'salary_usd_month', None, 'needs_review'),
        ('Qual seu nível de espanhol?', 'spanish', None, 'applicable'),
        ('Qual seu nível certificado de espanhol?', 'spanish', None, 'needs_review'),
        ('What is your level of confidence in a Spanish-speaking environment?', 'spanish', None, 'needs_review'),
        ('What is your level of confidence in an English-speaking environment?', 'english_level', None, 'needs_review'),
        ('What city do you reside in?', 'residence', None, 'needs_review'),
        ('What city do you reside in?', 'residence', 'Brazil', 'applicable'),
        ('What city do you reside in?', 'residence', 'Portugal', 'applicable'),
        ('What university awarded your degree?', 'edu_google_python', None, 'needs_review'),
        ('Are you authorized to work in Portugal?', 'work_authorization_brazil', None, 'needs_review'),
        ('Years of professional LLM experience?', 'prompt_llm_years', None, 'rejected'),
        ('Anything', 'nonexistent', None, 'rejected'),
    ]
    for question, fid, country, expected in cases:
        result = validate_candidates(question, [{'id': fid, 'cosine': 1.0}], profile, country)
        assert result['decisions'][0]['status'] == expected, (question, result)
        assert not result['ready_to_submit']
    logging.info('applicability_tests=PASSED cases=%d network_calls=0', len(cases))


def self_test(profile):
    cases = [
        ("What is your monthly salary expectation in USD?", "evidence_found", "salary_usd_month"),
        ("Qual sua pretensão salarial PJ em reais?", "missing_information", None),
        ("Expected annual salary in USD?", "missing_information", None),
        ("What is your monthly gross compensation in USD?", "missing_information", None),
        ("What is your current location?", "evidence_found", "residence"),
        ("Em qual cidade aceita trabalhar presencial?", "missing_information", None),
        ("Current company?", "evidence_found", "current_company_answer"),
        ("Do you have fine-tuning experience?", "evidence_found", "laya_finetuning_project"),
        ("How many years of professional machine learning?", "needs_review", "laya_finetuning_project"),
        ("Você tem graduação?", "missing_information", None),
        ("What is your salary and current location?", "missing_information", None),
        ("Experiência com backtesting no MT5?", "evidence_found", "financial_strategy_experience"),
        ("Quanto deseja ganhar por mês em USD?", "evidence_found", "salary_usd_month"),
        ("¿Cuánto esperas cobrar por mes en USD?", "evidence_found", "salary_usd_month"),
        ("What monthly pay do you expect in US dollars?", "evidence_found", "salary_usd_month"),
        ("What monthly salary in Canadian dollars?", "missing_information", None),
        ("Quanto quer ganhar por mês em dólares?", "missing_information", None),
        ("USD monthly contractor compensation?", "missing_information", None),
        ("What weekly pay do you expect in USD?", "missing_information", None),
        ("¿Cuál es tu ciudad de residencia?", "evidence_found", "residence"),
        ("¿En qué empresa trabajas?", "evidence_found", "current_company_answer"),
        ("What is your current employer?", "evidence_found", "current_company_answer"),
        ("Experiência com ajuste fino de modelos?", "evidence_found", "laya_finetuning_project"),
        ("Experiencia profesional con fine tuning en producción?", "needs_review", "laya_finetuning_project"),
        ("Where would you like to live?", "missing_information", None),
    ]
    for question, status, fact_id in cases:
        result = retrieve(question, profile)
        assert result["status"] == status, (question, result)
        assert [r["id"] for r in result["evidence"]] == ([fact_id] if fact_id else [])
        assert result["ready_to_submit"] is False
    # Pending facts must never be promoted by matching a keyword.
    altered = json.loads(json.dumps(profile))
    for fact in altered["facts"]:
        fact["status"] = "needs_clarification"
    assert not retrieve(cases[0][0], altered)["evidence"]
    logging.info("self_test=PASSED cases=%d pending_guard=PASSED", len(cases))
    applicability_test(profile)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question")
    group.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    profile_path = SELF_TEST_PROFILE if args.self_test else PROFILE
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if args.self_test:
        self_test(profile)
    else:
        logging.info("retrieval=%s", json.dumps(retrieve(args.question, profile), ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("retrieval_failed")
        raise
