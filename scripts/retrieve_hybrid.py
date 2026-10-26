"""Hybrid retrieval experiment: --benchmark --use-api; --self-test is offline."""
import argparse
import hashlib
import json
import logging
import math
import re
import sqlite3
import urllib.request
from datetime import datetime
from pathlib import Path

from retrieve_openrouter import CATALOG, confirmed_catalog, PacedClient, api_key
from retrieve_knowledge import retrieve, normalize

# ==================== CONFIGURATION ====================
ROOT = Path(__file__).resolve().parents[1]
MODEL = 'openai/text-embedding-3-small'
URL = 'https://openrouter.ai/api/v1/embeddings'
DIMENSIONS = 1536
CACHE = ROOT/'data/knowledge/embedding_cache.sqlite3'
VECTOR_PATH = ROOT/'data/knowledge/qdrant'
RRF_K = 60  # Equal-weight reciprocal-rank fusion; not a calibrated probability.
CASES = [  # Authored paraphrases, not unseen real-world benchmark.
    ('How would you rate your command of the English language?', 'english_level'),
    ('¿Cuándo podrías incorporarte si necesitamos que empieces ya?', 'immediate_start'),
    ('Name the municipality you currently call home.', 'residence'),
    ('Indique el importe que desea cobrar cada mes en dólares estadounidenses.', 'salary_usd_month'),
    ('Tell us about your personal experiment adapting the Laya model.', 'laya_finetuning_project'),
    ('How long have you been using Python in practical projects?', 'python_practical_years'),
    ('Which university granted your degree?', None),
    ('Do you hold a work permit for Portugal?', None),
    ('Qual é seu nível de espanhol?', 'spanish'),
    ('What hourly salary do you expect in USD?', 'salary_usd_hour'),
    ('Pretensão salarial mensal em reais para trabalho remoto?', 'salary_brl_month_remote'),
    ('Pretensão salarial mensal em reais para trabalho híbrido?', 'salary_brl_month_hybrid'),
    ('Have you completed the RAG course from DeepLearning.AI?', 'edu_rag_deeplearning'),
    ('Have you completed Google IT Automation with Python?', 'edu_google_python'),
    ('Are you authorized to work in Brazil?', 'work_authorization_brazil'),
    ('Are you available for hybrid work in Lima twice a week?', 'hybrid_lima'),
]
# =======================================================


def key(text):
    return hashlib.sha256(f'{MODEL}:{DIMENSIONS}:{text}'.encode()).hexdigest()


def validate_vectors(data, count):
    rows = data.get('data')
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError('Embedding count mismatch')
    indexed = {}
    for row in rows:
        index, vector = row.get('index'), row.get('embedding')
        if type(index) is not int or index in indexed or not 0 <= index < count:
            raise ValueError('Invalid embedding index')
        if not isinstance(vector, list) or len(vector) != DIMENSIONS or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in vector):
            raise ValueError('Invalid embedding vector')
        if not any(vector):
            raise ValueError('Zero embedding')
        indexed[index] = vector
    return [indexed[i] for i in range(count)]


def embed(texts, use_api, client=None):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CACHE) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)')
        vectors = {}
        for text in dict.fromkeys(texts):
            row = db.execute('SELECT vector FROM embeddings WHERE key=?', (key(text),)).fetchone()
            if row:
                vector = json.loads(row[0])
                vectors[text] = validate_vectors({'data': [{'index': 0, 'embedding': vector}]}, 1)[0]
        missing = [text for text in dict.fromkeys(texts) if text not in vectors]
        usage, transport = None, None
        if missing:
            if not use_api:
                raise ValueError('Uncached embeddings require explicit --use-api')
            # Catalog descriptions + diagnostic questions only; no private profile values.
            request = urllib.request.Request(URL, data=json.dumps({
                'model': MODEL, 'dimensions': DIMENSIONS, 'encoding_format': 'float',
                'input': missing}).encode(), headers={
                    'Authorization': 'Bearer '+api_key(), 'Content-Type': 'application/json'})
            data, transport = (client or PacedClient()).request(request)
            returned = validate_vectors(data, len(missing))
            usage = data.get('usage')
            for text, vector in zip(missing, returned):
                vectors[text] = vector
                db.execute('INSERT OR REPLACE INTO embeddings VALUES (?,?)', (key(text), json.dumps(vector)))
        return [vectors[t] for t in texts], {'new_vectors': len(missing), 'usage': usage, 'transport': transport}


def rank(db, vector_db, collection, question, vector, descriptions):
    dense = vector_db.query_points(collection_name=collection, query=vector, limit=len(descriptions)).points
    terms = sorted(set(re.findall(r'\w+', normalize(question))))
    expression = ' OR '.join('"'+term+'"' for term in terms)
    lexical = db.execute('SELECT id FROM documents WHERE documents MATCH ? ORDER BY bm25(documents)',
                         (expression,)).fetchall() if expression else []
    scores = {}
    cosine = {}
    for position, point in enumerate(dense, 1):
        fid = point.payload['fact_id']
        scores[fid] = 1/(RRF_K+position)
        cosine[fid] = point.score
    for position, (fid,) in enumerate(lexical, 1):
        scores[fid] = scores.get(fid, 0)+1/(RRF_K+position)
    ordered = sorted(scores, key=lambda fid: (-scores[fid], fid))
    return [{'id': fid, 'rrf': scores[fid], 'cosine': cosine.get(fid)} for fid in ordered[:3]]


def search_candidates(question, profile, use_api=False, client=None):
    """Retrieve descriptions only. Similarity never authorizes a factual answer."""
    from qdrant_client import QdrantClient, models
    descriptions = confirmed_catalog(profile)
    if not descriptions:
        return {'status': 'empty_catalog', 'candidates': [], 'ready_to_submit': False}
    vectors, usage = embed(list(descriptions.values())+[question], use_api, client)
    signature = hashlib.sha256(json.dumps([MODEL, DIMENSIONS, descriptions], sort_keys=True).encode()).hexdigest()[:16]
    collection = 'catalog_'+signature
    VECTOR_PATH.mkdir(parents=True, exist_ok=True)
    vector_db = QdrantClient(path=str(VECTOR_PATH))
    try:
        if not vector_db.collection_exists(collection):
            vector_db.create_collection(collection, vectors_config=models.VectorParams(size=DIMENSIONS, distance=models.Distance.COSINE))
        vector_db.upsert(collection, points=[models.PointStruct(id=i, vector=vectors[i], payload={'fact_id': fid})
                                             for i, fid in enumerate(descriptions)])
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE VIRTUAL TABLE documents USING fts5(id UNINDEXED, text, tokenize="unicode61 remove_diacritics 2")')
            db.executemany('INSERT INTO documents VALUES (?,?)', descriptions.items())
            candidates = rank(db, vector_db, collection, question, vectors[-1], descriptions)
    finally:
        vector_db.close()
    return {'status': 'candidates_only_requires_validation', 'candidates': candidates,
            'method': 'cosine+FTS5_BM25+RRF', 'usage': usage, 'ready_to_submit': False}


def benchmark(use_api):
    from qdrant_client import QdrantClient, models
    profile = json.loads((ROOT/'data/knowledge/profile_v1.json').read_text(encoding='utf-8'))
    descriptions = confirmed_catalog(profile)
    texts = list(descriptions.values()) + [q for q, _ in CASES]
    vectors, usage = embed(texts, use_api)
    signature = hashlib.sha256(json.dumps([MODEL, DIMENSIONS, descriptions], sort_keys=True).encode()).hexdigest()[:16]
    collection = 'catalog_'+signature
    VECTOR_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(VECTOR_PATH))
    results = []
    try:
        if not client.collection_exists(collection):
            client.create_collection(collection, vectors_config=models.VectorParams(size=DIMENSIONS, distance=models.Distance.COSINE))
        client.upsert(collection, points=[models.PointStruct(id=i, vector=vectors[i], payload={'fact_id': fid})
                                          for i, fid in enumerate(descriptions)])
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE VIRTUAL TABLE documents USING fts5(id UNINDEXED, text, tokenize="unicode61 remove_diacritics 2")')
            db.executemany('INSERT INTO documents VALUES (?,?)', descriptions.items())
            for (question, expected), vector in zip(CASES, vectors[len(descriptions):]):
                candidates = rank(db, client, collection, question, vector, descriptions)
                baseline = retrieve(question, profile)
                ids = [f['id'] for f in baseline['evidence']]
                results.append({'question': question, 'expected_id': expected, 'candidates': candidates,
                                'baseline_ids': ids, 'baseline_found': expected in ids if expected else None,
                                'hit_at_1': candidates[0]['id'] == expected if expected else None,
                                'hit_at_3': expected in [c['id'] for c in candidates] if expected else None,
                                'ready_to_submit': False, 'status': 'candidates_only_requires_validation'})
    finally:
        client.close()
    positives = [r for r in results if r['expected_id']]
    report = {'model': MODEL, 'indexed_records': len(descriptions), 'method': 'cosine+FTS5_BM25+RRF', 'usage': usage, 'results': results,
              'summary': {'supported_cases': len(positives), 'baseline_hits': sum(r['baseline_found'] for r in positives),
                          'hit_at_1': sum(r['hit_at_1'] for r in positives), 'hit_at_3': sum(r['hit_at_3'] for r in positives)},
              'limits': 'Authored diagnostic only. Out-of-catalog queries still return neighbors; no abstention threshold calibrated. Not integrated into answers.'}
    output = ROOT/'runs'/('hybrid_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    (output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    logging.info('hybrid_summary=%s usage=%s report=%s', report['summary'], usage, output/'result.json')


def self_test():
    profile = json.loads((ROOT/'data/knowledge/profile_v1.json').read_text(encoding='utf-8'))
    catalog = confirmed_catalog(profile)
    assert len(catalog) == sum(f['status'] == 'confirmed' for f in profile['facts'])
    assert 'prompt_llm_years' not in catalog
    altered = json.loads(json.dumps(profile))
    for fact in altered['facts']:
        fact['status'] = 'needs_clarification'
    assert not confirmed_catalog(altered)
    from qdrant_client import QdrantClient, models
    client = QdrantClient(':memory:')
    try:
        client.create_collection('test', vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
        client.upsert('test', points=[models.PointStruct(id=0, vector=[1.,0.,0.], payload={'fact_id':'a'}),
                                      models.PointStruct(id=1, vector=[0.,1.,0.], payload={'fact_id':'b'})])
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE VIRTUAL TABLE documents USING fts5(id UNINDEXED, text)')
            descriptions = {'a':'Python practical experience', 'b':'English level'}
            db.executemany('INSERT INTO documents VALUES (?,?)', descriptions.items())
            assert rank(db, client, 'test', 'Python', [1.,0.,0.], descriptions)[0]['id'] == 'a'
        try:
            validate_vectors({'data': [{'index':0, 'embedding':[float('nan')]*DIMENSIONS}]}, 1)
            raise AssertionError('Invalid vector accepted')
        except ValueError:
            pass
    finally:
        client.close()
    logging.info('hybrid_tests=PASSED network_calls=0')


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--benchmark', action='store_true')
    group.add_argument('--self-test', action='store_true')
    parser.add_argument('--use-api', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    self_test() if args.self_test else benchmark(args.use_api)


if __name__ == '__main__':
    main()
