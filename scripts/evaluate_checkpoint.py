"""Evaluate a saved checkpoint without training; log per-question errors locally."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import time
import unicodedata

# ==================== CONFIGURATION ====================
ROOT = Path(__file__).resolve().parents[1]
THREADS = 6  # CPU only; leave resources for desktop/OBS.
MAX_LEN, HEAD_LEN = 384, 240  # Match intent_v1 training.
NEW_CASES = [  # Agent-authored diagnostic examples, NOT real captured benchmark data.
    ("A", "pt", "Qual seria sua remuneração bruta desejada por ano em reais?"),
    ("A", "en", "State the minimum monthly contractor fee you would accept in dollars."),
    ("A", "es", "Indica el sueldo bruto anual que esperas en pesos mexicanos."),
    ("B", "pt", "Informe o município onde está morando atualmente."),
    ("B", "en", "Which country is your present home address in?"),
    ("B", "es", "Escribe la ciudad donde vives actualmente."),
    ("C", "pt", "Há quanto tempo você desenvolve sistemas com modelos de linguagem?"),
    ("C", "en", "Specify the number of years you have spent implementing retrieval pipelines."),
    ("C", "es", "¿Durante cuántos años has programado aplicaciones de servidor?"),
    ("D", "pt", "Como você classifica sua capacidade de conversar em inglês?"),
    ("D", "en", "Rate your spoken and written Spanish proficiency."),
    ("D", "es", "¿Qué dominio tienes del portugués hablado y escrito?"),
    ("E", "pt", "Você já integrou bancos vetoriais em aplicações de busca semântica?"),
    ("E", "en", "Have you implemented automated deployment pipelines for backend services?"),
    ("E", "es", "¿Sabes conectar servicios externos mediante APIs REST?"),
    ("F", "pt", "Descreva um projeto anterior no qual você reduziu falhas de produção."),
    ("F", "en", "Tell us about a previous project you delivered and its measurable impact."),
    ("F", "es", "Describe una solución que implementaste en un proyecto anterior."),
    ("G", "pt", "Você consegue comparecer ao escritório em Recife três dias por semana?"),
    ("G", "en", "Would you be available to commute to our office twice a week?"),
    ("G", "es", "¿Puedes trabajar de forma presencial en nuestra oficina de Guadalajara?"),
]
# =======================================================


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if c.isalnum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--field-type-only", action="store_true")
    args = parser.parse_args()
    if args.validation_only and not args.dataset:
        parser.error("--validation-only requires --dataset")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(args.output / "evaluation.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    status = {"state": "RUNNING", "pid": os.getpid(), "checkpoint": str(args.checkpoint)}

    def save_status():
        (args.output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    save_status()
    try:
        import torch
        from laya.train import load_checkpoint, items_from_rows, calibration_records, evaluate_records
        from laya.common import build_sequence
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(2)
        torch.manual_seed(42)
        start = time.monotonic()
        dataset = args.dataset or ROOT / "data" / "intent_v1"
        train = [json.loads(s) for s in (dataset / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        test = [] if args.validation_only else [json.loads(s) for s in (dataset / "test.jsonl").read_text(encoding="utf-8").splitlines()]
        questions = train[0]["questions"]
        existing = {normalize(row["state"]) for row in train + test}
        novel = [{"state": text, "questions": questions, "expected": {"intent": label},
                  "language": language, "source": "agent_authored_diagnostic"}
                 for label, language, text in NEW_CASES]
        if any(normalize(row["state"]) in existing for row in novel):
            raise ValueError("A new question duplicates an existing example")
        if not args.dataset:
            (args.output / "novel_questions.json").write_text(
                json.dumps(novel, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("stage=loading_saved_checkpoint device=cpu threads=%d", THREADS)
        model, tokenizer, cfg = load_checkpoint(str(args.checkpoint.resolve()))
        max_len = cfg.get("max_len", MAX_LEN)
        head_len = cfg.get("head_max_len", HEAD_LEN)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        results = {}
        labels = list(questions["intent"]["criteria"])
        suites = [("original_holdout", test), ("novel_synthetic", novel)]
        if args.dataset:
            validation = [json.loads(s) for s in (dataset / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
            suites = [("test", test), ("validation", validation)]
            if args.validation_only:
                suites = [("validation", validation)]
        for name, rows in suites:
            details, records = [], []
            grouped = defaultdict(lambda: {"correct": 0, "total": 0})
            for index, row in enumerate(rows, 1):
                if args.field_type_only:
                    row = dict(row)
                    field_lines = [line for line in row["state"].splitlines() if line.startswith("Field type:")]
                    if "question_text" not in row or len(field_lines) != 1:
                        raise ValueError("Field-only ablation requires context dataset")
                    row["state"] = row["question_text"] + "\n\n" + field_lines[0]
                items, skipped = items_from_rows(tokenizer, [row], max_len, head_len)
                if skipped or len(items) != 1:
                    raise ValueError(f"Invalid evaluation row {name}/{index}: {skipped}")
                with torch.inference_mode():
                    record = calibration_records(model, tokenizer, items, torch.device("cpu"),
                                                 max_len, head_len, batch_size=1)[0]
                records.append(record)
                logits = torch.as_tensor(record[1], dtype=torch.float32)
                probabilities = torch.softmax(logits, dim=-1)
                predicted = labels[int(probabilities.argmax())]
                expected = row["expected"]["intent"]
                correct = predicted == expected
                detail = {"question": row["state"], "expected": expected, "predicted": predicted,
                          "correct": correct, "confidence": float(probabilities.max()),
                          "language": row.get("language", "not_annotated")}
                detail["truncation"] = build_sequence(tokenizer, row["state"], items[0]["q"],
                    max_len, head_len, return_truncation_stats=True)[-1]
                details.append(detail)
                for group in [f"class:{expected}", f"language:{detail['language']}"]:
                    grouped[group]["total"] += 1
                    grouped[group]["correct"] += int(correct)
                logging.info("set=%s item=%d/%d expected=%s predicted=%s correct=%s question=%s",
                             name, index, len(rows), expected, predicted, correct, row["state"])
            results[name] = {"metrics": evaluate_records(records), "groups": dict(grouped),
                             "errors": [d for d in details if not d["correct"]], "details": details}
            logging.info("set=%s accuracy=%.4f errors=%d", name,
                         results[name]["metrics"]["accuracy"], len(results[name]["errors"]))
        report = {"checkpoint": str(args.checkpoint.resolve()), "seconds": time.monotonic() - start,
                  "training_performed": False, "calibrated": False,
                  "limitations": "Small previously examined diagnostic sets; not a fresh production benchmark. No training performed.",
                  "results": results}
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        status.update(state="COMPLETED", seconds=report["seconds"])
        logging.info("stage=COMPLETED report=%s", args.output / "report.json")
    except Exception as exc:
        status.update(state="FAILED", error=str(exc))
        logging.exception("evaluation_failed")
        raise
    finally:
        save_status()


if __name__ == "__main__":
    main()
