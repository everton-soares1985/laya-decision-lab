"""Finite CPU, head-only feasibility training; synthetic data, no private profile."""
import argparse
import contextlib
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from dataclasses import asdict

# ==================== CONFIGURATION ====================
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs"
MODEL_ID = "convaiinnovations/laya-multilingual"
THREADS = 4  # Leave CPU resources for the desktop.
MAX_LEN = 256  # Total short-input token budget.
HEAD_LEN = 128  # Question and options token budget.
EPOCHS = 2  # 12 examples x 2 epochs = 24 optimizer updates.
SEED = 42
# =======================================================


class LogOutput:
    """Route upstream training progress into structured logging."""
    def write(self, value):
        if value.strip():
            logging.info("upstream=%s", value.strip())

    def flush(self):
        pass


def fingerprint(model, encoder):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith("encoder.") == encoder:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    global RUN, THREADS, EPOCHS, MAX_LEN, HEAD_LEN
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    RUN = args.run_dir
    if args.dataset:
        THREADS, EPOCHS, MAX_LEN, HEAD_LEN = 6, 4, 384, 240
        sample = json.loads((args.dataset / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        if len(sample["questions"]["intent"]["criteria"]) > 7:
            MAX_LEN, HEAD_LEN = 512, 320
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("stage=imports")
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from laya.train import (TrainConfig, load_checkpoint, items_from_rows, train_model,
                            calibration_records, evaluate_records, save_checkpoint)
    if args.check_only:
        TrainConfig(freeze_encoder=not args.full, amp=False).validate()
        logging.info("dependencies_and_config=OK torch=%s full=%s", torch.__version__, args.full)
        return
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(2)
    torch.manual_seed(SEED)
    start = time.monotonic()
    logging.info("stage=download model=%s", MODEL_ID)
    # Keep the base identical to the first probe for a controlled comparison.
    revision = "1720e3e3357cfe1e281542e223f8273b0890ca34"
    folder = snapshot_download(MODEL_ID, revision=revision, allow_patterns=[
        "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
    logging.info("stage=load revision=%s", revision)
    model, tokenizer, cfg = load_checkpoint(folder)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if not args.full:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("mode=%s total_parameters=%d trainable_parameters=%d",
                 "FULL" if args.full else "HEAD_ONLY", total_parameters, trainable_parameters)
    examples = [
        ("Qual sua pretensão salarial?", "A"),
        ("Quanto deseja receber mensalmente?", "A"),
        ("What salary do you expect?", "A"),
        ("What is your annual compensation expectation?", "A"),
        ("¿Cuál es tu expectativa salarial?", "A"),
        ("¿Qué remuneración mensual esperas?", "A"),
        ("Em qual cidade você mora?", "B"),
        ("Qual é seu país de residência?", "B"),
        ("Where do you currently live?", "B"),
        ("What is your country of residence?", "B"),
        ("¿En qué ciudad resides?", "B"),
        ("¿Cuál es tu ubicación actual?", "B"),
    ]
    held = [("Informe sua remuneração desejada.", "A"),
            ("Please provide your current location.", "B"),
            ("Indique cuánto espera cobrar.", "A")]
    questions = {"intent": {"type": "choice", "instructions": "Classify the form question.",
                            "criteria": {"A": "Salary or compensation expectation",
                                         "B": "Current location or residence"}}}

    def rows(pairs):
        return [{"state": text, "questions": questions, "expected": {"intent": label}}
                for text, label in pairs]

    training_rows, evaluation_rows = rows(examples), rows(held)
    if args.dataset:
        training_rows = [json.loads(line) for line in (args.dataset / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        evaluation_rows = [json.loads(line) for line in (args.dataset / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    for name, data in [("train", training_rows), ("smoke_eval", evaluation_rows)]:
        (RUN / (name + ".jsonl")).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8")
    items, skipped = items_from_rows(tokenizer, training_rows, MAX_LEN, HEAD_LEN)
    eval_items, eval_skipped = items_from_rows(tokenizer, evaluation_rows, MAX_LEN, HEAD_LEN)
    if skipped or eval_skipped or not items or not eval_items:
        raise ValueError(f"Invalid items: train={skipped} eval={eval_skipped}")
    validation_items = []
    validation_rows = []
    if args.dataset and (args.dataset / "validation.jsonl").exists():
        validation_rows = [json.loads(s) for s in (args.dataset / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
        validation_items, invalid = items_from_rows(tokenizer, validation_rows, MAX_LEN, HEAD_LEN)
        if invalid or not validation_items:
            raise ValueError(f"Invalid validation data: {invalid}")
    from laya.train import encode_item
    from prepare_dataset import norm
    seen_text, seen_families = set(), set()
    for name, data, prepared in [("train", training_rows, items),
                                  ("test", evaluation_rows, eval_items),
                                  ("validation", validation_rows, validation_items)]:
        texts = {norm(r["state"]) for r in data}
        families = {r["family"] for r in data if "family" in r}
        if seen_text & texts or seen_families & families:
            raise ValueError(f"Split leakage in {name}")
        seen_text.update(texts)
        seen_families.update(families)
        for item in prepared:
            encoded = encode_item(tokenizer, item, MAX_LEN, HEAD_LEN)
            if len(encoded["markers"]) != item["k"]:
                raise ValueError("Option markers lost")
        logging.info("validated_split=%s rows=%d", name, len(prepared))
    if args.validate_only:
        with torch.inference_mode():
            records = calibration_records(model, tokenizer, items[:1], torch.device("cpu"),
                                          MAX_LEN, HEAD_LEN, batch_size=1)
        if not records:
            raise ValueError("Forward pass failed")
        logging.info("validation=PASSED max_len=%d head_len=%d", MAX_LEN, HEAD_LEN)
        return
    config = TrainConfig(epochs=EPOCHS, micro_batch=1, grad_accum=1,
                         freeze_encoder=not args.full, loss="soft-ce", max_len=MAX_LEN,
                         head_max_len=HEAD_LEN, log_every=1, seed=SEED,
                         amp=False, gradient_checkpointing=False,
                         shuffle_options=("choice",))
    device = torch.device("cpu")

    def evaluate():
        return evaluate_records(calibration_records(model, tokenizer, eval_items, device,
                                                    MAX_LEN, HEAD_LEN, batch_size=1))

    logging.info("stage=baseline_evaluation")
    before = evaluate()
    head_before, encoder_before = fingerprint(model, False), fingerprint(model, True)
    updates = len(items) * EPOCHS
    logging.info("stage=training updates=%d freeze_encoder=%s", updates, not args.full)
    training_start = time.monotonic()
    with contextlib.redirect_stdout(LogOutput()):
        history = train_model(model, tokenizer, items, config, device, MAX_LEN, HEAD_LEN)
    training_seconds = time.monotonic() - training_start
    head_changed = head_before != fingerprint(model, False)
    encoder_unchanged = encoder_before == fingerprint(model, True)
    after = evaluate()
    validation_metrics = None
    if validation_items:
        validation_metrics = evaluate_records(calibration_records(
            model, tokenizer, validation_items, device, MAX_LEN, HEAD_LEN, batch_size=1))
        logging.info("validation_metrics=%s", validation_metrics)
    if not all(math.isfinite(loss) for loss in history) or not head_changed or encoder_unchanged == args.full:
        raise RuntimeError("Nonfinite loss, no head update, or encoder update inconsistent with mode")
    logging.info("stage=saving_checkpoint")
    cfg.update({"max_len": MAX_LEN, "head_max_len": HEAD_LEN})
    cfg["local_probe_training"] = asdict(config)
    save_checkpoint(model, tokenizer, cfg, str(RUN / "checkpoint"))
    result = {"purpose": "intent_v1_training" if args.dataset else "feasibility_only_not_generalization_benchmark", "model": MODEL_ID,
              "revision": revision, "device": "cpu", "threads": THREADS,
              "mode": "full" if args.full else "head_only",
              "total_parameters": total_parameters, "trainable_parameters": trainable_parameters,
              "training_config": asdict(config),
              "train_examples": len(items), "eval_examples": len(eval_items), "optimizer_updates": updates,
              "head_changed": head_changed, "encoder_unchanged": encoder_unchanged,
              "epoch_losses": history, "before": before, "after": after,
              "training_seconds": training_seconds, "seconds_per_update": training_seconds / updates,
              "total_worker_seconds": time.monotonic() - start,
              "calibrated": False, "checkpoint": str(RUN / "checkpoint")}
    result["validation"] = validation_metrics
    result["dataset"] = str(args.dataset) if args.dataset else None
    (RUN / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logging.info("stage=completed training_seconds=%.2f", training_seconds)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("training_probe_failed")
        raise
