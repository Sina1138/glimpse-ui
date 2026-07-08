"""Score the frozen study papers into a standalone study dataset.

Reads frozen review texts from study/papers_raw/ and writes
study/study_data/study_scored_reviews.csv in the same schema as the demo
preprocessed CSV, except the `year` column holds the TASK LABEL (directory
name, e.g. "Practice", "Paper A", "Paper B") and each row contains exactly
one submission. The study interface builds (Demo_study_full.py /
Demo_study_no_highlight.py) read from this file only — the demo dataset in
data/ is never shown to participants.

Input layout (one directory per task):

    study/papers_raw/
        Practice/
            meta.json        {"forum_url": "https://openreview.net/forum?id=...",
                              "paper_title": "...", "iclr_year": 2025,
                              "abstract": "..."}  # abstract optional but
                                                  # expected (shown in the UI)
            review_1.txt     frozen review texts, one file per review,
            review_2.txt     in reviewer order (R1, R2, ...)
            ...

Rebuttals are intentionally NOT included: the study shows the review state
that matches the frozen reviewer-aspect reference.

Usage:
    python pipeline/preprocess_study_papers.py                # all tasks
    python pipeline/preprocess_study_papers.py --tasks Practice "Paper A"
    python pipeline/preprocess_study_papers.py --list          # show inputs, no scoring
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_dir = Path(__file__).resolve().parent
sys.path[:0] = [str(_dir), str(_dir.parent)]

BASE_DIR = _dir.parent
RAW_DIR = BASE_DIR / "study" / "papers_raw"
OUT_DIR = BASE_DIR / "study" / "study_data"
OUT_CSV = OUT_DIR / "study_scored_reviews.csv"


def _discover_tasks(selected=None):
    """Return [(task_label, task_dir)] for every papers_raw subdir with meta.json."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {RAW_DIR}")
    tasks = []
    for d in sorted(RAW_DIR.iterdir()):
        if d.is_dir() and (d / "meta.json").exists():
            if selected and d.name not in selected:
                continue
            tasks.append((d.name, d))
    return tasks


def _read_task_inputs(task_dir: Path):
    """Read meta.json + ordered review texts for one task."""
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    for key in ("forum_url", "paper_title", "iclr_year"):
        if key not in meta:
            raise ValueError(f"{task_dir / 'meta.json'} is missing required key: {key}")
    if not str(meta.get("abstract", "")).strip():
        print(f"[STUDY] WARNING: {task_dir / 'meta.json'} has no 'abstract' — "
              "the UI will show the reviews without one.")
    review_files = sorted(task_dir.glob("review_*.txt"))
    texts = []
    for f in review_files:
        t = f.read_text(encoding="utf-8").strip()
        if t:
            texts.append(t)
    if len(texts) < 2:
        raise ValueError(f"{task_dir}: need >= 2 non-empty review_*.txt files, found {len(texts)}")
    return meta, texts


def _classify_pool(processor, sentences, classifier_backend):
    """Polarity + topic over the sentence pool, via local models or Gemini.

    Returns (polarity_emoji_map, topic_map) with the exact same contract as the
    local processor methods. Gemini is used only when explicitly selected; there is
    NO fallback here — an offline supervised build should fail loudly if Gemini errors.
    """
    if classifier_backend == "gemini":
        from interface import gemini_backend as gb
        if not gb.gemini_available():
            raise RuntimeError(
                "classifier-backend=gemini but google-genai/GEMINI_API_KEY is not available"
            )
        print(f"[STUDY] Classifying {len(sentences)} sentences via Gemini ({gb._model_name()}) ...")
        return gb.predict_polarity_gemini(sentences), gb.predict_topic_gemini(sentences)
    return processor.predict_polarity(sentences), processor.predict_topic(sentences)


def _score_task(processor, task_label: str, meta: dict, texts: list,
                classifier_backend: str = "local") -> dict:
    """Score one submission. Returns a row dict for the output CSV.

    Mirrors the demo pipeline's schema exactly:
      scored_dict[forum_url] = [per review: {"sentences": {sent: {consensuality,
          polarity (0/1/2), topic}}, "rebuttal": ""}]
      metadata[forum_url] = {rebuttal, paper_title, abstract, has_rebuttal,
          iclr_year, rsa: {listener, speaker}}
    Consensuality values are stored RAW (the pre-processed tab median/IQR
    normalizes at render time, same as the demo dataset).
    """
    from dependencies.Glimpse_tokenizer import glimpse_tokenizer
    from dependencies.sentence_filter import filter_and_clean_sentences
    from dependencies.rsa_reranker import RSARerankingCached as RSAReranking

    forum_url = meta["forum_url"]

    sentence_lists = [[s for s in glimpse_tokenizer(t) if s.strip()] for t in texts]
    unique_sentences = list(set(s for sl in sentence_lists for s in sl))
    scored_sentences_pool = filter_and_clean_sentences(unique_sentences)

    # --- Polarity + topic (local models or Gemini, per --classifier-backend) ---
    polarity_emoji, topic_map = _classify_pool(processor, scored_sentences_pool, classifier_backend)
    emoji_to_num = {"➖": 0, None: 1, "➕": 2}

    # --- RSA / GLIMPSE: raw consensuality + listener/speaker distributions ---
    print(f"[STUDY] {task_label}: running RSA over {len(scored_sentences_pool)} sentences "
          f"across {len(texts)} reviews ...")
    rsa_reranker = RSAReranking(
        processor.rsa_model,
        processor.rsa_tokenizer,
        candidates=scored_sentences_pool,
        source_texts=list(texts),
        device=str(processor.device),
        rationality=1.0,
    )
    _, _, speaker_df, listener_df, _, _, _, consensuality_raw = rsa_reranker.rerank(t=1)
    consensuality = {s: float(v) for s, v in dict(consensuality_raw).items()}

    review_labels = [f"R{i+1}" for i in range(len(texts))]

    listener_probs = np.exp(listener_df.values)
    col_sums = listener_probs.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    listener_probs = listener_probs / col_sums
    listener = {
        sent: {review_labels[i]: float(listener_probs[i, j]) for i in range(len(texts))}
        for j, sent in enumerate(listener_df.columns)
    }

    speaker_probs = np.exp(speaker_df.values)
    row_sums = speaker_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    speaker_probs = speaker_probs / row_sums
    speaker = {
        review_labels[i]: {sent: float(speaker_probs[i, j]) for j, sent in enumerate(speaker_df.columns)}
        for i in range(len(texts))
    }

    # --- Assemble per-review sentence dicts ---
    # Every tokenized sentence is included (text fidelity for participants);
    # noise-filtered sentences simply carry no scores.
    per_review = []
    for sents in sentence_lists:
        sentences = {}
        for sent in sents:
            data = {}
            if sent in consensuality:
                data["consensuality"] = consensuality[sent]
            if sent in polarity_emoji:
                data["polarity"] = emoji_to_num.get(polarity_emoji[sent], 1)
            if sent in topic_map:
                data["topic"] = topic_map[sent] or "NONE"
            sentences[sent] = data
        per_review.append({"sentences": sentences, "rebuttal": ""})

    return {
        "year": task_label,
        "scored_dict": {forum_url: per_review},
        "metadata": {
            forum_url: {
                "rebuttal": "",
                "paper_title": meta["paper_title"],
                "abstract": str(meta.get("abstract", "")).strip(),
                "has_rebuttal": False,
                "iclr_year": meta["iclr_year"],
                "rsa": {"listener": listener, "speaker": speaker},
            }
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build the study-papers dataset")
    parser.add_argument("--tasks", nargs="+", help="Task labels to include (default: all)")
    parser.add_argument("--list", action="store_true", help="List discovered inputs and exit")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument(
        "--classifier-backend",
        choices=["local", "gemini"],
        default=os.environ.get("STUDY_CLASSIFIER_BACKEND", "local"),
        help="Backend for polarity/topic (RSA always local). Default: local, "
             "or $STUDY_CLASSIFIER_BACKEND. Use 'gemini' only if the validation gate "
             "(validation/validate_gemini_classifiers.py) selected it.",
    )
    args = parser.parse_args()

    tasks = _discover_tasks(args.tasks)
    if not tasks:
        print(f"No task directories with meta.json found under {RAW_DIR}")
        sys.exit(1)

    print(f"Discovered {len(tasks)} task(s):")
    for label, d in tasks:
        n = len(list(d.glob("review_*.txt")))
        print(f"  {label}: {n} review files  ({d})")
    if args.list:
        return

    import torch
    from interface.interactive_processor import InteractiveReviewProcessor
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on {device} ...")
    processor = InteractiveReviewProcessor(device=device)
    processor.ensure_device()

    print(f"Classifier backend: {args.classifier_backend} (RSA always local)")
    rows = []
    for label, d in tasks:
        meta, texts = _read_task_inputs(d)
        rows.append(_score_task(processor, label, meta, texts, args.classifier_backend))
        print(f"[STUDY] {label}: done ({len(texts)} reviews)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n✓ Study dataset saved to: {OUT_CSV}")
    print(f"  Tasks: {', '.join(label for label, _ in tasks)}")
    print(f"  File size: {OUT_CSV.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
