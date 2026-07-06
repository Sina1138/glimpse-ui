"""Optional Gemini backend for polarity + topic sentence classification.

This is a drop-in sibling of the local `InteractiveReviewProcessor.predict_polarity`
/ `predict_topic` methods. It returns the **exact same contracts** so callers can swap
backends transparently:

    predict_polarity_gemini(sentences) -> {sentence: "➕" | "➖" | None}
    predict_topic_gemini(sentences)    -> {sentence: <one of 7 topic strings> | None}

It is used in two places:
  * the public demo runtime (behind the Local/Gemini privacy toggle, with fallback), and
  * the offline study-data preprocessing script (behind a --classifier-backend flag).

RSA/consensuality is intentionally NOT here — it is a white-box algorithm that cannot be
served by a generation API and always runs locally.

Requires the `google-genai` package and a `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) env var.
No torch, no Gradio — pure API client.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Auto-load .env (gitignored, see .env.example) so GEMINI_API_KEY doesn't need to be
# exported manually every session. Never overrides an already-exported env var.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Label taxonomies — these MUST stay in lockstep with the local backend.
#   Polarity: interactive_processor.predict_polarity  emoji_map = {0:"➖",1:None,2:"➕"}
#   Topic:    interactive_processor.id2topic  (7 strings + None), also the keys of
#             constants.TOPIC_HTML_COLORS / TOPIC_COLOR_MAP.
# If either taxonomy changes there, change it here too.
# ---------------------------------------------------------------------------
POLARITY_ENUM = ["positive", "negative", "neutral"]
_POLARITY_TO_SYMBOL: Dict[str, Optional[str]] = {
    "positive": "➕",
    "negative": "➖",
    "neutral": None,
}

# The 7 real topic labels (exact id2topic strings) plus an explicit unclassified bucket.
TOPIC_LABELS = [
    "Substance",
    "Clarity",
    "Soundness/Correctness",
    "Originality",
    "Motivation/Impact",
    "Meaningful Comparison",
    "Replicability",
]
_TOPIC_UNCLASSIFIED = "Unclassified"
TOPIC_ENUM = TOPIC_LABELS + [_TOPIC_UNCLASSIFIED]

# Pinned, versioned Flash model (not the floating "-latest" alias) for reproducibility.
_DEFAULT_MODEL = "gemini-2.5-flash"
# Request timeout so the demo fallback fires quickly instead of hanging (milliseconds).
# Gemini 2.5 Flash defaults to "thinking" mode, which is disabled below (thinking_budget=0)
# since this is pure classification — without that, 15s was too tight for ~50 sentences
# and tripped a client-side 504 DEADLINE_EXCEEDED. 20s gives headroom even with thinking off.
_TIMEOUT_MS = 20_000
# Chunk size to keep any single structured-output response comfortably small.
_MAX_BATCH = 80


def _api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def gemini_available() -> bool:
    """True iff the SDK imports and an API key is present. Never raises."""
    if not _api_key():
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL") or _DEFAULT_MODEL


_POLARITY_PROMPT = (
    "You are labeling the sentiment/polarity of individual sentences taken from peer "
    "reviews of scientific papers. For each numbered sentence, decide whether it expresses:\n"
    "  - \"positive\": praise, a strength, or an approving judgment about the paper;\n"
    "  - \"negative\": a criticism, weakness, concern, or disapproving judgment;\n"
    "  - \"neutral\": neither — factual restatement, a question, or a summary with no judgment.\n"
    "Return one label per input index."
)

_TOPIC_PROMPT = (
    "You are labeling the aspect/topic each sentence from a scientific peer review is about. "
    "Choose exactly one label per sentence from:\n"
    "  - \"Substance\": amount/quality of experiments, analyses, or evidence;\n"
    "  - \"Clarity\": readability, writing, presentation, organization;\n"
    "  - \"Soundness/Correctness\": technical/methodological correctness or validity;\n"
    "  - \"Originality\": novelty relative to prior work;\n"
    "  - \"Motivation/Impact\": importance, motivation, or significance of the problem;\n"
    "  - \"Meaningful Comparison\": comparison to and citation of related work/baselines;\n"
    "  - \"Replicability\": reproducibility, availability of code/data/details;\n"
    "  - \"Unclassified\": none of the above clearly applies.\n"
    "Return one label per input index."
)


def _build_schema():
    """JSON schema: array of {index:int, label:str(enum)}. Built lazily to avoid
    importing the SDK at module import time."""
    from google.genai import types

    def _item(enum_values):
        return types.Schema(
            type=types.Type.OBJECT,
            required=["index", "label"],
            properties={
                "index": types.Schema(type=types.Type.INTEGER),
                "label": types.Schema(type=types.Type.STRING, enum=list(enum_values)),
            },
        )

    return _item


def _classify(sentences: List[str], enum_values: List[str], prompt: str) -> List[Optional[str]]:
    """Return a label (from enum_values) per input sentence, in original order.

    Missing / out-of-enum entries default to None. Raises on any API error or timeout
    so callers can decide whether to fall back (demo) or fail loudly (offline build).
    """
    from google import genai
    from google.genai import types

    key = _api_key()
    if not key:
        raise RuntimeError("No GEMINI_API_KEY / GOOGLE_API_KEY set")

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=_TIMEOUT_MS))
    item_schema_factory = _build_schema()
    array_schema = types.Schema(type=types.Type.ARRAY, items=item_schema_factory(enum_values))

    labels: List[Optional[str]] = [None] * len(sentences)

    for start in range(0, len(sentences), _MAX_BATCH):
        chunk = sentences[start:start + _MAX_BATCH]
        numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(chunk))
        contents = f"{prompt}\n\nSentences:\n{numbered}"
        response = client.models.generate_content(
            model=_model_name(),
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=array_schema,
                # This is pure classification, not reasoning — disable Gemini 2.5's
                # "thinking" mode. Left on, it easily blows past our client timeout
                # on batches of even ~50 sentences.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        parsed = _parse_response(response)
        valid = set(enum_values)
        for entry in parsed:
            idx = entry.get("index")
            label = entry.get("label")
            if isinstance(idx, int) and 0 <= idx < len(chunk) and label in valid:
                labels[start + idx] = label

    return labels


def _parse_response(response) -> List[dict]:
    """Extract the list-of-dicts payload from a genai response, tolerant of shape."""
    import json

    data = getattr(response, "parsed", None)
    if data is None:
        text = getattr(response, "text", None)
        if not text:
            return []
        data = json.loads(text)
    out = []
    for entry in data or []:
        if isinstance(entry, dict):
            out.append(entry)
        else:  # pydantic-style object
            out.append({"index": getattr(entry, "index", None), "label": getattr(entry, "label", None)})
    return out


def predict_polarity_gemini(sentences: List[str]) -> Dict[str, Optional[str]]:
    """Gemini polarity. Returns {sentence: "➕" | "➖" | None}. Matches the local contract."""
    if not sentences:
        return {}
    raw = _classify(sentences, POLARITY_ENUM, _POLARITY_PROMPT)
    return {sent: _POLARITY_TO_SYMBOL.get(lbl) for sent, lbl in zip(sentences, raw)}


def predict_topic_gemini(sentences: List[str]) -> Dict[str, Optional[str]]:
    """Gemini topic. Returns {sentence: <one of the 7 topic strings> | None}."""
    if not sentences:
        return {}
    raw = _classify(sentences, TOPIC_ENUM, _TOPIC_PROMPT)
    # Map the explicit unclassified bucket (and any miss) back to None, matching id2topic.
    return {
        sent: (lbl if lbl in TOPIC_LABELS else None)
        for sent, lbl in zip(sentences, raw)
    }
