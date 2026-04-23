"""
genai_translator.py
-------------------
LLM-powered post-processing layer. Responsibilities:

    1. Correct OCR errors (swapped / missing glyphs)
    2. Reconstruct plausible readings for damaged portions (marked with ⟨…⟩)
    3. Translate into modern English or Hindi
    4. Produce a clean, human-readable output with optional alternative
       interpretations when the text is ambiguous.

Backends supported
------------------
- Google Gemini   (via `google-generativeai`) — primary
- Groq            (via `groq`)                 — automatic fallback
- Local HuggingFace (e.g., IndicTrans2)        — offline option
- StubTranslator                               — when nothing is configured

`make_translator("auto")` returns a FailoverTranslator that tries Gemini
first and falls back to Groq the moment Gemini throws. If only one key is
set, the failover wrapper degrades gracefully to the single configured
backend.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TranslationOutput:
    cleaned_text: str = ""
    english: str = ""
    hindi: str = ""
    alternatives: List[str] = field(default_factory=list)
    notes: str = ""
    confidence: float = 0.0
    backend: str = ""

    def as_dict(self) -> Dict:
        return {
            "cleaned_text": self.cleaned_text,
            "english": self.english,
            "hindi": self.hindi,
            "alternatives": self.alternatives,
            "notes": self.notes,
            "confidence": self.confidence,
            "backend": self.backend,
        }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Indologist and epigraphist specializing in
ancient Indian scripts (Brahmi, Tamil-Brahmi, Kharoshthi, early Sanskrit,
Prakrit, Pali, and medieval Tamil / Devanagari). You will be given:

  1. Raw OCR output from a photograph of a temple inscription. The OCR is
     noisy — glyphs may be missing, merged, or misread.
  2. A rough modern-script rendering produced by a dictionary-based mapping.
  3. Optional metadata (script guess, language, region, date).

Your job:
  - Infer the most likely original text. Where a portion is unreadable, mark
    it with angle brackets ⟨…⟩ and give your best reconstruction.
  - Produce a clean transliteration in IAST (Roman) and a modern-script
    form (Devanagari for Sanskrit/Prakrit/Pali; Tamil for Tamil-Brahmi).
  - Translate into clear, readable MODERN ENGLISH. Prioritise readability
    over word-for-word literalism.
  - Also provide a translation in MODERN HINDI.
  - If the inscription is ambiguous, list up to 2 alternative readings.
  - Explain any reconstruction choices briefly in "notes".
  - Give a self-assessed confidence score from 0.0 to 1.0.

Output STRICT JSON with exactly these keys:
  "cleaned_text", "english", "hindi", "alternatives", "notes", "confidence"

Do not include any prose outside the JSON.
"""


def _build_user_prompt(raw_text: str, mapped_text: str,
                       script_hint: Optional[str] = None,
                       extra_context: Optional[str] = None) -> str:
    parts = ["OCR raw text:", raw_text or "(empty)", ""]
    parts += ["Dictionary-mapped modern form:", mapped_text or "(empty)", ""]
    if script_hint:
        parts += [f"Script hint: {script_hint}", ""]
    if extra_context:
        parts += ["Extra context:", extra_context, ""]
    parts.append("Return JSON only.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class BaseTranslator:
    name: str = "base"

    def translate(self, raw_text: str, mapped_text: str,
                  script_hint: Optional[str] = None,
                  extra_context: Optional[str] = None) -> TranslationOutput:
        raise NotImplementedError


# ----- Google Gemini backend -----
class GeminiTranslator(BaseTranslator):
    """
    Uses Google's Gemini API. Requires GEMINI_API_KEY (or GOOGLE_API_KEY).

    Supports BOTH the new `google-genai` SDK (preferred) and the legacy
    `google-generativeai` SDK (fallback), so existing installs keep working
    while new ones use the maintained package.

    Default model: `gemini-1.5-pro`. Use `gemini-1.5-flash` for speed/cost,
    or `gemini-2.5-pro` / `gemini-2.5-flash` if your key has access.
    """
    name = "gemini"

    def __init__(self, model: str = "gemini-1.5-pro",
                 api_key: Optional[str] = None):
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")

        self._api_key = key
        self._model_name = model
        self._mode = None  # "new" (google-genai) or "legacy" (google-generativeai)

        # Try the new SDK first.
        try:
            from google import genai as _new_genai
            from google.genai import types as _new_types
            self._new_genai = _new_genai
            self._new_types = _new_types
            self._client = _new_genai.Client(api_key=key)
            self._mode = "new"
            return
        except ImportError:
            pass

        # Fall back to the legacy SDK.
        try:
            import google.generativeai as _legacy_genai  # type: ignore
            _legacy_genai.configure(api_key=key)
            self._legacy_genai = _legacy_genai
            self._legacy_model = _legacy_genai.GenerativeModel(
                model_name=model,
                system_instruction=SYSTEM_PROMPT,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                    "max_output_tokens": 1500,
                },
            )
            self._mode = "legacy"
            return
        except ImportError as e:
            raise ImportError(
                "No Gemini SDK installed. Run: pip install google-genai "
                "(or the legacy: pip install google-generativeai)"
            ) from e

    def translate(self, raw_text, mapped_text, script_hint=None, extra_context=None):
        user = _build_user_prompt(raw_text, mapped_text, script_hint, extra_context)

        if self._mode == "new":
            config = self._new_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=1500,
            )
            resp = self._client.models.generate_content(
                model=self._model_name,
                contents=user,
                config=config,
            )
            text = getattr(resp, "text", None) or ""
            if not text:
                # Fallback: walk candidates directly.
                try:
                    text = "".join(
                        part.text for cand in (resp.candidates or [])
                        for part in (cand.content.parts or []) if hasattr(part, "text")
                    )
                except Exception:
                    text = ""
        else:  # legacy
            resp = self._legacy_model.generate_content(user)
            text = getattr(resp, "text", None) or ""
            if not text:
                try:
                    text = "".join(
                        part.text for cand in resp.candidates
                        for part in cand.content.parts if hasattr(part, "text")
                    )
                except Exception:
                    text = ""

        out = _parse_json_output(text)
        out.backend = f"gemini:{self._model_name}({self._mode})"
        return out


# ----- Groq backend -----
class GroqTranslator(BaseTranslator):
    """
    Uses Groq's OpenAI-compatible chat completions endpoint.
    Requires GROQ_API_KEY. Llama 3.3 70B is a strong default for this task.
    """
    name = "groq"

    def __init__(self, model: str = "llama-3.3-70b-versatile",
                 api_key: Optional[str] = None):
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError("pip install groq") from e
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set")
        self._client = Groq(api_key=key)
        self._model = model

    def translate(self, raw_text, mapped_text, script_hint=None, extra_context=None):
        user = _build_user_prompt(raw_text, mapped_text, script_hint, extra_context)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1500,
        )
        text = resp.choices[0].message.content or ""
        out = _parse_json_output(text)
        out.backend = f"groq:{self._model}"
        return out


# ----- Failover wrapper -----
class FailoverTranslator(BaseTranslator):
    """
    Tries each wrapped backend in order and returns the first successful
    response. A backend is considered "failed" if:
      - it raises an exception during `.translate()`, OR
      - it returns an empty result (no english/cleaned_text at all).
    """
    name = "failover"

    def __init__(self, backends: List[BaseTranslator]):
        if not backends:
            raise ValueError("FailoverTranslator needs at least one backend")
        self._backends = backends

    def translate(self, raw_text, mapped_text, script_hint=None, extra_context=None):
        last_err: Optional[Exception] = None
        for b in self._backends:
            try:
                log.info("FailoverTranslator: trying %s", b.name)
                out = b.translate(raw_text, mapped_text, script_hint, extra_context)
                if (out.english or out.cleaned_text or out.hindi):
                    return out
                log.warning("%s returned empty output, falling back.", b.name)
            except Exception as e:
                log.warning("%s failed (%s: %s); trying next backend.",
                            b.name, type(e).__name__, e)
                last_err = e
        # Nothing worked — degrade to the stub so the pipeline still returns
        # *something* rather than throwing.
        stub = StubTranslator().translate(raw_text, mapped_text, script_hint, extra_context)
        stub.notes = (stub.notes + f" All LLM backends failed. Last error: {last_err}").strip()
        return stub


# ----- Local HuggingFace backend (offline) -----
class LocalHFTranslator(BaseTranslator):
    """
    Minimal offline fallback. Uses an Indic seq2seq model (IndicTrans2 is a
    good default) to translate the dictionary-mapped form into English/Hindi.
    It cannot do reasoning/reconstruction like the hosted LLMs.
    """
    name = "local-hf"

    def __init__(self, model_name: str = "ai4bharat/indictrans2-indic-en-1B"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        self._model_name = model_name

    def _translate(self, text: str) -> str:
        enc = self.tok(text, return_tensors="pt")
        out = self.mdl.generate(**enc, max_new_tokens=256)
        return self.tok.decode(out[0], skip_special_tokens=True)

    def translate(self, raw_text, mapped_text, script_hint=None, extra_context=None):
        src = mapped_text or raw_text
        eng = self._translate(src)
        return TranslationOutput(
            cleaned_text=src,
            english=eng,
            hindi="",
            notes="Offline HF model — no reconstruction attempted.",
            confidence=0.45,
            backend=f"local-hf:{self._model_name}",
        )


# ----- Rule-based stub (used if no backend available) -----
class StubTranslator(BaseTranslator):
    """Fallback when no LLM is configured — returns the mapped text verbatim."""
    name = "stub"

    def translate(self, raw_text, mapped_text, script_hint=None, extra_context=None):
        return TranslationOutput(
            cleaned_text=mapped_text or raw_text,
            english="[No LLM backend configured — set GEMINI_API_KEY or GROQ_API_KEY.]",
            hindi="[कोई LLM बैकएंड कॉन्फ़िगर नहीं — कृपया GEMINI_API_KEY या GROQ_API_KEY सेट करें।]",
            notes="StubTranslator active — dictionary output only.",
            confidence=0.15,
            backend="stub",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_output(text: str) -> TranslationOutput:
    """Parse LLM output, tolerating stray markdown fences."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        # drop a possible language tag
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.endswith("```"):
            t = t[:-3]
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        log.warning("LLM did not return valid JSON; wrapping raw text.")
        return TranslationOutput(english=text or "", confidence=0.3,
                                 notes="Non-JSON response from LLM.")
    return TranslationOutput(
        cleaned_text=obj.get("cleaned_text", ""),
        english=obj.get("english", ""),
        hindi=obj.get("hindi", ""),
        alternatives=obj.get("alternatives", []) or [],
        notes=obj.get("notes", ""),
        confidence=float(obj.get("confidence", 0.0) or 0.0),
    )


def make_translator(preferred: str = "auto") -> BaseTranslator:
    """
    Factory that picks the best available backend.

    preferred:
        "auto"       — Gemini (if key) → Groq (if key) → Stub, with runtime failover
        "gemini"     — Gemini only
        "groq"       — Groq only
        "failover"   — Same as "auto" but always returns FailoverTranslator
        "local"      — Offline HF model
        "stub"       — Never calls the network
    """
    preferred = (preferred or "auto").lower()

    if preferred == "gemini":
        return GeminiTranslator()
    if preferred == "groq":
        return GroqTranslator()
    if preferred == "local":
        return LocalHFTranslator()
    if preferred == "stub":
        return StubTranslator()

    # auto / failover — try to assemble a chain of whichever backends we can build.
    chain: List[BaseTranslator] = []
    for ctor, label in [(GeminiTranslator, "Gemini"), (GroqTranslator, "Groq")]:
        try:
            chain.append(ctor())
            log.info("%s backend initialised.", label)
        except Exception as e:
            log.info("%s backend unavailable: %s", label, e)

    if not chain:
        log.warning("No hosted LLM backends available — using StubTranslator.")
        return StubTranslator()

    if len(chain) == 1 and preferred == "auto":
        return chain[0]
    return FailoverTranslator(chain)
