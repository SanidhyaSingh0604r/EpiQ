# Ancient Inscription Translator

A hybrid Computer-Vision + OCR + Dataset-matching + GenAI pipeline that takes a
photograph of an ancient temple inscription (Brahmi, Tamil-Brahmi, Sanskrit,
Prakrit, Pali, early Devanagari, etc.) and produces a clean, readable
translation in modern English and Hindi.

---

## 1. Project structure

```
ancient_inscription_translator/
├── main.py                 # CLI + orchestrator (InscriptionTranslator)
├── preprocessing.py        # OpenCV preprocessing pipeline
├── ocr_engine.py           # Tesseract + EasyOCR ensemble wrapper
├── dataset_matcher.py      # Script-mapping lookup (exact + Levenshtein)
├── genai_translator.py     # LLM post-processing (Claude / OpenAI / local)
├── utils.py                # I/O + the PipelineReport dataclass
├── data/
│   └── script_mappings.json  # Seed dataset — extend this
├── samples/                # Drop your test images here
└── requirements.txt
```

---

## 2. Installation

```bash
cd ancient_inscription_translator
python -m venv .venv && source .venv/bin/activate      # or Scripts\activate on Windows
pip install -r requirements.txt

# System dependency: Tesseract binary
#   Ubuntu   :  sudo apt install tesseract-ocr
#   macOS    :  brew install tesseract
#   Windows  :  https://github.com/UB-Mannheim/tesseract/wiki

# Language packs (put in your tessdata directory):
#   https://github.com/tesseract-ocr/tessdata_best
#     - san.traineddata   (Sanskrit)
#     - tam.traineddata   (Tamil)
#     - hin.traineddata   (Hindi / Devanagari)
#     - eng.traineddata   (English)
```

Set one or both of these environment variables before running so the LLM
step works. With `--backend auto` the pipeline uses **Gemini first** and
automatically **fails over to Groq** on any error (auth issues, quota,
timeouts, empty output, etc.):

```bash
export GEMINI_API_KEY=...      # Google AI Studio → https://aistudio.google.com/apikey
export GROQ_API_KEY=...        # https://console.groq.com/keys
```

If only one key is set, the pipeline uses that backend alone. If neither
is set, it falls back to the `StubTranslator` that emits the
dictionary-mapped form only.

---

## 3. Quick start

### Web UI (Streamlit)

```bash
# from the project root, with your venv activated and keys exported
streamlit run streamlit_app.py
```

Then open the printed URL (defaults to http://localhost:8501). Drag an
inscription photo into the uploader, tweak the sidebar (backend, script
hint, perspective toggle, OCR languages), and hit **Translate**. You get:

- Original, preprocessed, and bounding-box-highlighted images side-by-side
- All seven preprocessing stages under a collapsible panel
- Tabbed translation output: Modern English, Modern Hindi, cleaned
  transliteration, raw OCR + dictionary mapping
- Alternative readings and epigraphist notes
- A one-click "Download full report as JSON" button

### Command line

```bash
python main.py samples/brahmi_stone.jpg \
    --perspective \
    --save-debug out/ \
    --script-hint brahmi \
    --backend auto
```

Or programmatically:

```python
from main import InscriptionTranslator

pipe = InscriptionTranslator(llm_backend="auto")   # or "gemini" / "groq" / "local" / "stub"
report = pipe.process("samples/brahmi_stone.jpg",
                      apply_perspective=True,
                      script_hint="brahmi",
                      debug_dir="out/")

print(report.pretty())
print(report.as_dict())     # JSON-serialisable
```

A `PipelineReport` carries:

```
source_image, extracted_text, mapped_text, cleaned_text,
english, hindi, alternatives, notes,
ocr_confidence, llm_confidence, overall_confidence
```

---

## 4. How each stage works

### 4.1 Preprocessing (`preprocessing.py`)
Inscriptions are hard to OCR because the glyphs sit on uneven stone under
uneven light. The pipeline therefore does, in order:

1. **Grayscale** — OCR engines are luminance-driven; colour doesn't help.
2. **Denoise** — Gaussian blur for sensor noise + median blur for
   salt-and-pepper artifacts from weathering.
3. **CLAHE** — contrast-limited adaptive histogram equalization. This is the
   single most impactful step for shadowed temple walls.
4. **Adaptive thresholding** — Gaussian-weighted per-block binarization,
   which is far more robust than a global threshold on curved stone.
5. **Morphological close → open** — closes hairline breaks in eroded strokes
   and then strips stray specks.
6. **Canny edges** — kept as an auxiliary channel for panel detection.
7. **(Optional) perspective correction** — detects the largest
   quadrilateral in the edge map and warps it to a frontal rectangle.

Enable `return_intermediates=True` to dump every stage to disk — invaluable
when tuning parameters for a new site.

### 4.2 OCR (`ocr_engine.py`)
No single engine wins on every inscription, so we run an ensemble:

- **Tesseract** with `san+tam+hin+eng` (PSM 6 — single uniform block).
- **EasyOCR** with `['en', 'hi', 'ta']` — useful when the stone has
  semi-modernised glyphs (Devanagari inscriptions, Tamil temple plaques).
- **Multi-scale** (1×, 1.5×, 2×) — each scale is OCRed and we keep the run
  with the highest mean confidence.
- Plug-in point for a **TrOCR / fine-tuned ViT** model (see §7 on Brahmi
  datasets).

Every detected token comes back as an `OCRBox` with its bounding box so we
can highlight hits on the original image.

### 4.3 Dataset matching (`dataset_matcher.py`)
`data/script_mappings.json` is a seed dataset of glyph/token → modern-script
mappings. For each OCR token we do:

1. Exact lookup in the dictionary.
2. If no exact hit, fall back to Levenshtein-based fuzzy matching with a
   configurable minimum similarity (0.55 by default).
3. Unmatched tokens are passed through verbatim to the LLM, which can still
   reason about them in context.

**This file is meant to be extended.** The seed set contains the Brahmi
consonant/vowel inventory plus a few high-value multi-character tokens
(`piyadasi`, `dhamma`, `rājan`, `svasti`, `siddham`, etc.). Adding more
tokens — especially phrase-level entries — is the fastest way to improve
output quality.

### 4.4 GenAI translation (`genai_translator.py`)
The LLM does four things the CV/OCR stack cannot:

- **Correct OCR errors** (e.g., ळ misread as ल, 𑀤 misread as 𑀥).
- **Reconstruct** damaged portions and mark them with `⟨…⟩`.
- **Translate** into readable modern English and Hindi — *readability over
  literalism*, per project requirements.
- **List alternatives** when the text is ambiguous and give a
  self-assessed confidence.

The system prompt (in `genai_translator.py`) frames the model as an
epigraphist/Indologist and forces a strict JSON output schema so downstream
code can ingest the result without brittle parsing.

Interchangeable backends with built-in failover:

| Backend | When to use | Dependency |
|---|---|---|
| `GeminiTranslator` | Primary. `gemini-1.5-pro` has strong reasoning + native JSON mode. | `pip install google-generativeai` |
| `GroqTranslator` | Automatic fallback. `llama-3.3-70b-versatile` is fast and free-tier friendly. | `pip install groq` |
| `FailoverTranslator` | Wraps Gemini → Groq (→ Stub). Returned by `make_translator("auto")` when both keys are present. | — |
| `LocalHFTranslator` | Offline, privacy-sensitive sites | `transformers` + IndicTrans2 |
| `StubTranslator` | No network / testing | — |

Failure policy used by `FailoverTranslator`:
a backend is considered failed if it raises an exception OR returns an
empty result. The next backend in the chain is tried immediately; the
final `TranslationOutput` carries a `backend` field so you can see which
one actually produced the result.

---

## 5. Suggested datasets

Mapping / training sources worth pulling into `data/`:

- **Unicode Brahmi block (U+11000–U+1107F)** — authoritative glyph ↔ IAST map.
- **AI4Bharat IndicTrans2** corpora — modern Indic ↔ English parallel text.
- **Indology Corpus / GRETIL** (`gretil.sub.uni-goettingen.de`) — Sanskrit
  and Prakrit primary texts; great for fine-tuning an LLM on epigraphic style.
- **Mahavamsa & Ashokan edicts** — transliterated Brahmi inscriptions (e.g.
  Hultzsch's *Corpus Inscriptionum Indicarum* Vol. I).
- **Tamil-Brahmi inscription corpus** — Iravatham Mahadevan's catalogue
  (published 2003); digitised fragments circulate in academic repositories.
- **ASI (Archaeological Survey of India) Epigraphy reports** — photographs
  + transliterations per site. Excellent for supervised OCR training.
- **Sanskrit Heritage Segmenter** (INRIA) — helpful for segmenting sandhi
  in reconstructed text.
- **Kaggle: Ancient Brahmi Character Recognition** — labelled glyph images
  usable for training a dedicated Brahmi OCR head (CNN / ViT).
- **IIIT-H Sanskrit OCR dataset** — printed Devanagari, a good warm-start
  before fine-tuning on stone-inscription images.

Pattern for extending the seed JSON:

```json
{
  "script": "sanskrit",
  "token": "महाराज",
  "devanagari": "महाराज",
  "iast": "mahārāja",
  "meaning": "great king"
}
```

---

## 6. Output example

```
=== Ancient Inscription Translation Report ===
Source        : samples/brahmi_stone.jpg
Extracted     : 𑀧𑀺𑀬𑀤𑀲𑀺 𑀥𑀁𑀫𑀁 𑀳𑀺𑀢𑀲𑀼𑀔𑀸𑀬
Mapped        : पियदसि धंमं हितसुखाय
Cleaned       : देवानं पियदसि राजा धम्मं हितसुखाय ⟨जनस्स⟩
English       : "King Piyadasi, Beloved of the Gods, (declared) the Dhamma
                for the welfare and happiness of ⟨the people⟩."
Hindi         : "देवताओं के प्रिय राजा पियदसि ने ⟨प्रजा⟩ के कल्याण और सुख के लिए
                धर्म की घोषणा की।"
Alternatives  : ["... for the welfare and happiness of beings"]
Notes         : The last word was partially eroded; reconstructed as 'जनस्स'
                (Prakrit genitive of 'people') based on formulaic parallels
                in the Major Rock Edicts.
OCR conf      : 71.4
LLM conf      : 0.82
Overall conf  : 0.78
```

---

## 7. Improvements for real-world deployment

**CV / OCR layer**
- Replace the generic OCR stack with a **fine-tuned TrOCR or Donut** model
  on a labelled Brahmi / Tamil-Brahmi glyph dataset. A CNN head trained on
  the Kaggle Brahmi dataset already beats Tesseract on eroded text.
- Add **per-site calibration**: cache per-site lighting/warp parameters so
  subsequent photos from the same temple start from known-good defaults.
- Use **illumination unshadowing** (e.g., Retinex or a lightweight U-Net)
  before binarization — it handles carved inscriptions in deep relief.
- For 3D-scanned inscriptions, **depth-map thresholding** (RTI / PTM data)
  dramatically outperforms RGB-only OCR.

**Dataset layer**
- Grow `script_mappings.json` into a **graph dataset** (glyph → variant
  glyphs → modern forms → semantic concepts) so the matcher can reason
  across regional variants.
- Add an **embedding index** (e.g., FAISS with sentence-transformers) so
  phrase-level fuzzy matching scales beyond the pure-Python Levenshtein
  fallback.
- Cross-reference each mapped token with a **dated regional corpus**
  (e.g., 3rd-century BCE Mauryan vs. 10th-century Chola) to narrow
  translation ambiguity.

**LLM layer**
- **Fine-tune or RAG** with the epigraphy corpus listed in §5 so the model
  is primed with formulaic inscription phrases.
- Add a **self-consistency vote**: run the model 3 times at `temperature=0.3`
  and keep the majority reading.
- Route low-confidence results to a **human-in-the-loop** annotation UI and
  feed corrections back into `data/script_mappings.json`.

**Ops / deployment**
- Wrap the pipeline as a **FastAPI service** with a streaming endpoint so
  mobile clients can upload photos from the field.
- Cache OCR results by image hash — re-processing the same inscription is
  expensive and deterministic.
- Log every `PipelineReport` to a datastore (S3 + Postgres) so archaeologists
  can review disagreements over time.
- Add a **heatmap overlay** in the UI: colour each glyph by its confidence
  so domain experts know where to look first.

---

## 8. Known limitations

- Heavily weathered or sandstone inscriptions may produce empty OCR — there
  is no substitute for a Brahmi-specific trained model.
- EasyOCR has no Brahmi model; the pipeline uses Devanagari/Tamil as the
  nearest recognisable script. Accuracy on pre-Gupta Brahmi is limited
  without a fine-tuned head.
- The seed mapping file is intentionally small — production use requires
  extending it with site-specific vocabulary.
