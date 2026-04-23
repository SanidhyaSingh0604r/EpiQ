"""
main.py
-------
End-to-end pipeline for translating ancient temple inscriptions.

Usage (CLI):

    python main.py path/to/inscription.jpg \
        --perspective \
        --save-debug out/ \
        --backend auto

Programmatic:

    from main import InscriptionTranslator
    t = InscriptionTranslator()
    report = t.process("samples/my_stone.jpg")
    print(report.pretty())
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

import cv2

import preprocessing
import ocr_engine
import dataset_matcher
import genai_translator
import utils

log = logging.getLogger("inscription-translator")


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class InscriptionTranslator:
    def __init__(
        self,
        data_dir: Optional[str] = None,
        llm_backend: str = "auto",
        tesseract_langs: str = "eng+san+tam+hin",
        easyocr_langs=None,
    ):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.mapping = dataset_matcher.load_default_dataset(self.data_dir)
        self.translator = genai_translator.make_translator(llm_backend)
        self.tesseract_langs = tesseract_langs
        self.easyocr_langs = easyocr_langs

    # ---- Pipeline stages exposed individually so they can be swapped ----

    def preprocess(self, img, apply_perspective: bool = False, debug_dir: Optional[str] = None):
        stages = preprocessing.preprocess_inscription(
            img, apply_perspective=apply_perspective, return_intermediates=True
        )
        if debug_dir:
            for name, arr in stages.items():
                utils.save_image(os.path.join(debug_dir, f"{name}.png"), arr)
        return stages["cleaned"], stages

    def ocr(self, binary_img):
        return ocr_engine.extract_text(
            binary_img,
            tesseract_langs=self.tesseract_langs,
            easyocr_langs=self.easyocr_langs,
        )

    def dictionary_map(self, raw_text: str):
        tokens = raw_text.split()
        mapped, details = self.mapping.translate_tokens(tokens)
        return mapped, details

    def translate(self, raw_text: str, mapped_text: str, script_hint: Optional[str] = None):
        return self.translator.translate(raw_text, mapped_text, script_hint=script_hint)

    # ---- Orchestration ----

    def process(
        self,
        image_path: str,
        apply_perspective: bool = False,
        script_hint: Optional[str] = None,
        debug_dir: Optional[str] = None,
    ) -> utils.PipelineReport:
        log.info("Loading %s", image_path)
        img = utils.load_image(image_path)

        log.info("Preprocessing …")
        cleaned, stages = self.preprocess(img, apply_perspective=apply_perspective, debug_dir=debug_dir)

        log.info("Running OCR ensemble …")
        ocr_result = self.ocr(cleaned)

        log.info("Dictionary mapping …")
        mapped_text, _details = self.dictionary_map(ocr_result.raw_text)

        log.info("LLM clean-up + translation …")
        llm_out = self.translate(ocr_result.raw_text, mapped_text, script_hint=script_hint)

        # Optional: render detected regions on top of the original image.
        if debug_dir:
            vis = ocr_engine.draw_boxes(img, ocr_result)
            utils.save_image(os.path.join(debug_dir, "detected_regions.png"), vis)

        return utils.PipelineReport(
            source_image=image_path,
            extracted_text=ocr_result.raw_text,
            mapped_text=mapped_text,
            cleaned_text=llm_out.cleaned_text,
            english=llm_out.english,
            hindi=llm_out.hindi,
            alternatives=llm_out.alternatives,
            notes=llm_out.notes,
            ocr_confidence=ocr_result.mean_confidence,
            llm_confidence=llm_out.confidence,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Translate ancient temple inscriptions from an image.")
    p.add_argument("image", help="Path to the inscription image.")
    p.add_argument("--perspective", action="store_true",
                   help="Try to detect and warp the inscription panel.")
    p.add_argument("--script-hint", default=None,
                   help="Optional hint: brahmi | tamil-brahmi | sanskrit | prakrit | pali ...")
    p.add_argument("--save-debug", default=None,
                   help="Directory to save intermediate preprocessing images.")
    p.add_argument("--backend", default="auto",
                   help="LLM backend: auto (Gemini→Groq failover) | gemini | groq | local | stub")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of pretty text.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    t = InscriptionTranslator(llm_backend=args.backend)
    report = t.process(
        args.image,
        apply_perspective=args.perspective,
        script_hint=args.script_hint,
        debug_dir=args.save_debug,
    )

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.pretty())


if __name__ == "__main__":
    main()
