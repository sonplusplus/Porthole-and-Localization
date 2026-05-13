from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .phase4_landmark_schema import normalize_text


@dataclass
class OcrResult:
    text: str
    confidence: float

    @property
    def text_norm(self) -> Optional[str]:
        return normalize_text(self.text)


class OcrBackend:
    name = "paddleocr"

    def recognize(self, image_bgr: np.ndarray) -> Optional[OcrResult]:
        raise NotImplementedError


class PaddleOcrBackend(OcrBackend):
    name = "paddleocr"

    def __init__(self, lang: str = "vi") -> None:
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        self.lang = lang

    def recognize(self, image_bgr: np.ndarray) -> Optional[OcrResult]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.ocr.ocr(image_rgb, cls=True)
        best_text = ""
        best_score = 0.0
        for page in result or []:
            for item in page or []:
                if len(item) < 2:
                    continue
                text, score = item[1]
                if score > best_score and text:
                    best_text = str(text)
                    best_score = float(score)
        if not best_text:
            return None
        return OcrResult(best_text, best_score)


def create_ocr_backend(lang: str = "vi") -> OcrBackend:
    return PaddleOcrBackend(lang=lang)
