from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import easyocr
import numpy as np


PLATE_TEXT_PATTERN = re.compile(r"[^A-Z0-9]")
INDIAN_PLATE_TEMPLATES = (
    "LLDDLLDDDD",
    "LLDDLDDDD",
    "LLDDLDDDDD",
    "LLDLLDDDD",
    "LLDDLLLDDDD",
    "DDLLDDDDLL",
)
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}
LETTER_TO_DIGIT = {
    "A": "4",
    "B": "8",
    "D": "0",
    "G": "6",
    "I": "1",
    "L": "1",
    "O": "0",
    "Q": "0",
    "S": "5",
    "T": "1",
    "Z": "2",
}


@dataclass(frozen=True)
class PlateText:
    text: str
    confidence: float


class PlateReader:
    def __init__(
        self,
        languages: list[str] | None = None,
        use_gpu: bool = True,
        fast_mode: bool = False,
    ) -> None:
        self.fast_mode = fast_mode
        self.reader = easyocr.Reader(languages or ["en"], gpu=use_gpu)

    def read(self, plate_crop: np.ndarray) -> PlateText | None:
        enhanced_images = self._enhance_for_ocr(plate_crop)
        best: PlateText | None = None

        for image in enhanced_images:
            results = self.reader.readtext(
                image,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
                paragraph=False,
                decoder="greedy",
                contrast_ths=0.2,
                adjust_contrast=0.7,
            )
            for raw_text, confidence in self._candidate_texts(results):
                normalized = self._normalize(raw_text)
                if len(normalized) < 4:
                    continue
                corrected, correction_score = self._correct_plate_text(normalized)
                confidence = min(float(confidence) + correction_score, 1.0)
                candidate = PlateText(corrected, confidence)
                if best is None or self._rank_plate_text(candidate) > self._rank_plate_text(best):
                    best = candidate
            if best is not None and self._rank_plate_text(best) >= 0.90:
                return best

        return best

    def _enhance_for_ocr(self, crop: np.ndarray) -> list[np.ndarray]:
        resized = self._resize_for_ocr(crop)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(denoised)
        sharpened = cv2.filter2D(
            equalized,
            -1,
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        )
        _, otsu = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )
        inverted = cv2.bitwise_not(adaptive)
        if self.fast_mode:
            return [sharpened, otsu]
        return [sharpened, otsu, adaptive, inverted]

    @staticmethod
    def _resize_for_ocr(crop: np.ndarray) -> np.ndarray:
        height, width = crop.shape[:2]
        if width == 0 or height == 0:
            return crop

        target_width = 480
        target_height = 120
        if width >= target_width and height >= target_height:
            return crop

        scale = max(target_width / width, target_height / height)
        return cv2.resize(
            crop,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return PLATE_TEXT_PATTERN.sub("", text.upper())

    @staticmethod
    def _candidate_texts(results: list[tuple]) -> list[tuple[str, float]]:
        candidates: list[tuple[str, float]] = []
        readable_results = []

        for box, raw_text, confidence in results:
            candidates.append((raw_text, float(confidence)))
            x_position = min(point[0] for point in box)
            y_position = min(point[1] for point in box)
            readable_results.append((y_position, x_position, raw_text, float(confidence)))

        if len(readable_results) > 1:
            readable_results.sort()
            combined_text = "".join(item[2] for item in readable_results)
            average_confidence = sum(item[3] for item in readable_results) / len(readable_results)
            candidates.append((combined_text, average_confidence))

        return candidates

    @staticmethod
    def _correct_plate_text(text: str) -> tuple[str, float]:
        candidates = PlateReader._template_candidates(text)
        deletion_candidates = []
        if 10 <= len(text) <= 11:
            for index in range(5, len(text)):
                shorter_text = text[:index] + text[index + 1 :]
                deletion_candidates.extend(
                    (
                        candidate,
                        score - 0.03 - ((len(text) - index - 1) * 0.012),
                    )
                    for candidate, score in PlateReader._template_candidates(shorter_text)
                )
        candidates.extend(deletion_candidates)
        if candidates:
            return max(candidates, key=lambda item: item[1])

        if 8 <= len(text) <= 11:
            compact = text[:10]
            if len(compact) == 10:
                return PlateReader._apply_template(compact, "LLDDLLDDDD")

        return text, 0.0

    @staticmethod
    def _template_candidates(text: str) -> list[tuple[str, float]]:
        return [
            PlateReader._apply_template(text, template)
            for template in INDIAN_PLATE_TEMPLATES
            if len(text) == len(template)
        ]

    @staticmethod
    def _rank_plate_text(plate_text: PlateText) -> float:
        text = plate_text.text
        pattern_score = max(
            PlateReader._template_fit(text, template)
            for template in INDIAN_PLATE_TEMPLATES
        )
        length_score = 1.0 if 9 <= len(text) <= 10 else 0.5 if 7 <= len(text) <= 11 else 0.0
        starts_like_plate = 1.0 if len(text) >= 4 and text[:2].isalpha() and text[2:4].isdigit() else 0.0
        return (
            plate_text.confidence * 0.45
            + pattern_score * 0.35
            + length_score * 0.12
            + starts_like_plate * 0.08
        )

    @staticmethod
    def _apply_template(text: str, template: str) -> tuple[str, float]:
        corrected = []
        changes = 0
        matches = 0

        for character, expected in zip(text, template):
            if expected == "L":
                if character.isalpha():
                    corrected.append(character)
                    matches += 1
                else:
                    replacement = DIGIT_TO_LETTER.get(character, character)
                    corrected.append(replacement)
                    changes += int(replacement != character)
                    matches += int(replacement.isalpha())
            else:
                if character.isdigit():
                    corrected.append(character)
                    matches += 1
                else:
                    replacement = LETTER_TO_DIGIT.get(character, character)
                    corrected.append(replacement)
                    changes += int(replacement != character)
                    matches += int(replacement.isdigit())

        pattern_fit = matches / max(len(template), 1)
        change_penalty = changes * 0.015
        preferred_length_bonus = 0.05 if len(template) in (9, 10) else 0.0
        score_adjustment = max(
            (pattern_fit - 0.7) * 0.12 + preferred_length_bonus - change_penalty,
            -0.05,
        )
        return "".join(corrected), score_adjustment

    @staticmethod
    def _template_fit(text: str, template: str) -> float:
        if not text:
            return 0.0

        comparable_length = min(len(text), len(template))
        matches = 0
        for character, expected in zip(text[:comparable_length], template[:comparable_length]):
            if expected == "L":
                matches += int(character.isalpha() or character in DIGIT_TO_LETTER)
            else:
                matches += int(character.isdigit() or character in LETTER_TO_DIGIT)

        length_penalty = abs(len(text) - len(template)) * 0.08
        return max((matches / len(template)) - length_penalty, 0.0)
