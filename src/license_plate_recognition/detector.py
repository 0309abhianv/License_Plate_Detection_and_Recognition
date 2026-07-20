from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PlateCandidate:
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    score: float


class PlateDetector:
    """Find license-plate-like regions using OpenCV contour analysis."""

    def __init__(
        self,
        min_area: int = 1500,
        max_area: int = 40000,
        min_ratio: float = 2.0,
        max_ratio: float = 6.5,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def detect(self, frame: np.ndarray) -> list[PlateCandidate]:
        preprocessed = self._preprocess(frame)
        contours, _ = cv2.findContours(
            preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: list[PlateCandidate] = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            if not self._valid_rotated_rect(rect):
                continue

            x, y, width, height = cv2.boundingRect(contour)
            x, y, width, height = self._pad_box(frame, x, y, width, height)
            crop = frame[y : y + height, x : x + width]
            if crop.size == 0:
                continue

            score = self._score_candidate(contour, width, height)
            candidates.append(PlateCandidate((x, y, width, height), crop, score))

        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, threshold = cv2.threshold(
            sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, self.close_kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.open_kernel)
        return opened

    def _valid_rotated_rect(self, rect: tuple) -> bool:
        (_, _), (width, height), angle = rect
        if width <= 0 or height <= 0:
            return False

        area = width * height
        ratio = max(width, height) / min(width, height)
        normalized_angle = abs(angle)
        if normalized_angle > 45:
            normalized_angle = 90 - normalized_angle

        return (
            self.min_area <= area <= self.max_area
            and self.min_ratio <= ratio <= self.max_ratio
            and normalized_angle <= 25
        )

    def _score_candidate(self, contour: np.ndarray, width: int, height: int) -> float:
        rect_area = max(width * height, 1)
        contour_area = cv2.contourArea(contour)
        ratio = width / max(height, 1)
        ratio_score = 1.0 - min(abs(ratio - 4.2) / 4.2, 1.0)
        fill_score = min(contour_area / rect_area, 1.0)
        return (ratio_score * 0.65) + (fill_score * 0.35)

    @staticmethod
    def _pad_box(
        frame: np.ndarray, x: int, y: int, width: int, height: int
    ) -> tuple[int, int, int, int]:
        pad_left = int(width * 0.85)
        pad_right = int(width * 0.45)
        pad_y = int(height * 0.35)
        x1 = max(0, x - pad_left)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + width + pad_right)
        y2 = min(frame.shape[0], y + height + pad_y)
        return x1, y1, x2 - x1, y2 - y1
