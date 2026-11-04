#!/usr/bin/env python3
"""Détection de texte sur les images stacked et export des résultats OCR.

Dépendances : opencv-python, numpy, Pillow, pytesseract.
Tesseract doit également être installé sur le système.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    return np.array(
        [
            points[np.argmin(sums)],   # top-left
            points[np.argmin(diffs)],  # top-right
            points[np.argmax(sums)],   # bottom-right
            points[np.argmax(diffs)],  # bottom-left
        ],
        dtype=np.float32,
    )


def expand_quad(points: np.ndarray, margin: int = 10) -> np.ndarray:
    pts = order_quad(points)
    return np.array(
        [
            [pts[0, 0] - margin, pts[0, 1] - margin],
            [pts[1, 0] + margin, pts[1, 1] - margin],
            [pts[2, 0] + margin, pts[2, 1] + margin],
            [pts[3, 0] - margin, pts[3, 1] + margin],
        ],
        dtype=np.float32,
    )


def warp_box(image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    rect = order_quad(vertices)
    tl, tr, br, bl = rect

    width = max(
        int(np.linalg.norm(br - bl)),
        int(np.linalg.norm(tr - tl)),
        1,
    )
    height = max(
        int(np.linalg.norm(tr - br)),
        int(np.linalg.norm(tl - bl)),
        1,
    )

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
    )


def resize_fixed(image: np.ndarray, target_width: int = 1024,
                 target_height: int = 768) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def build_detector(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Modèle ONNX introuvable : {model_path}")

    detector = cv2.dnn_TextDetectionModel_DB(str(model_path))
    detector.setBinaryThreshold(0.3).setPolygonThreshold(0.5)
    detector.setInputParams(
        scale=1.0 / 255,
        size=(1024, 768),
        mean=(122.67891434, 116.66876762, 104.00698793),
        swapRB=True,
    )
    return detector


def extract_text(image_path: Path, detector) -> tuple[str, str]:
    image = cv2.imread(str(image_path))
    if image is None:
        return "", "image_unreadable"

    resized = resize_fixed(image)
    try:
        boxes, _ = detector.detect(resized)
    except cv2.error as exc:
        return "", f"detection_error: {exc}"

    if boxes is None or len(boxes) == 0:
        return "", "no_text_region_detected"

    recognized = []
    # Ordre de lecture approximatif : haut vers bas, puis gauche vers droite.
    boxes = sorted(
        boxes,
        key=lambda box: (
            float(np.mean(np.asarray(box)[:, 1])),
            float(np.mean(np.asarray(box)[:, 0])),
        ),
    )

    for box in boxes:
        try:
            crop = warp_box(resized, expand_quad(box, margin=10))
            if crop.size == 0:
                continue
            gray = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("L")
            config = (
                "--psm 13 "
                "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz0123456789"
            )
            raw = pytesseract.image_to_string(gray, config=config)
            text = re.sub(r"\s+", "", raw.replace("\x0c", ""))
            if text:
                recognized.append(text)
        except (cv2.error, ValueError):
            # Une zone défectueuse ne doit pas empêcher le traitement des autres.
            continue

    # On conserve le texte brut détecté : le format d'identifiant est défini
    # plus tard par l'utilisateur dans l'interface de validation.
    unique_texts = list(dict.fromkeys(recognized))
    return " | ".join(unique_texts), "ok" if unique_texts else "no_text_recognized"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extrait le texte des étiquettes des images stacked."
    )
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Dossier contenant les images stacked.")
    parser.add_argument("--model", required=True, type=Path,
                        help="Chemin du modèle DB_TD500_resnet18.onnx.")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="CSV de sortie (par défaut : <input-dir>/labels.csv).")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"Dossier d'entrée inexistant : {input_dir}")

    output_csv = (args.output_csv or (input_dir / "labels.csv")).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"Aucune image trouvée dans {input_dir}")
        return 0

    detector = build_detector(args.model.resolve())

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] OCR : {image_path.name}", flush=True)
        try:
            ocr_text, ocr_status = extract_text(image_path, detector)
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract est introuvable. Installez Tesseract et vérifiez "
                "qu'il est accessible dans le PATH."
            )
        rows.append(
            {
                "filename": image_path.name,
                "ocr_text": ocr_text,
                "ocr_status": ocr_status,
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filename", "ocr_text", "ocr_status"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Résultats OCR enregistrés dans : {output_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERREUR OCR : {exc}", file=sys.stderr)
        raise SystemExit(1)
