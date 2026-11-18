#!/usr/bin/env python3

"""
extract_labels.py

Extraction OCR des étiquettes présentes sur les images empilées.
Extrait les identifiants respectant le format attendu
    pour chaque musée.

Entrées :
    - Un dossier contenant les images sélectionnées.
    - Un modèle OpenCV DB de détection de texte.

Sortie :
    - Un fichier CSV contenant le texte OCR de chaque image.

Les images sources ne sont jamais modifiées.
"""

import argparse
import csv
import logging
import re
import sys

from pathlib import Path

import cv2
import numpy as np
import pytesseract



SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff"
}

def extract_identifiers(ocr_text, museum_id):
    """
    Extrait les identifiants respectant le format attendu
    pour chaque musée.
    """

    patterns = {
        "MHNGR": r"\bMHNGR_[A-Z]+_[0-9]{1,3}\b",

        # Ajouter ici les formats des autres musées.
    }

    if museum_id not in patterns:
        raise ValueError(
            f"Aucun format d'identifiant défini pour {museum_id}"
        )

    pattern = patterns[museum_id]

    identifiers = re.findall(pattern, ocr_text)

    # Supprime les doublons en conservant l'ordre.
    return list(dict.fromkeys(identifiers))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extraction OCR des étiquettes d'insectes."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Dossier contenant les images stacked."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Chemin vers le modèle OpenCV DB (.onnx)."
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help="Chemin du fichier CSV de sortie."
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"],
        help="Extensions des images à traiter."
    )

    parser.add_argument(
        "--lang",
        default="eng",
        help="Langue Tesseract (par défaut : eng)."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre d'images pour un test."
    )

    parser.add_argument(
    "--museum",
    required=True,
    help="Identifiant du musée."
)

    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_model(model_path):
    """Charge le modèle de détection de texte OpenCV DB."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    if not hasattr(cv2, "dnn_TextDetectionModel_DB"):
        raise RuntimeError(
            "La fonction cv2.dnn_TextDetectionModel_DB est indisponible. "
            "Installe opencv-contrib-python."
        )

    logging.info("Chargement du modèle : %s", model_path)

    model = cv2.dnn_TextDetectionModel_DB(str(model_path))

    model.setBinaryThreshold(0.3)
    model.setPolygonThreshold(0.5)
    model.setMaxCandidates(200)
    model.setUnclipRatio(2.0)

    model.setInputParams(
        scale=1.0 / 255.0,
        size=(1024, 768),
        mean=(122.6789, 116.6688, 104.0070),
        swapRB=True
    )

    return model


def order_points(points):
    """Ordonne les quatre points d'un rectangle."""

    points = np.asarray(points, dtype="float32")

    ordered = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).flatten()

    ordered[0] = points[np.argmin(sums)]        # Haut gauche
    ordered[2] = points[np.argmax(sums)]        # Bas droite
    ordered[1] = points[np.argmin(differences)] # Haut droite
    ordered[3] = points[np.argmax(differences)] # Bas gauche

    return ordered


def crop_text_region(image, box):
    """Redresse une zone de texte détectée."""

    points = order_points(box)

    top_left, top_right, bottom_right, bottom_left = points

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)

    width = max(int(max(width_a, width_b)), 1)
    height = max(int(max(height_a, height_b)), 1)

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        points,
        destination
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )


def clean_ocr_text(text):
    """Nettoie les espaces et retours à la ligne."""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_image(image, model, language):
    """
    Détecte les zones de texte puis applique Tesseract.

    Si aucune zone n'est détectée, applique Tesseract
    directement sur l'image entière.
    """

    boxes, confidences = model.detect(image)

    text_regions = []

    if boxes is not None and len(boxes) > 0:
        # Lecture des zones de haut en bas,
        # puis de gauche à droite.
        boxes = sorted(
            boxes,
            key=lambda box: (
                np.mean(np.asarray(box)[:, 1]),
                np.mean(np.asarray(box)[:, 0])
            )
        )

        for box in boxes:
            crop = crop_text_region(image, box)

            if crop.size == 0:
                continue

            gray = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY
            )

            # Agrandissement pour faciliter la lecture OCR.
            gray = cv2.resize(
                gray,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )

            text = pytesseract.image_to_string(
                gray,
                lang=language,
                config="--psm 6"
            )

            text = clean_ocr_text(text)

            if text:
                text_regions.append(text)

    else:
        # Repli : OCR sur l'image entière.
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        text = pytesseract.image_to_string(
            gray,
            lang=language,
            config="--psm 6"
        )

        text = clean_ocr_text(text)

        if text:
            text_regions.append(text)

    return " | ".join(text_regions)


def read_image(image_path):
    """Lecture compatible avec les chemins contenant des caractères spéciaux."""

    data = np.fromfile(str(image_path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    args = parse_arguments()
    configure_logging()

    input_dir = Path(args.input_dir)
    model_path = Path(args.model)
    output_csv = Path(args.output_csv)

    if not input_dir.is_dir():
        logging.error(
            "Dossier d'entrée introuvable : %s",
            input_dir
        )
        return 1

    extensions = {
        extension.lower()
        if extension.startswith(".")
        else "." + extension.lower()
        for extension in args.extensions
    }

    images = sorted(
        path for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    if args.limit is not None:
        if args.limit < 1:
            logging.error("--limit doit être supérieur à zéro.")
            return 1

        images = images[:args.limit]

    if not images:
        logging.warning(
            "Aucune image trouvée dans %s",
            input_dir
        )

    try:
        model = load_model(model_path)

        # Vérifie que Tesseract est accessible.
        pytesseract.get_tesseract_version()

    except Exception as error:
        logging.error("Initialisation impossible : %s", error)
        return 1

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    logging.info(
        "%d image(s) à traiter.",
        len(images)
    )

    for index, image_path in enumerate(images, start=1):
        logging.info(
            "[%d/%d] Traitement de %s",
            index,
            len(images),
            image_path.name
        )

        try:
            image = read_image(image_path)

            if image is None:
                raise ValueError("Impossible de lire l'image.")

            ocr_text = extract_text_from_image(
                image,
                model,
                args.lang
            )

            status = "text_found" if ocr_text else "no_text"

        except Exception as error:
            logging.exception(
                "Erreur pendant le traitement de %s",
                image_path.name
            )

            ocr_text = ""
            status = "error"

        identifiers = extract_identifiers(
            ocr_text,
            args.museum
        )

        if identifiers:
            status = "identifier_found"
        else:
            status = "no_valid_identifier"

        results.append({
            "filename": image_path.name,
            "ocr_text": ocr_text,
            "identifiers": ";".join(identifiers),
            "ocr_status": status
        })

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "filename",
                "ocr_text",
                "identifiers",
                "ocr_status"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    logging.info(
        "Extraction terminée : %s",
        output_csv
    )

    logging.info(
        "Images traitées : %d",
        len(results)
    )

    logging.info(
        "Textes détectés : %d",
        sum(row["ocr_status"] == "text_found" for row in results)
    )

    logging.info(
        "Sans texte détecté : %d",
        sum(row["ocr_status"] == "no_text" for row in results)
    )

    logging.info(
        "Erreurs : %d",
        sum(row["ocr_status"] == "error" for row in results)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())