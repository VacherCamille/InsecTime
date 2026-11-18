
#!/usr/bin/env python3

"""
extract_labels.py

Détection des étiquettes et extraction OCR des identifiants.
Le format des identifiants est défini dans un fichier JSON.

Les images sources ne sont jamais modifiées.
"""

import argparse
import csv
import json
import logging
import re
import sys

from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff"
}

# Caractères autorisés pour la reconnaissance des identifiants.
OCR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Détection OCR des étiquettes d'insectes."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Dossier contenant les images stacked."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Chemin vers le modèle ONNX DB."
    )

    parser.add_argument(
        "--identifier-config",
        required=True,
        help="Fichier JSON contenant le format des identifiants."
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help="Chemin du CSV de sortie."
    )

    parser.add_argument(
        "--lang",
        default="eng",
        help="Langue Tesseract."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre maximal d'images à traiter."
    )

    return parser.parse_args()


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

def configure_logging():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# --------------------------------------------------
# CONFIGURATION DES IDENTIFIANTS
# --------------------------------------------------

def load_identifier_pattern(config_path):

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration introuvable : {config_path}"
        )

    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    if "pattern" not in config:
        raise ValueError(
            "Le fichier JSON ne contient pas de champ 'pattern'."
        )

    return re.compile(config["pattern"])


# --------------------------------------------------
# REDIMENSIONNEMENT
# --------------------------------------------------

def resize_fixed(
    image,
    target_width=1024,
    target_height=768
):
    """
    Redimensionne l'image dans une limite maximale
    de 1024 x 768 pixels en conservant les proportions.
    """

    height, width = image.shape[:2]

    scale = min(
        target_width / width,
        target_height / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# --------------------------------------------------
# CHARGEMENT DU MODÈLE DB
# --------------------------------------------------

def load_model(model_path):

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    if not hasattr(cv2, "dnn_TextDetectionModel_DB"):
        raise RuntimeError(
            "cv2.dnn_TextDetectionModel_DB est indisponible. "
            "Vérifie ton installation OpenCV."
        )

    logging.info("Chargement du modèle DB : %s", model_path)

    model = cv2.dnn_TextDetectionModel_DB(str(model_path))

    # Paramètres identiques au script de référence.
    bin_thresh = 0.3
    poly_thresh = 0.5

    mean = (
        122.67891434,
        116.66876762,
        104.00698793
    )

    model.setBinaryThreshold(bin_thresh)
    model.setPolygonThreshold(poly_thresh)

    return model, mean


# --------------------------------------------------
# ORDRE DES POINTS
# --------------------------------------------------

def order_quad(points):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1)

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(differences)]
    bottom_left = points[np.argmax(differences)]

    return np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ],
        dtype=np.float32
    )


# --------------------------------------------------
# AGRANDISSEMENT DES ZONES DÉTECTÉES
# --------------------------------------------------

def expand_box(
    bbox,
    margin=10,
    margin_side=10
):

    points = order_quad(bbox)

    return np.array(
        [
            [
                points[0, 0] - margin - margin_side,
                points[0, 1] - margin
            ],
            [
                points[1, 0] + margin + margin_side,
                points[1, 1] - margin
            ],
            [
                points[2, 0] + margin + margin_side,
                points[2, 1] + margin
            ],
            [
                points[3, 0] - margin - margin_side,
                points[3, 1] + margin
            ]
        ],
        dtype=np.float32
    )


# --------------------------------------------------
# REDRESSEMENT HAUTE RÉSOLUTION
# --------------------------------------------------

def warp_box_high_res(image, vertices):

    points = order_quad(vertices)

    top_left, top_right, bottom_right, bottom_left = points

    width_a = np.linalg.norm(
        bottom_right - bottom_left
    )

    width_b = np.linalg.norm(
        top_right - top_left
    )

    height_a = np.linalg.norm(
        top_right - bottom_right
    )

    height_b = np.linalg.norm(
        top_left - bottom_left
    )

    max_width = max(
        int(max(width_a, width_b)),
        1
    )

    max_height = max(
        int(max(height_a, height_b)),
        1
    )

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        points,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_NEAREST
    )

    return warped


# --------------------------------------------------
# OCR D'UNE ZONE DE TEXTE
# --------------------------------------------------

def recognize_text(crop, language):

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(crop_rgb)

    gray = pil_image.convert("L")

    # Paramètres OCR du script de référence,
    # avec une whitelist élargie.
    config = (
        "--psm 7 "
        f"-c tessedit_char_whitelist={OCR_WHITELIST}"
    )

    raw = pytesseract.image_to_string(
        gray,
        lang=language,
        config=config
    )

    # Nettoyage identique au script de référence.
    text = (
        raw.replace("\x0c", "")
        .replace("\n", "")
        .strip()
    )

    text = text.replace(" ", "")

    return text


# --------------------------------------------------
# OCR COMPLET D'UNE IMAGE
# --------------------------------------------------

def extract_text_from_image(image, model, language):

    # Redimensionnement identique au script de référence.
    image = resize_fixed(
        image,
        target_width=1024,
        target_height=768
    )

    # Taille d'entrée du modèle.
    input_size = image.shape[:2]

    # Paramètres du modèle DB.
    mean = (
        122.67891434,
        116.66876762,
        104.00698793
    )

    model.setInputParams(
        1.0 / 255.0,
        input_size,
        mean,
        True
    )

    boxes, _ = model.detect(image)

    all_texts = []

    if boxes is None or len(boxes) == 0:
        return ""

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

        expanded = expand_box(
            box,
            margin=10,
            margin_side=10
        )

        crop = warp_box_high_res(
            image,
            expanded
        )

        if crop.size == 0:
            continue

        text = recognize_text(
            crop,
            language
        )

        if text:
            all_texts.append(text)

    return " | ".join(all_texts)


# --------------------------------------------------
# LECTURE DES IMAGES
# --------------------------------------------------

def read_image(image_path):
    """
    Lecture compatible avec les chemins contenant
    des caractères spéciaux.
    """

    data = np.fromfile(
        str(image_path),
        dtype=np.uint8
    )

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )


# --------------------------------------------------
# EXTRACTION DES IDENTIFIANTS
# --------------------------------------------------

def extract_identifiers(ocr_text, identifier_pattern):

    identifiers = identifier_pattern.findall(ocr_text)

    # Suppression des doublons en conservant l'ordre.
    return list(dict.fromkeys(identifiers))


# --------------------------------------------------
# TRAITEMENT D'UNE IMAGE
# --------------------------------------------------

def process_image(
    image_path,
    model,
    identifier_pattern,
    language
):

    result = {
        "filename": image_path.name,
        "ocr_text": "",
        "identifiers": "",
        "identifier_count": 0,
        "ocr_status": "",
        "identifier_status": ""
    }

    try:

        image = read_image(image_path)

        if image is None:
            raise ValueError(
                "Impossible de lire l'image."
            )

        ocr_text = extract_text_from_image(
            image,
            model,
            language
        )

        result["ocr_text"] = ocr_text

        if ocr_text:
            result["ocr_status"] = "text_found"
        else:
            result["ocr_status"] = "no_text"

        identifiers = extract_identifiers(
            ocr_text,
            identifier_pattern
        )

        result["identifiers"] = ";".join(identifiers)

        result["identifier_count"] = len(identifiers)

        if len(identifiers) == 0:
            result["identifier_status"] = "no_valid_identifier"

        elif len(identifiers) == 1:
            result["identifier_status"] = "one_identifier"

        else:
            result["identifier_status"] = "multiple_identifiers"

    except Exception:

        logging.exception(
            "Erreur pendant le traitement de %s",
            image_path.name
        )

        result["ocr_status"] = "error"
        result["identifier_status"] = "not_processed"

    return result


# --------------------------------------------------
# PROGRAMME PRINCIPAL
# --------------------------------------------------

def main():

    args = parse_arguments()

    configure_logging()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.is_dir():
        logging.error(
            "Dossier d'entrée introuvable : %s",
            input_dir
        )
        return 1

    if args.limit is not None and args.limit < 1:
        logging.error(
            "--limit doit être supérieur à zéro."
        )
        return 1

    # Chargement de la regex.
    try:

        identifier_pattern = load_identifier_pattern(
            args.identifier_config
        )

        logging.info(
            "Regex utilisée : %s",
            identifier_pattern.pattern
        )

    except Exception as error:

        logging.error(
            "Erreur de configuration : %s",
            error
        )
        return 1

    # Recherche des images.
    images = sorted(
        path for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if args.limit is not None:
        images = images[:args.limit]

    if not images:
        logging.warning(
            "Aucune image trouvée dans %s",
            input_dir
        )
        return 0

    # Chargement du modèle.
    try:

        model, _ = load_model(args.model)

        pytesseract.get_tesseract_version()

    except Exception as error:

        logging.error(
            "Initialisation impossible : %s",
            error
        )
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

        result = process_image(
            image_path,
            model,
            identifier_pattern,
            args.lang
        )

        results.append(result)

    # Écriture du CSV.
    fieldnames = [
        "filename",
        "ocr_text",
        "identifiers",
        "identifier_count",
        "ocr_status",
        "identifier_status"
    ]

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # Bilan.
    logging.info(
        "Extraction terminée : %s",
        output_csv
    )

    logging.info(
        "Images traitées : %d",
        len(results)
    )

    logging.info(
        "Identifiants trouvés : %d",
        sum(
            row["identifier_count"] > 0
            for row in results
        )
    )

    logging.info(
        "Images sans identifiant valide : %d",
        sum(
            row["identifier_status"] == "no_valid_identifier"
            for row in results
        )
    )

    logging.info(
        "Erreurs : %d",
        sum(
            row["ocr_status"] == "error"
            for row in results
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())