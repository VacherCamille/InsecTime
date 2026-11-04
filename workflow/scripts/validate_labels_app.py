#!/usr/bin/env python3
"""Interface Streamlit de validation humaine des identifiants OCR.

Lancer depuis la racine du projet :
streamlit run workflow/scripts/validate_labels_app.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OCR_ROOT = PROJECT_ROOT / "data" / "intermediate" / "ocr"
VALIDATION_ROOT = PROJECT_ROOT / "results" / "validation"


def discover_label_files() -> list[Path]:
    if not OCR_ROOT.exists():
        return []
    return sorted(OCR_ROOT.rglob("labels.csv"))


def infer_images_dir(labels_path: Path) -> Path:
    """Déduit le dossier stacked à partir du chemin du dossier OCR."""
    try:
        relative = labels_path.parent.relative_to(OCR_ROOT)
        return PROJECT_ROOT / "data" / "intermediate" / "stacked" / relative
    except ValueError:
        return labels_path.parent


def infer_validation_dir(labels_path: Path) -> Path:
    relative = labels_path.parent.relative_to(OCR_ROOT)
    return VALIDATION_ROOT / relative


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = ["filename", "ocr_text", "validated_label", "status"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def config_path_for(labels_path: Path) -> Path:
    return infer_validation_dir(labels_path) / "validation_config.json"


def validated_csv_for(labels_path: Path) -> Path:
    return infer_validation_dir(labels_path) / "validated_labels.csv"


def load_config(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def make_pattern(mode: str, prefix: str, digits: int, custom: str) -> str:
    if mode == "Préfixe + nombre de chiffres":
        if not prefix:
            raise ValueError("Le préfixe ne peut pas être vide.")
        if digits < 1:
            raise ValueError("Le nombre de chiffres doit être supérieur à zéro.")
        return rf"^{re.escape(prefix)}\d{{{digits}}}$"
    re.compile(custom)
    return custom


def extract_suggestion(ocr_text: str, pattern: str) -> str:
    try:
        match = re.search(pattern, ocr_text or "")
    except re.error:
        return ""
    return match.group(0) if match else ""


def load_working_rows(labels_path: Path, validated_path: Path) -> list[dict]:
    source_rows = read_csv(labels_path)
    existing = {}
    if validated_path.is_file():
        existing = {
            row.get("filename", ""): row
            for row in read_csv(validated_path)
        }

    result = []
    for source in source_rows:
        filename = source.get("filename", "")
        previous = existing.get(filename, {})
        result.append(
            {
                "filename": filename,
                "ocr_text": source.get("ocr_text", source.get("ocr_label", "")),
                "validated_label": previous.get("validated_label", ""),
                "status": previous.get("status", "pending") or "pending",
            }
        )
    return result


st.set_page_config(page_title="Validation des identifiants OCR", layout="wide")
st.title("Validation humaine des identifiants OCR")
st.caption(
    "Choisissez un jeu de photos, définissez son format d'identifiant, "
    "puis vérifiez chaque proposition OCR."
)

label_files = discover_label_files()
if not label_files:
    st.error(
        f"Aucun fichier labels.csv trouvé sous : {OCR_ROOT}\n\n"
        "Exécutez d'abord le script OCR pour produire ce fichier."
    )
    st.stop()

selected_label_file = st.selectbox(
    "Jeu de photos (fichier labels.csv)",
    label_files,
    format_func=lambda path: str(path.relative_to(PROJECT_ROOT))
    if path.is_relative_to(PROJECT_ROOT) else str(path),
)
labels_path = Path(selected_label_file)
images_dir_default = infer_images_dir(labels_path)
images_dir_text = st.text_input(
    "Dossier des images stacked",
    value=str(images_dir_default),
)
images_dir = Path(images_dir_text).expanduser()

config_path = config_path_for(labels_path)
validated_path = validated_csv_for(labels_path)
saved_config = load_config(config_path)
saved_format = saved_config.get("identifier_format", {})

st.subheader("1. Format attendu pour ce jeu de photos")
format_mode_default = saved_format.get("mode", "Préfixe + nombre de chiffres")
mode_options = ["Préfixe + nombre de chiffres", "Expression régulière personnalisée"]
mode_index = mode_options.index(format_mode_default) if format_mode_default in mode_options else 0
mode = st.radio(
    "Définition du format",
    mode_options,
    index=mode_index,
    horizontal=True,
)

if mode == "Préfixe + nombre de chiffres":
    col1, col2 = st.columns(2)
    with col1:
        prefix = st.text_input(
            "Préfixe",
            value=saved_format.get("prefix", "GBIFCH"),
        )
    with col2:
        digits = st.number_input(
            "Nombre de chiffres après le préfixe",
            min_value=1,
            max_value=40,
            value=int(saved_format.get("number_of_digits", 8)),
            step=1,
        )
    custom_pattern = ""
else:
    prefix = ""
    digits = 0
    custom_pattern = st.text_input(
        "Expression régulière",
        value=saved_format.get("regex", r"^GBIFCH\d{8}$"),
        help=r"Exemple : ^GBIFCH\d{8}$",
    )

try:
    expected_pattern = make_pattern(mode, prefix, int(digits), custom_pattern)
    re.compile(expected_pattern)
    st.success(f"Format utilisé : `{expected_pattern}`")
except (ValueError, re.error) as exc:
    st.error(f"Format invalide : {exc}")
    st.stop()

if not images_dir.is_dir():
    st.error(f"Le dossier d'images n'existe pas : {images_dir}")
    st.stop()

rows = load_working_rows(labels_path, validated_path)
if not rows:
    st.warning("Le fichier OCR ne contient aucune image.")
    st.stop()

# Enregistre le format choisi pour conserver la règle de validation de ce lot.
config = {
    "labels_file": str(labels_path),
    "images_dir": str(images_dir.resolve()),
    "identifier_format": {
        "mode": mode,
        "prefix": prefix if mode == mode_options[0] else None,
        "number_of_digits": int(digits) if mode == mode_options[0] else None,
        "regex": expected_pattern,
    },
}
if st.button("Enregistrer le format pour ce jeu de photos"):
    save_config(config_path, config)
    st.success(f"Configuration enregistrée : {config_path}")

# Les modifications sont conservées dans le CSV après chaque validation.
if "selected_image_index" not in st.session_state:
    st.session_state.selected_image_index = 0

index = min(st.session_state.selected_image_index, len(rows) - 1)
row = rows[index]

validated_count = sum(r["status"] == "validated" for r in rows)
unidentified_count = sum(r["status"] == "unidentified" for r in rows)
ambiguous_count = sum(r["status"] == "ambiguous" for r in rows)
pending_count = len(rows) - validated_count - unidentified_count - ambiguous_count

st.subheader("2. Vérification des images")
st.write(
    f"Image **{index + 1}/{len(rows)}** — "
    f"Validées : **{validated_count}** · "
    f"Non identifiables : **{unidentified_count}** · "
    f"Ambiguës : **{ambiguous_count}** · "
    f"En attente : **{pending_count}**"
)
st.progress((index + 1) / len(rows))

image_path = images_dir / row["filename"]
left, right = st.columns([1.5, 1])

with left:
    if image_path.is_file():
        try:
            with Image.open(image_path) as img:
                st.image(img.copy(), caption=row["filename"], use_container_width=True)
        except Exception as exc:
            st.error(f"Impossible d'afficher l'image : {exc}")
    else:
        st.error(f"Image introuvable : {image_path}")

with right:
    st.markdown("**Texte détecté par OCR**")
    st.code(row["ocr_text"] or "(aucun texte détecté)")
    suggestion = extract_suggestion(row["ocr_text"], expected_pattern)
    st.write(f"Suggestion correspondant au format : **{suggestion or 'Aucune'}**")

    current_label = row["validated_label"] or suggestion
    label_value = st.text_input(
        "Identifiant confirmé ou corrigé",
        value=current_label,
        key=f"label_{labels_path}_{index}",
    )
    status_options = ["pending", "validated", "unidentified", "ambiguous"]
    current_status = row["status"] if row["status"] in status_options else "pending"
    status_value = st.selectbox(
        "Statut",
        status_options,
        index=status_options.index(current_status),
        key=f"status_{labels_path}_{index}",
        help="validated = identifiant confirmé ; unidentified = illisible ; "
             "ambiguous = plusieurs possibilités.",
    )

    if st.button("Enregistrer cette image", type="primary"):
        label_value = label_value.strip()
        if status_value == "validated":
            if not re.fullmatch(expected_pattern, label_value):
                st.error("L'identifiant ne respecte pas le format attendu.")
                st.stop()
        elif status_value in {"unidentified", "ambiguous"}:
            label_value = ""

        rows[index]["validated_label"] = label_value
        rows[index]["status"] = status_value
        write_csv_atomic(validated_path, rows)
        save_config(config_path, config)
        st.success("Validation enregistrée.")
        st.rerun()

nav1, nav2, nav3 = st.columns(3)
with nav1:
    if st.button("← Précédente", disabled=index == 0):
        st.session_state.selected_image_index -= 1
        st.rerun()
with nav2:
    if st.button("Suivante →", disabled=index >= len(rows) - 1):
        st.session_state.selected_image_index += 1
        st.rerun()
with nav3:
    if st.button("Aller à la première en attente"):
        pending_indices = [i for i, r in enumerate(rows) if r["status"] == "pending"]
        if pending_indices:
            st.session_state.selected_image_index = pending_indices[0]
            st.rerun()
        else:
            st.info("Aucune image en attente.")

st.subheader("3. État de la validation")
if validated_path.is_file():
    st.write(f"Fichier de validation : `{validated_path}`")
    st.download_button(
        "Télécharger validated_labels.csv",
        data=validated_path.read_bytes(),
        file_name="validated_labels.csv",
        mime="text/csv",
    )
else:
    st.info("Le CSV de validation sera créé au premier enregistrement.")

st.caption(
    "Le renommage ne sera autorisé que lorsque toutes les images auront un statut "
    "défini et que les identifiants validés respecteront le format attendu."
)
