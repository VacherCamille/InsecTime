#!/usr/bin/env python3
"""Copie les images vers un dossier final selon les validations humaines."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Renomme les images à partir des identifiants validés."
    )
    parser.add_argument("--images-dir", required=True, type=Path,
                        help="Dossier contenant les images stacked.")
    parser.add_argument("--validated-csv", required=True, type=Path,
                        help="CSV produit par l'interface de validation.")
    parser.add_argument("--config", required=True, type=Path,
                        help="validation_config.json associé au jeu de photos.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Dossier de destination.")
    parser.add_argument("--copy", action="store_true", default=True,
                        help="Copie les fichiers (comportement par défaut).")
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    validated_csv = args.validated_csv.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()

    if not images_dir.is_dir():
        parser.error(f"Dossier d'images introuvable : {images_dir}")
    if not validated_csv.is_file():
        parser.error(f"CSV de validation introuvable : {validated_csv}")
    if not config_path.is_file():
        parser.error(f"Configuration introuvable : {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    pattern = config.get("identifier_format", {}).get("regex")
    if not pattern:
        parser.error("La configuration ne contient pas d'expression régulière.")
    try:
        identifier_regex = re.compile(pattern)
    except re.error as exc:
        parser.error(f"Expression régulière invalide : {exc}")

    rows = read_rows(validated_csv)
    if not rows:
        parser.error("Le CSV de validation est vide.")

    errors = []
    planned = []
    seen_identifiers = {}

    for row in rows:
        filename = (row.get("filename") or "").strip()
        label = (row.get("validated_label") or "").strip()
        status = (row.get("status") or "pending").strip().lower()

        if not filename:
            errors.append("Une ligne ne contient pas de nom de fichier.")
            continue

        # Empêche les chemins relatifs inattendus dans le CSV.
        if Path(filename).name != filename:
            errors.append(f"Nom de fichier non autorisé : {filename}")
            continue

        source = images_dir / filename
        if not source.is_file():
            errors.append(f"Image source introuvable : {source}")
            continue

        if status == "pending":
            errors.append(f"Validation encore en attente : {filename}")
        elif status == "validated":
            if not re.fullmatch(identifier_regex, label):
                errors.append(
                    f"Identifiant absent ou non conforme pour {filename} : {label!r}"
                )
                continue
            if label in seen_identifiers:
                errors.append(
                    f"Identifiant dupliqué {label!r} pour "
                    f"{seen_identifiers[label]} et {filename}"
                )
                continue
            seen_identifiers[label] = filename
            destination = output_dir / f"{label}{source.suffix.lower()}"
            planned.append((source, destination))
        elif status in {"unidentified", "ambiguous"}:
            subdir = "unidentified" if status == "unidentified" else "ambiguous"
            destination = output_dir / subdir / source.name
            planned.append((source, destination))
        else:
            errors.append(f"Statut inconnu pour {filename} : {status!r}")

    if errors:
        print("Renommage annulé. Corrigez les problèmes suivants :", file=sys.stderr)
        for error in errors:
            print(f" - {error}", file=sys.stderr)
        return 2

    # Pré-vérification : aucune sortie existante ne sera écrasée.
    existing = [dst for _, dst in planned if dst.exists()]
    if existing:
        print("Renommage annulé : certaines destinations existent déjà :", file=sys.stderr)
        for path in existing:
            print(f" - {path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"{source.name} -> {destination.relative_to(output_dir)}")

    print(f"\n{len(planned)} image(s) copiée(s) dans {output_dir}")
    print("Les images originales du dossier stacked n'ont pas été modifiées.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
