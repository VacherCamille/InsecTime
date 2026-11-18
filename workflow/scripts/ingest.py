#!/usr/bin/env python3
"""
Safely ingest camera folders into the insect-morphology raw-data store.

Expected incoming folder name:
    Raw_{MUSEUM}_{CAMERA}_{DD-MM-YYYY}

Example:
    Raw_MNHN_MOMO_10-10-2026

Destination:
    data/raw/{MUSEUM}/{YYYY-MM-DD}/{CAMERA}/

The script copies files to a staging directory, verifies SHA-256 checksums,
writes acquisition metadata, atomically publishes the acquisition directory,
and only then removes the incoming folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
FOLDER_PATTERN = re.compile(
    r"^Raw_(?P<museum>[A-Za-z0-9-]+)_(?P<camera>[A-Za-z0-9-]+)_(?P<date>\d{2}-\d{2}-\d{4})$"
)
REGISTRY_FIELDS = [
    "acquisition_id",
    "museum_id",
    "camera_id",
    "acquisition_date",
    "number_of_images",
    "relative_path",
    "imported_at_utc",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_folder_name(folder: Path) -> tuple[str, str, str, str]:
    match = FOLDER_PATTERN.fullmatch(folder.name)
    if not match:
        raise ValueError(
            f"Nom de dossier invalide : {folder.name!r}. "
            "Format attendu : Raw_{MUSEUM}_{CAMERA}_{DD-MM-YYYY}"
        )

    museum = match.group("museum")
    camera = match.group("camera")
    date_raw = match.group("date")

    try:
        acquisition_date = datetime.strptime(date_raw, "%d-%m-%Y").date()
    except ValueError as exc:
        raise ValueError(f"Date invalide dans {folder.name!r}: {exc}") from exc

    date_iso = acquisition_date.isoformat()
    acquisition_id = f"{museum}_{camera}_{acquisition_date:%Y%m%d}"
    return museum, camera, date_iso, acquisition_id


def list_images(folder: Path) -> list[Path]:
    return sorted(
        (
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def load_registry(registry_path: Path) -> list[dict[str, str]]:
    if not registry_path.exists():
        return []

    with registry_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            return []
        missing = set(REGISTRY_FIELDS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Le registre {registry_path} ne contient pas les colonnes : "
                f"{', '.join(sorted(missing))}"
            )
        return list(reader)


def write_registry_atomic(
    registry_path: Path,
    rows: list[dict[str, str]],
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=registry_path.parent,
        prefix=f".{registry_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        writer = csv.DictWriter(
            stream,
            fieldnames=REGISTRY_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(registry_path)


def ingest_one(
    source: Path,
    raw_root: Path,
    registry_path: Path,
    dry_run: bool,
) -> str:
    museum, camera, date_iso, acquisition_id = parse_folder_name(source)

    if not source.is_dir():
        raise ValueError(f"Le chemin source n'est pas un dossier : {source}")

    images = list_images(source)
    if not images:
        raise ValueError(f"Aucune photographie reconnue dans {source}")

    # Refuse unexpected subdirectories: they should not be silently discarded.
    subdirectories = [p.name for p in source.iterdir() if p.is_dir()]
    if subdirectories:
        raise ValueError(
            f"Sous-dossiers inattendus dans {source}: {', '.join(subdirectories)}"
        )

    destination = raw_root / museum / date_iso / camera
    relative_destination = destination.relative_to(raw_root)

    registry_rows = load_registry(registry_path)
    if any(row["acquisition_id"] == acquisition_id for row in registry_rows):
        raise ValueError(
            f"L'acquisition {acquisition_id} est déjà présente dans "
            f"le registre {registry_path}"
        )
    if destination.exists():
        raise FileExistsError(
            f"La destination existe déjà : {destination}. "
            "Aucun fichier n'a été modifié ; vérifier s'il s'agit d'une "
            "importation précédente à réconcilier."
        )

    # Calculate source hashes before making any changes.
    image_records: list[dict[str, Any]] = []
    for image in images:
        image_records.append(
            {
                "filename": image.name,
                "extension": image.suffix.lower(),
                "size_bytes": image.stat().st_size,
                "sha256": sha256_file(image),
            }
        )

    imported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if dry_run:
        logging.info(
            "[SIMULATION] %s : %d image(s) -> %s",
            acquisition_id,
            len(images),
            destination,
        )
        return acquisition_id

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Staging directory is created alongside destination so final rename stays
    # on the same filesystem and is atomic on normal local filesystems.
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{camera}_{date_iso}_ingest_",
            dir=destination.parent,
        )
    )

    published = False
    try:
        # Copy to staging first; source files remain untouched until validation.
        for record in image_records:
            source_file = source / record["filename"]
            staged_file = staging / record["filename"]
            shutil.copy2(source_file, staged_file)

            if staged_file.stat().st_size != record["size_bytes"]:
                raise IOError(f"Taille différente après copie : {record['filename']}")
            if sha256_file(staged_file) != record["sha256"]:
                raise IOError(f"Checksum différent après copie : {record['filename']}")

        manifest = {
            "acquisition_id": acquisition_id,
            "museum_id": museum,
            "camera_id": camera,
            "acquisition_date": date_iso,
            "source_folder_name": source.name,
            "number_of_images": len(image_records),
            "imported_at_utc": imported_at,
            "checksum_algorithm": "sha256",
            "status": "validated",
        }

        with (staging / "acquisition.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        with (staging / "inventory.tsv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["filename", "extension", "size_bytes", "sha256"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(image_records)

        # Recheck destination immediately before publishing.
        if destination.exists():
            raise FileExistsError(f"La destination est apparue entre-temps : {destination}")

        staging.rename(destination)
        published = True

        # Update registry after the validated acquisition is published.
        registry_rows.append(
            {
                "acquisition_id": acquisition_id,
                "museum_id": museum,
                "camera_id": camera,
                "acquisition_date": date_iso,
                "number_of_images": str(len(image_records)),
                "relative_path": relative_destination.as_posix(),
                "imported_at_utc": imported_at,
            }
        )
        write_registry_atomic(registry_path, registry_rows)

        logging.info(
            "Importation réussie : %s (%d image(s))",
            acquisition_id,
            len(image_records),
        )
        return acquisition_id

    except Exception:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if published:
            logging.exception(
                "L'acquisition a été publiée dans %s, mais une étape finale "
                "a échoué. Le dossier source a été conservé si sa suppression "
                "n'avait pas encore eu lieu. Vérifier le registre avant de "
                "relancer.",
                destination,
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Importe les dossiers Raw_* vers le stockage raw du projet."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Racine du projet (défaut : répertoire courant).",
    )
    parser.add_argument(
        "--incoming",
        type=Path,
        default=Path("data/incoming"),
        help="Dossier des acquisitions déposées manuellement.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw"),
        help="Dossier de stockage des données brutes.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("config/acquisitions.tsv"),
        help="Chemin du registre global des acquisitions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simule les importations sans déplacer ni modifier les fichiers.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Ne traiter qu'un dossier entrant précis (nom du dossier, sans chemin).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    project_root = args.project_root.resolve()
    incoming_root = (project_root / args.incoming).resolve()
    raw_root = (project_root / args.raw).resolve()
    registry_path = (project_root / args.registry).resolve()

    if not incoming_root.is_dir():
        logging.error("Dossier incoming introuvable : %s", incoming_root)
        return 2

    if args.folder:
        candidates = [incoming_root / args.folder]
    else:
        candidates = sorted(
            (path for path in incoming_root.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )

    if not candidates:
        logging.info("Aucun dossier d'acquisition à importer.")
        return 0

    failures = 0
    successes = 0

    for candidate in candidates:
        try:
            ingest_one(
                source=candidate,
                raw_root=raw_root,
                registry_path=registry_path,
                dry_run=args.dry_run,
            )
            successes += 1
        except Exception as exc:
            failures += 1
            logging.error("%s : %s", candidate.name, exc)

    logging.info(
        "Terminé : %d réussite(s), %d erreur(s).",
        successes,
        failures,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
