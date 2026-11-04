#!/usr/bin/env python3

"""
Sélection des images focus-stacked à partir des métadonnées EXIF.

Fonctionnement :
1. Parcourt les images du dossier source.
2. Lit les métadonnées avec ExifTool.
3. Sélectionne les images dont StackedImage contient "Focus-stacked".
4. Récupère DateTimeOriginal.
5. Ajoute la date au début du nom de fichier (JJ-MM-AAAA).
6. Copie les images sélectionnées dans le dossier de destination.

Les images originales ne sont jamais modifiées.
"""

import argparse
import json
import shutil
import subprocess
import sys

from datetime import datetime
from pathlib import Path


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Sélectionner les images focus-stacked."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Dossier contenant les images originales."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Dossier de destination des images stacked."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les images sélectionnées sans les copier."
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".JPG"],
        help="Extensions des images à traiter."
    )

    parser.add_argument(
        "--exiftool",
        default="exiftool",
        help="Commande ou chemin vers ExifTool."
    )

    return parser.parse_args()


def get_metadata(images, exiftool):

    """
    Récupère les métadonnées des images avec ExifTool.
    """

    command = [
        exiftool,
        "-json",
        "-SourceFile",
        "-StackedImage",
        "-DateTimeOriginal"
    ]

    command.extend(str(image) for image in images)

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

    except FileNotFoundError:

        raise RuntimeError(
            "ExifTool est introuvable. Vérifie son installation."
        )

    except subprocess.CalledProcessError as error:

        raise RuntimeError(
            f"Erreur ExifTool : {error.stderr}"
        )

    try:

        metadata = json.loads(result.stdout)

    except json.JSONDecodeError:

        raise RuntimeError(
            "Impossible de lire les métadonnées renvoyées par ExifTool."
        )

    return metadata


def format_date(date_string):

    """
    Convertit DateTimeOriginal au format JJ-MM-AAAA.
    """

    formats = [
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d",
        "%Y-%m-%d"
    ]

    for date_format in formats:

        try:

            date = datetime.strptime(
                date_string[:19],
                date_format
            )

            return date.strftime("%d-%m-%Y")

        except ValueError:
            continue

    raise ValueError(
        f"Format de date non reconnu : {date_string}"
    )


def main():

    args = parse_arguments()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in args.extensions
    }

    # Vérifier le dossier source

    if not input_dir.is_dir():

        sys.exit(
            f"ERREUR : dossier source introuvable : {input_dir}"
        )

    # Éviter de traiter les fichiers de sortie comme des entrées

    if input_dir == output_dir or input_dir in output_dir.parents:

        sys.exit(
            "ERREUR : le dossier de sortie ne doit pas être "
            "à l'intérieur du dossier source."
        )

    # Récupérer les images

    images = sorted(
        image
        for image in input_dir.iterdir()
        if image.is_file()
        and image.suffix.lower() in extensions
    )

    if not images:

        print("Aucune image trouvée.")

        return

    print(f"{len(images)} image(s) trouvée(s).")

    # Lire les métadonnées

    metadata = get_metadata(
        images,
        args.exiftool
    )

    selected_images = []

    for image_metadata in metadata:

        source = Path(image_metadata["SourceFile"])

        stacked_value = str(
            image_metadata.get("StackedImage", "")
        )

        # Vérifier si l'image est focus-stacked

        if "focus-stacked" not in stacked_value.lower():

            continue

        # Récupérer la date de prise de vue

        date_original = image_metadata.get(
            "DateTimeOriginal"
        )

        if not date_original:

            print(
                f"AVERTISSEMENT : date absente pour {source.name}"
            )

            continue

        try:

            date_prefix = format_date(
                str(date_original)
            )

        except ValueError as error:

            print(
                f"AVERTISSEMENT : {error}"
            )

            continue

        # Construire le nouveau nom

        new_name = f"{date_prefix}_{source.name}"

        destination = output_dir / new_name

        selected_images.append(
            (source, destination)
        )

    # Vérifier les doublons de destination

    destinations = [
        destination
        for _, destination in selected_images
    ]

    if len(destinations) != len(set(destinations)):

        sys.exit(
            "ERREUR : plusieurs images produisent le même nom."
        )

    # Vérifier qu'aucun fichier ne sera écrasé

    existing_files = [
        destination
        for destination in destinations
        if destination.exists()
    ]

    if existing_files:

        print(
            "ERREUR : certains fichiers existent déjà :"
        )

        for file in existing_files:
            print(f"  - {file}")

        sys.exit(
            "Aucune copie effectuée. Utilise un dossier de sortie vide."
        )

    # Afficher les résultats

    print(
        f"\n{len(selected_images)} image(s) focus-stacked sélectionnée(s)."
    )

    for source, destination in selected_images:

        print(
            f"{source.name} -> {destination.name}"
        )

    # Mode simulation

    if args.dry_run:

        print(
            "\nMode dry-run : aucune image n'a été copiée."
        )

        return

    # Créer le dossier de destination

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Copier les images

    copied_files = []

    try:

        for source, destination in selected_images:

            shutil.copy2(
                source,
                destination
            )

            copied_files.append(destination)

    except OSError as error:

        # Nettoyer les copies partielles en cas d'erreur

        for file in copied_files:

            try:
                file.unlink()
            except OSError:
                pass

        sys.exit(
            f"Erreur pendant la copie : {error}"
        )

    print(
        f"\nTerminé : {len(copied_files)} image(s) copiée(s)."
    )

    print(
        f"Dossier de destination : {output_dir}"
    )

    print(
        "Les images originales ont été conservées."
    )


if __name__ == "__main__":

    main()