import argparse
import json
import re
from pathlib import Path


def escape_separator(separator):
    """Échappe les caractères spéciaux pour une utilisation en regex."""
    return re.escape(separator)


def ask_separator(label):
    print(f"\n{label}")
    print("Exemples : _ (underscore), - (tiret), . (point), espace")
    separator = input("Séparateur : ")

    if separator == "":
        raise ValueError("Le séparateur ne peut pas être vide.")

    return separator


def main():
    parser = argparse.ArgumentParser(
        description="Configurer le format des identifiants d'une acquisition."
    )

    parser.add_argument("--museum", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--camera", required=True)

    args = parser.parse_args()

    print("\n=== Configuration du format des identifiants ===")

    # Préfixe
    prefix = input("\nPréfixe de l'identifiant (ex. MHNGR) : ").strip()

    if not prefix:
        raise ValueError("Le préfixe ne peut pas être vide.")

    # Premier séparateur
    separator_1 = ask_separator(
        "Séparateur entre le préfixe et le corps"
    )

    # Corps
    print("\n=== Corps de l'identifiant ===")
    print("Le corps est constitué de lettres majuscules.")

    min_letters = int(input("Nombre minimum de lettres : "))
    max_letters = int(input("Nombre maximum de lettres : "))

    if min_letters < 1 or max_letters < min_letters:
        raise ValueError("Intervalle de lettres invalide.")

    # Deuxième séparateur
    separator_2 = ask_separator(
        "Séparateur entre le corps et le numéro"
    )

    # Numéro
    print("\n=== Numéro de l'identifiant ===")

    min_digits = int(input("Nombre minimum de chiffres : "))
    max_digits = int(input("Nombre maximum de chiffres : "))

    if min_digits < 1 or max_digits < min_digits:
        raise ValueError("Intervalle de chiffres invalide.")

    # Construction de la regex
    pattern = (
        r"\b"
        + re.escape(prefix)
        + escape_separator(separator_1)
        + rf"[A-Z]{{{min_letters},{max_letters}}}"
        + escape_separator(separator_2)
        + rf"[0-9]{{{min_digits},{max_digits}}}"
        + r"\b"
    )

    # Configuration JSON
    config = {
        "museum": args.museum,
        "date": args.date,
        "camera": args.camera,
        "identifier_type": "prefix_body_number",
        "prefix": prefix,
        "separator_1": separator_1,
        "body": {
            "type": "uppercase_letters",
            "min_length": min_letters,
            "max_length": max_letters
        },
        "separator_2": separator_2,
        "number": {
            "type": "digits",
            "min_length": min_digits,
            "max_length": max_digits
        },
        "pattern": pattern
    }

    output_dir = Path("config/identifier_formats")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (
        f"{args.museum}_{args.date.replace('-', '')}_{args.camera}.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("\nConfiguration enregistrée :")
    print(output_file)

    print("\nExpression régulière générée :")
    print(pattern)


if __name__ == "__main__":
    main()