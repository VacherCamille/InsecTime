# Scripts OCR et validation humaine

## Contenu

- `workflow/scripts/label_extractor_Tesseract.py` : détection des zones de texte DB18 et OCR Tesseract.
- `workflow/scripts/validate_labels_app.py` : interface Streamlit pour choisir le format d'identifiant et valider chaque image.
- `workflow/scripts/rename_images.py` : copie les images validées dans le dossier final avec leur identifiant comme nom.

## Dépendances

Dans l'environnement Python du workflow :

```bash
pip install opencv-python numpy Pillow pytesseract streamlit
```

Tesseract doit aussi être installé sur le système (par exemple `sudo apt install tesseract-ocr` sous Debian/Ubuntu).

## 1. Lancer l'OCR

Exemple depuis la racine du projet :

```bash
python workflow/scripts/label_extractor_Tesseract.py \
  --input-dir data/intermediate/stacked/MNHN/2026-10-10/MOMO \
  --model Models/DB_TD500_resnet18.onnx \
  --output-csv data/intermediate/ocr/MNHN/2026-10-10/MOMO/labels.csv
```

Le dossier de sortie est créé automatiquement.
Le CSV conserve le texte OCR brut. Le format d'identifiant n'est pas imposé par le script OCR : il sera défini dans l'interface pour chaque jeu de photos.

## 2. Ouvrir l'interface de validation

L'interface recherche les fichiers `labels.csv` sous :

`data/intermediate/ocr/`

Elle déduit alors le dossier d'images stacked correspondant sous `data/intermediate/stacked/`.

Lancer :

```bash
streamlit run workflow/scripts/validate_labels_app.py
```

Sélectionner le jeu de photos, définir le format attendu (préfixe + nombre de chiffres ou regex personnalisée), puis vérifier chaque image. Chaque enregistrement est écrit dans :

`results/validation/MNHN/2026-10-10/MOMO/validated_labels.csv`

La configuration du format est conservée dans le même dossier sous `validation_config.json`.

## 3. Renommer/copier après validation

```bash
python workflow/scripts/rename_images.py \
  --images-dir data/intermediate/stacked/MNHN/2026-10-10/MOMO \
  --validated-csv results/validation/MNHN/2026-10-10/MOMO/validated_labels.csv \
  --config results/validation/MNHN/2026-10-10/MOMO/validation_config.json \
  --output-dir data/intermediate/identified/MNHN/2026-10-10/MOMO
```

Le script :
- refuse de procéder si une image est encore `pending` ;
- vérifie le format des identifiants validés ;
- refuse les identifiants dupliqués ;
- place les images `unidentified` et `ambiguous` dans des sous-dossiers dédiés ;
- copie les images et ne modifie jamais les originaux.

## Points à vérifier avant traitement réel

1. Le modèle ONNX doit être disponible au chemin indiqué.
2. Le code OCR reprend l'approche DB18 + Tesseract du script fourni, mais son résultat doit être testé sur quelques images représentatives.
3. L'interface suppose que les fichiers OCR sont rangés sous `data/intermediate/ocr/{museum}/{date}/{camera}/labels.csv` et les images sous `data/intermediate/stacked/{museum}/{date}/{camera}/`.
4. La règle de renommage interdit les identifiants dupliqués. Si plusieurs images peuvent légitimement représenter le même spécimen, il faudra définir une convention de nommage qui ajoute un suffixe à l'identifiant.
