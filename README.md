# tuto

tutoriel michel

Ceci est la première modif - zuper

## Colorisation de photos 14-18

`coloriser.py` colorise des photos noir et blanc avec le modèle DDColor
(code dans `vendor/DDColor`), puis ajuste la palette sur une image de référence.
La luminance d'origine est conservée pixel pour pixel : composition, détails et
grain ne changent pas, seules les couleurs sont ajoutées.

### Installation

Python 3.12 et une carte NVIDIA sont recommandés (sur CPU ça marche, mais lentement).
`.venv/` et `vendor/` ne sont pas versionnés : il faut les recréer.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
git clone https://github.com/piddnad/DDColor.git vendor/DDColor
git -C vendor/DDColor checkout 2adb63f2656ac41cbdf7b894cddd94121a3faf13
```

Le modèle (`piddnad/ddcolor_modelscope`) se télécharge tout seul au premier lancement.

### Utilisation

```bash
.venv\Scripts\python coloriser.py images_source\photo.jpg --reference images_source\ref.png
```

Options : `--force` (poids de la palette de référence), `--saturation`, `--chaleur`,
`--contraste`, `--sortie`. Les retouches locales (uniformes, visages) se décrivent
dans `retouches/<nom de l'image>.json`. Les résultats vont dans `resultats/`.
