"""Colorise des photos noir et blanc avec DDColor, puis ajuste la palette
sur une image de référence (optionnelle).

Exemples :
    python coloriser.py images_source/photo.jpg
    python coloriser.py images_source/photo.jpg --reference images_source/ref.png
    python coloriser.py images_source/ --sortie resultats/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE / "vendor" / "DDColor"))

from ddcolor import DDColor, ColorizationPipeline  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def lire_image(chemin: Path) -> np.ndarray:
    # cv2.imread ne gère pas les chemins avec accents sous Windows
    donnees = np.fromfile(str(chemin), dtype=np.uint8)
    img = cv2.imdecode(donnees, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image illisible : {chemin}")
    return img


def ecrire_image(chemin: Path, img: np.ndarray) -> Path:
    ok, tampon = cv2.imencode(chemin.suffix, img)
    if not ok:
        raise ValueError(f"Impossible d'écrire : {chemin}")
    # si le fichier est ouvert ailleurs (visionneuse), on écrit à côté : _2, _3…
    candidat, n = chemin, 1
    while True:
        try:
            tampon.tofile(str(candidat))
            return candidat
        except OSError:
            n += 1
            if n > 50:
                raise
            candidat = chemin.with_stem(f"{chemin.stem}_{n}")


def charger_modele(nom: str = "ddcolor_modelscope", taille: int = 512) -> ColorizationPipeline:
    from huggingface_hub import PyTorchModelHubMixin

    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modele = DDColorHF.from_pretrained(f"piddnad/{nom}").to(device)
    return ColorizationPipeline(modele, input_size=taille, device=device)


def en_lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def depuis_lab(lab: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR)
    return (np.clip(bgr, 0, 1) * 255.0).round().astype(np.uint8)


def ajuster_tons(l_src: np.ndarray, l_ref: np.ndarray, contraste: float) -> np.ndarray:
    """Étire la luminance sur la plage de la référence (enlève le voile gris),
    puis ajoute une légère courbe en S. Transformation point par point :
    le grain et les détails sont conservés."""
    bas, haut = np.percentile(l_src, [0.5, 99.5])
    ref_bas, ref_haut = np.percentile(l_ref, [0.5, 99.5])
    l = (l_src - bas) / max(haut - bas, 1e-3)
    l = np.clip(l, 0, 1)
    # courbe en S douce centrée sur la médiane
    milieu = np.median(l)
    l = np.where(
        l < milieu,
        milieu * (l / max(milieu, 1e-3)) ** contraste,
        1 - (1 - milieu) * ((1 - l) / max(1 - milieu, 1e-3)) ** contraste,
    )
    return ref_bas + l * (ref_haut - ref_bas)


def eclaircir(l: np.ndarray, l_ref: np.ndarray) -> np.ndarray:
    """Gamma qui amène la luminance moyenne au niveau de la référence."""
    bas, haut = l.min(), l.max()
    x = (l - bas) / max(haut - bas, 1e-3)
    cible = (l_ref.mean() - bas) / max(haut - bas, 1e-3)
    gamma = np.log(np.clip(cible, 0.05, 0.95)) / np.log(np.clip(x.mean(), 0.05, 0.95))
    return bas + x ** np.clip(gamma, 0.6, 1.4) * (haut - bas)


def transferer_couleurs(lab: np.ndarray, lab_ref: np.ndarray, force: float, saturation: float) -> np.ndarray:
    """Rapproche la teinte moyenne (a, b) de celle de la référence, sans élargir
    l'écart des couleurs au-delà du sien, puis augmente la saturation."""
    sortie = lab.copy()
    for c in (1, 2):
        canal = sortie[..., c]
        moy, ecart = canal.mean(), canal.std() + 1e-3
        moy_ref, ecart_ref = lab_ref[..., c].mean(), lab_ref[..., c].std()
        cible = (canal - moy) * min(ecart_ref / ecart, 1.0) + moy_ref
        sortie[..., c] = (1 - force) * canal + force * cible
    sortie[..., 1:] *= saturation
    return sortie


def appliquer_retouches(lab: np.ndarray, chemin: Path) -> np.ndarray:
    """Recolore des zones précises (uniformes, visages…) décrites dans un JSON :
    {"zones": [{"nom": ..., "polygone": [[x, y], ...] ou "ellipse": [cx, cy, rx, ry],
                "ab": [a, b], "force": 0.85, "l_max": 75}]}
    Seule la couleur change : la luminance (le détail) n'est jamais touchée."""
    if not chemin.exists():
        return lab
    sortie = lab.copy()
    h, w = lab.shape[:2]
    for zone in json.loads(chemin.read_text(encoding="utf-8"))["zones"]:
        masque = np.zeros((h, w), np.float32)
        if "polygone" in zone:
            cv2.fillPoly(masque, [np.array(zone["polygone"], np.int32)], 1.0)
        else:
            cx, cy, rx, ry = zone["ellipse"]
            cv2.ellipse(masque, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        masque = cv2.GaussianBlur(masque, (0, 0), 2.5)
        # épargne les pixels clairs (bois, ciel) pris dans la zone
        masque *= np.clip((zone.get("l_max", 100) - lab[..., 0]) / 8, 0, 1)
        masque = (masque * zone.get("force", 0.85))[..., None]
        sortie[..., 1:] = (1 - masque) * sortie[..., 1:] + masque * np.array(zone["ab"], np.float32)
    return sortie


def eclaircir_ciel(lab: np.ndarray) -> np.ndarray:
    """Hautes lumières : blanc cassé chaud à peine bleuté, sans dominante."""
    l = lab[..., 0]
    masque = np.clip((l - 80) / 12, 0, 1)[..., None]
    teinte_ciel = np.array([0.5, -1.5], dtype=np.float32)  # (a, b) : quasi neutre, très léger bleu-gris
    sortie = lab.copy()
    sortie[..., 1:] = (1 - masque * 0.7) * lab[..., 1:] + masque * 0.7 * teinte_ciel
    return sortie


def traiter(
    pipeline: ColorizationPipeline,
    source: Path,
    reference: Path | None,
    dossier_sortie: Path,
    force: float,
    saturation: float,
    contraste: float,
    chaleur: float = 0.0,
) -> Path:
    img = lire_image(source)
    colorisee = pipeline.process(img)
    lab = en_lab(colorisee)
    lab[..., 0] = en_lab(img)[..., 0]  # luminance d'origine, pixel pour pixel

    if reference is not None:
        lab_ref = en_lab(lire_image(reference))
        lab[..., 0] = ajuster_tons(lab[..., 0], lab_ref[..., 0], contraste)
        lab[..., 0] = eclaircir(lab[..., 0], lab_ref[..., 0])
        lab = transferer_couleurs(lab, lab_ref, force, saturation)
        lab[..., 1] += chaleur * 0.3  # un peu de rouge, surtout du jaune
        lab[..., 2] += chaleur
        lab = eclaircir_ciel(lab)
    else:
        lab[..., 1:] *= saturation
    lab = appliquer_retouches(lab, RACINE / "retouches" / f"{source.stem}.json")

    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin = dossier_sortie / f"{source.stem}_couleur.png"
    return ecrire_image(chemin, depuis_lab(lab))


def main() -> None:
    p = argparse.ArgumentParser(description="Colorise des photos noir et blanc.")
    p.add_argument("entree", type=Path, help="Image ou dossier d'images noir et blanc")
    p.add_argument("--reference", type=Path, help="Image couleur dont on reprend la palette")
    p.add_argument("--sortie", type=Path, default=RACINE / "resultats", help="Dossier de sortie")
    p.add_argument("--force", type=float, default=0.95, help="Poids du transfert de palette (0 à 1)")
    p.add_argument("--saturation", type=float, default=1.45, help="Multiplicateur de saturation")
    p.add_argument("--chaleur", type=float, default=3.0, help="Décalage vers le jaune-orangé (0 = aucun)")
    p.add_argument("--contraste", type=float, default=1.15, help="Force de la courbe en S (1 = aucune)")
    p.add_argument("--taille", type=int, default=512, help="Résolution de travail du modèle")
    args = p.parse_args()

    if args.entree.is_dir():
        sources = sorted(f for f in args.entree.iterdir() if f.suffix.lower() in EXTENSIONS)
        if args.reference is not None:
            sources = [f for f in sources if f.resolve() != args.reference.resolve()]
    else:
        sources = [args.entree]
    if not sources:
        sys.exit("Aucune image à traiter.")

    pipeline = charger_modele(taille=args.taille)
    for source in sources:
        chemin = traiter(pipeline, source, args.reference, args.sortie, args.force, args.saturation, args.contraste, args.chaleur)
        print(f"{source.name} -> {os.path.relpath(chemin)}")


if __name__ == "__main__":
    main()
