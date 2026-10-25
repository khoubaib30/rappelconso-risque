"""Chargement du modèle CamemBERT fine-tuné et prédiction des familles de risque."""
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def nettoyer(texte: str) -> str:
    """Même nettoyage que dans le notebook d'entraînement."""
    texte = (texte or "").replace("¤", " ")
    return re.sub(r"\s+", " ", texte).strip()


def construire_texte(produit: str, categorie: str, sous_categorie: str, motif: str) -> str:
    """Reproduit exactement le format d'entrée utilisé à l'entraînement.

    Les données RappelConso sont en minuscules : on met donc l'entrée en minuscules
    pour rester dans la même distribution que les données d'entraînement.
    """
    texte = (
        f"Produit : {nettoyer(produit).lower()}"
        f" | Catégorie : {nettoyer(categorie).lower()} / {nettoyer(sous_categorie).lower()}"
        f" | Motif : {nettoyer(motif).lower()}"
    )
    return texte


class ClassifieurRisque:
    def __init__(self, model_dir: Path | None = None):
        # dossier lu au démarrage (variable d'environnement MODEL_DIR), pas à l'import du module
        model_dir = Path(model_dir or os.getenv("MODEL_DIR", "modele_rappelconso"))
        with open(model_dir / "config_api.json", encoding="utf-8") as f:
            config = json.load(f)
        self.labels: list[str] = config["labels"]
        self.seuil: float = config["seuil"]
        self.max_len: int = config["max_len"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()  # mode inférence : désactive le dropout

    @torch.no_grad()
    def predire(self, texte: str) -> dict:
        enc = self.tokenizer(texte, truncation=True, max_length=self.max_len, return_tensors="pt")
        probas = torch.sigmoid(self.model(**enc).logits)[0].tolist()

        scores = {label: round(p, 4) for label, p in zip(self.labels, probas)}
        familles = [label for label, p in scores.items() if p >= self.seuil]
        return {
            "familles": familles or ["autre_a_verifier"],
            "a_verifier": not familles,  # aucune famille assez sûre → relecture humaine
            "scores": dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)),
            "seuil": self.seuil,
        }
