"""Crée un mini-modèle aléatoire pour tester l'API sans télécharger CamemBERT.

Les tests vérifient la *mécanique* de l'API (chargement, format d'entrée, réponse),
pas la qualité des prédictions.
"""
import json
import os

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

LABELS = ["microbiologique", "chimique", "allergene", "corps_etranger",
          "blessure_physique", "incendie_electrique", "etiquetage_qualite"]


@pytest.fixture(scope="session")
def mini_modele(tmp_path_factory):
    dossier = tmp_path_factory.mktemp("modele")

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for i, mot in enumerate("produit catégorie motif listeria verre allergène".split()):
        vocab[mot] = i + 4
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                        cls_token="[CLS]", sep_token="[SEP]")
    tokenizer.save_pretrained(dossier)

    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, num_labels=len(LABELS), problem_type="multi_label_classification")
    BertForSequenceClassification(config).save_pretrained(dossier)

    with open(dossier / "config_api.json", "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "seuil": 0.5, "max_len": 64}, f)

    os.environ["MODEL_DIR"] = str(dossier)
    return dossier
