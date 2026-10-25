from fastapi.testclient import TestClient

from app.modele import ClassifieurRisque, construire_texte

FICHE = {
    "produit": "Fromage au lait cru",
    "categorie": "Alimentation",
    "sous_categorie": "Lait et produits laitiers",
    "motif": "Présence de listeria monocytogenes ¤ détectée",
}


def test_format_texte_identique_a_l_entrainement():
    texte = construire_texte(FICHE["produit"], FICHE["categorie"], FICHE["sous_categorie"], FICHE["motif"])
    assert texte == ("Produit : fromage au lait cru | Catégorie : alimentation / lait et produits laitiers"
                     " | Motif : présence de listeria monocytogenes détectée")


def test_prediction_bien_formee(mini_modele):
    resultat = ClassifieurRisque(mini_modele).predire("Produit : test | Motif : listeria")
    assert len(resultat["scores"]) == 7
    assert all(0.0 <= p <= 1.0 for p in resultat["scores"].values())
    assert resultat["familles"]
    assert resultat["a_verifier"] == (resultat["familles"] == ["autre_a_verifier"])


def test_api_de_bout_en_bout(mini_modele):
    from app.main import app

    with TestClient(app) as client:  # le "with" déclenche le chargement du modèle au démarrage
        assert client.get("/health").json() == {"statut": "ok", "modele_charge": True}

        rep = client.post("/predict", json=FICHE)
        assert rep.status_code == 200
        assert set(rep.json()) == {"familles", "a_verifier", "scores", "seuil"}

        # motif manquant → erreur de validation
        assert client.post("/predict", json={"produit": "x"}).status_code == 422

        assert client.get("/").status_code == 200
