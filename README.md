# Classification des rappels de produits par famille de risque (CamemBERT)

Chaque année, des milliers de produits sont rappelés en France (aliments contaminés, jouets dangereux, appareils défectueux…).
Les fiches publiées sur **RappelConso** décrivent le problème en texte libre. Ce projet entraîne un modèle de langue français
(**CamemBERT**, fine-tuné en **PyTorch**) qui lit une fiche et prédit sa ou ses **familles de risque**, puis le sert via une
**API FastAPI** conteneurisée avec **Docker** et déployée sur **Azure**.

**Usage visé :** trier automatiquement les nouveaux rappels (distributeurs, applications de consommateurs, services de contrôle),
et renvoyer en relecture humaine les fiches pour lesquelles le modèle n'est pas assez sûr.

## Données

- Source : [RappelConso V2](https://www.data.gouv.fr/datasets/rappelconso-v2-rappels-de-produits/) (DGCCRF, DGAL…), Licence Ouverte 2.0
- 19 111 fiches publiées entre 2018 et 2026 ; 18 341 après retrait des fiches dont le seul risque est « autre risque »
- Entrée du modèle : nom du produit + catégorie + motif du rappel
- Étiquettes : les 300+ valeurs de `risques_encourus` regroupées en **7 familles** (multi-label, une fiche peut en avoir plusieurs) :
  microbiologique, chimique, allergène, corps étranger, blessure / physique, incendie / électrique, étiquetage / qualité
- Les colonnes qui décrivent directement le risque (`preconisations_sanitaires`, `description_complementaire_risque`) sont exclues pour éviter toute **fuite de données**

**Découpage temporel**, comme en production (on prédit le futur à partir du passé) :

| Jeu | Période | Fiches |
|---|---|---|
| Entraînement | 2018 – 2024 | 12 625 |
| Validation (choix du modèle et du seuil) | 2025 | 2 965 |
| Test (score final) | 2026 | 2 751 |

## Résultats (test 2026)

| Modèle | F1 macro | F1 micro |
|---|---|---|
| Baseline TF-IDF + régression logistique | 0,836 | 0,862 |
| **CamemBERT fine-tuné** | **0,902** | **0,919** |

F1 par famille :

| Famille | Baseline | CamemBERT | Écart |
|---|---|---|---|
| Microbiologique | 0,951 | 0,959 | +0,008 |
| Chimique | 0,786 | 0,874 | +0,088 |
| Allergène | 0,779 | 0,918 | +0,139 |
| Corps étranger | 0,883 | 0,915 | +0,032 |
| Blessure / physique | 0,789 | 0,901 | +0,112 |
| Incendie / électrique | 0,898 | 0,946 | +0,048 |
| Étiquetage / qualité | 0,765 | 0,801 | +0,036 |

Le gain est le plus fort là où le vocabulaire varie le plus (allergènes, blessures, risques chimiques) : la baseline dépend
des mots exacts, alors que CamemBERT reconnaît des formulations différentes d'un même risque. Sur le microbiologique, où le
nom de la bactérie apparaît souvent dans le texte, la baseline était déjà très bonne. La famille allergène ne compte que
61 fiches dans le test : son score est donc moins stable que les autres.

Seuil de décision : 0,45, choisi sur la validation 2025 (jamais sur le test).

### Analyse d'erreurs

295 fiches du test sur 2 751 (10,7 %) ont au moins une famille mal prédite. Les 10 confusions les plus fréquentes
représentent 171 de ces erreurs :

| Vraies familles → prédites | Fiches |
|---|---|
| chimique → blessure / physique | 33 |
| microbiologique → chimique | 23 |
| chimique + blessure → blessure | 21 |
| microbiologique → étiquetage / qualité | 20 |
| chimique → chimique + blessure | 16 |
| étiquetage / qualité → microbiologique | 15 |
| blessure / physique → chimique | 13 |
| blessure + incendie / électrique → incendie / électrique | 12 |
| blessure / physique → corps étranger | 10 |
| microbiologique + étiquetage → étiquetage / qualité | 8 |

Trois zones concentrent les erreurs :

- **Chimique ↔ blessure (83 fiches, environ 28 % des erreurs).** Surtout des produits non alimentaires (jouets,
  cosmétiques, piles bouton) où la source associe souvent un risque chimique et un risque de blessure.
- **Multi-label partiel (57 fiches).** Le modèle trouve le risque principal mais oublie ou ajoute le risque secondaire
  (blessure + incendie → incendie seul, par exemple).
- **Microbiologique ↔ étiquetage / qualité (43 fiches).** Les défauts de stérilisation et les emballages gonflés sont
  étiquetés tantôt comme défaut de fabrication (la cause), tantôt comme risque microbiologique (la conséquence).

L'erreur microbiologique → chimique (23 fiches) vient en partie de mon regroupement : l'histamine est rangée en
« chimique » quand la source écrit « toxines endogènes : histamine », mais en « microbiologique » quand elle écrit
« autres contaminants biologiques ». Le modèle reçoit donc deux étiquettes différentes pour le même problème.

En relisant un échantillon de 15 erreurs, on retrouve ces mêmes causes, qui viennent surtout des données plutôt que
du modèle :

1. **Étiquettes incohérentes dans la source.** Un même motif (« perte de maîtrise du process de stérilisation ») est
   étiqueté « défaut de fabrication » sur une fiche et « botulisme + défaut de fabrication » sur une autre. L'histamine est
   classée tantôt en contaminant biologique, tantôt en toxine. Ces incohérences fixent une limite au score atteignable.
2. **Cause ou conséquence.** Pour un « corps étranger métallique » dans une merguez, la fiche indique la conséquence
   (« blessures, étouffement ») ; le modèle prédit la cause (« corps étranger »). Les deux sont défendables.
3. **Motifs sans information.** « Lot non conforme », « une non-conformité a été mise en évidence », « ne satisfait pas
   à la norme EN 14350 » : le texte ne permet pas de deviner le risque. Ce sont exactement les fiches qui devraient partir
   en relecture humaine.
4. **Risques secondaires oubliés.** Le modèle trouve le risque principal (incendie, choc électrique) mais rate parfois
   un risque associé (brûlures, risque chimique des piles bouton).

**Pistes d'amélioration :** ranger l'histamine dans une seule famille, harmoniser les défauts de stérilisation,
ajuster un seuil par famille plutôt qu'un seuil global pour mieux capter les risques secondaires, et
envoyer en relecture humaine non seulement les fiches sans famille, mais aussi celles dont le motif est trop court ou
trop générique.

Les expériences (paramètres, courbes, métriques) sont suivies avec **MLflow**.

## Architecture

```
CSV RappelConso ──► Notebook Colab (GPU) ──► modèle CamemBERT fine-tuné
                     baseline + MLflow              │
                                                    ▼
                         API FastAPI (app/) ◄── image Docker
                                                    │
                     Azure Container Registry ──► Azure Container Apps (URL publique)
```

## Structure

```
app/
  main.py          API FastAPI : /predict, /health, page de démo
  modele.py        chargement du modèle et prédiction
  static/index.html  page de démo
notebooks/
  rappelconso_camembert.ipynb   préparation, baseline, fine-tuning, MLflow, analyse d'erreurs
tests/             tests de l'API (avec un mini-modèle factice)
Dockerfile
```

## Lancer en local

1. Entraîner le modèle avec le notebook (Google Colab, GPU T4), puis décompresser `modele_rappelconso.zip` à la racine du projet.
2. Installer et lancer l'API :

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

3. Ouvrir http://localhost:8000 (page de démo) ou http://localhost:8000/docs (documentation de l'API).

Exemple d'appel :

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"produit": "fromage au lait cru", "categorie": "alimentation", "sous_categorie": "lait et produits laitiers", "motif": "présence de listeria monocytogenes détectée lors d un autocontrôle"}'
```

Tests : `pytest`

## Docker

```bash
docker build -t rappelconso-api .
docker run -p 8000:8000 rappelconso-api
```

## Déploiement sur Azure

_À compléter lors du déploiement (Azure Container Registry + Azure Container Apps)._
