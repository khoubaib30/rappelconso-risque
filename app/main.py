"""API FastAPI : classification des fiches de rappel RappelConso par famille de risque."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.modele import ClassifieurRisque, construire_texte

etat = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # le modèle est chargé une seule fois au démarrage, pas à chaque requête
    etat["classifieur"] = ClassifieurRisque()
    yield
    etat.clear()


app = FastAPI(
    title="RappelConso – familles de risque",
    description="Prédit la ou les familles de risque d'une fiche de rappel de produit à partir de son texte (CamemBERT fine-tuné).",
    version="1.0.0",
    lifespan=lifespan,
)


class FicheRappel(BaseModel):
    produit: str = Field(..., min_length=1, max_length=300, examples=["fromage au lait cru 250 g"])
    categorie: str = Field("", max_length=100, examples=["alimentation"])
    sous_categorie: str = Field("", max_length=100, examples=["lait et produits laitiers"])
    motif: str = Field(..., min_length=3, max_length=2000,
                       examples=["présence de listeria monocytogenes détectée lors d'un autocontrôle"])


class Prediction(BaseModel):
    familles: list[str]
    a_verifier: bool
    scores: dict[str, float]
    seuil: float


@app.get("/health")
def health():
    return {"statut": "ok", "modele_charge": "classifieur" in etat}


@app.post("/predict", response_model=Prediction)
def predict(fiche: FicheRappel):
    texte = construire_texte(fiche.produit, fiche.categorie, fiche.sous_categorie, fiche.motif)
    return etat["classifieur"].predire(texte)


@app.get("/", include_in_schema=False)
def demo():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
