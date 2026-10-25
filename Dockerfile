# Image légère pour servir le modèle sur CPU
FROM python:3.11-slim

# Dépôt Hugging Face qui contient le modèle fine-tuné
ARG MODEL_REPO=khoubeibbourbia/camembert-rappelconso

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/modele_rappelconso

WORKDIR /app

# 1) PyTorch version CPU seulement (beaucoup plus léger que la version GPU)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Dépendances, installées avant le code pour profiter du cache Docker
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Téléchargement du modèle depuis Hugging Face pendant la construction de l'image
RUN test -n "$MODEL_REPO" || (echo "Préciser --build-arg MODEL_REPO=pseudo/nom-du-modele" && exit 1) && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_REPO}', local_dir='${MODEL_DIR}')"

# 4) Code de l'API
COPY app ./app

# 5) On ne lance pas l'application en root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
