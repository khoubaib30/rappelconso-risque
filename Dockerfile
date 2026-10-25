# Image légère pour servir le modèle sur CPU
FROM python:3.11-slim

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

# 3) Code de l'API et modèle entraîné
COPY app ./app
COPY modele_rappelconso ./modele_rappelconso

# 4) On ne lance pas l'application en root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
