# ─────────────────────────────────────────────────────────────────────────────
# Image unique : API FastAPI + frontend React compilé.   Port 7860 (Hugging Face)
# Build local :  docker build -t platewatch .  &&  docker run -p 7860:7860 platewatch
# ─────────────────────────────────────────────────────────────────────────────

# 1) Compilation du frontend
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# 2) Runtime Python (CPU uniquement)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face exécute le conteneur avec l'utilisateur 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 \
    DEMO_MODE=1 SEED_ON_STARTUP=1
WORKDIR /home/user/app

RUN pip install --user torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# Code + modèles (seuls les 2 scripts CRNN sont nécessaires à l'inférence)
COPY --chown=user backend_pfa/ ./backend_pfa/
COPY --chown=user model/CRNN/04_inference.py model/CRNN/model_02.py ./model/CRNN/
COPY --chown=user models/ ./models/
COPY --chown=user --from=web /web/dist ./frontend/dist

WORKDIR /home/user/app/backend_pfa
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
