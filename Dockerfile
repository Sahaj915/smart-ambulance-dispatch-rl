# ─────────────────────────────────────────────────────────────────────────────
#  Dockerfile — AmbulanceDispatchEnv
#  Smart Ambulance Dispatch & Hospital Routing RL Environment
#  Meta PyTorch OpenEnv Hackathon
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . .

# ── Create model directory ────────────────────────────────────────────────────
RUN mkdir -p models

# ── Non-root user (security best practice) ───────────────────────────────────
RUN useradd -m -u 1000 dispatch
RUN chown -R dispatch:dispatch /app
USER dispatch

# ── Default: Launch Gradio app ────────────────────────────────────────────────
EXPOSE 7860
ENV PORT=7860
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "app.py"]

# ─────────────────────────────────────────────────────────────────────────────
#  Alternative commands:
#
#  Training:
#    docker run --rm -v $(pwd)/models:/app/models dispatch-rl \
#      python -m src.train --task all --curriculum
#
#  Inference:
#    docker run --rm -v $(pwd)/models:/app/models dispatch-rl \
#      python -m src.inference --model models/ppo_medium --task medium
#
#  Grader:
#    docker run --rm -v $(pwd)/models:/app/models dispatch-rl \
#      python -m src.grader --task all --model models/ppo_medium --episodes 20
# ─────────────────────────────────────────────────────────────────────────────
