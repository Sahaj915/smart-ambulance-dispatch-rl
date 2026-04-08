FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models

# Required env vars (set via HF Spaces Secrets)
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV LOCAL_IMAGE_NAME=""
ENV GROQ_API_KEY=""
ENV PORT=7860
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 dispatch && chown -R dispatch:dispatch /app
USER dispatch

EXPOSE 7860

# Run with uvicorn pointing to the `app` object inside app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
