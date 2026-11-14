# Hugging Face Docker Space for the ArtFAE-Net study (CPU only; no model weights, no torch).
# Follows the Spaces Docker guidance: the container runs as user ID 1000; set up the user and WORKDIR before COPY.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
# Secrets (HF_TOKEN) and Variables (STUDY_MODE, DATASET_REPO, ...) arrive as environment variables at run time.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
