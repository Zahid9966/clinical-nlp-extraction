FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# python-crfsuite builds from source on some platforms
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .
COPY scripts ./scripts
# Public data is downloaded and models are trained at build time (network needed, ~10 min).
RUN python scripts/fetch_data.py && python scripts/train.py
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "clinnlp.api:app", "--host", "0.0.0.0", "--port", "8000"]
