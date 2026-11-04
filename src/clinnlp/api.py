"""FastAPI service. Research/demo only - NOT for clinical use."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import MODELS, Pipeline

MAX_CHARS = 50_000
DISCLAIMER = "Research prototype. Not a medical device; not validated on real clinical notes; do not use for clinical decisions or with real patient data."
state = {}


@asynccontextmanager
async def lifespan(app):
    state["pipe"] = Pipeline(os.environ.get("CLINNLP_MODELS", MODELS))
    yield
    state.clear()


app = FastAPI(title="clinnlp", version="0.1.0", description=DISCLAIMER, lifespan=lifespan)


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


class ExtractIn(TextIn):
    aggressive_deid: bool = True
    assertion_mode: str = Field("ml", pattern="^(ml|rules)$")


class DeidIn(TextIn):
    aggressive: bool = True
    style: str = Field("tag", pattern="^(tag|mask)$")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "pipe" in state, "disclaimer": DISCLAIMER}


@app.post("/extract")
def extract(body: ExtractIn):
    """PHI detection -> disease NER -> assertion (negation/uncertainty/hypothetical, temporality, experiencer) -> MeSH id."""
    return {**state["pipe"].extract(body.text, body.aggressive_deid, body.assertion_mode), "disclaimer": DISCLAIMER}


@app.post("/deidentify")
def deidentify(body: DeidIn):
    return {**state["pipe"].deidentify(body.text, body.aggressive, body.style), "disclaimer": DISCLAIMER}


@app.post("/normalize")
def normalize(body: TextIn):
    ui, name, score, method = state["pipe"].norm.lookup(body.text)
    if ui is None:
        raise HTTPException(404, detail={"message": "no MeSH match above threshold", "best_score": round(score, 4)})
    return {"mesh_id": ui, "mesh_name": name, "score": round(score, 4), "method": method}
