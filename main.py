"""
MVP.coaching — Serveur FastAPI
Lance avec : uvicorn main:app --reload --port 8000
"""

import os
import uuid
import json
import re
import asyncio
from pathlib import Path
from typing import Optional

import anthropic
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

app = FastAPI(title="MVP.coaching API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyses: dict = {}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class AnalysisStatus(BaseModel):
    id: str
    status: str
    progress: int
    report: Optional[dict] = None
    error: Optional[str] = None


@app.get("/")
def health():
    return {"status": "ok", "service": "MVP.coaching API"}


@app.post("/v1/analyses", response_model=AnalysisStatus)
async def create_analysis(
    file: UploadFile = File(...),
    game: str = Form(default="Valorant")
):
    aid = str(uuid.uuid4())
    suffix = Path(file.filename).suffix or ".mp4"
    filepath = UPLOAD_DIR / f"{aid}{suffix}"
    content = await file.read()
    filepath.write_bytes(content)

    analyses[aid] = {
        "id": aid,
        "status": "pending",
        "progress": 0,
        "report": None,
        "error": None,
        "game": game,
        "filename": file.filename,
        "filesize": len(content),
        "filepath": str(filepath),
    }

    asyncio.create_task(run_analysis(aid))
    return AnalysisStatus(**analyses[aid])


@app.get("/v1/analyses/{aid}", response_model=AnalysisStatus)
async def get_analysis(aid: str):
    if aid not in analyses:
        raise HTTPException(404, "Analyse introuvable")
    return AnalysisStatus(**analyses[aid])


async def run_analysis(aid: str):
    job = analyses[aid]
    try:
        job["status"] = "processing"
        job["progress"] = 15
        await asyncio.sleep(1.2)

        job["progress"] = 35
        await asyncio.sleep(1.5)

        job["progress"] = 55
        report = await call_claude(job["game"], job["filename"], job["filesize"])

        job["progress"] = 85
        await asyncio.sleep(0.8)

        job["progress"] = 100
        job["status"] = "completed"
        job["report"] = report

        try:
            Path(job["filepath"]).unlink(missing_ok=True)
        except Exception:
            pass

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        print(f"[ERREUR] Analyse {aid} : {e}")


async def call_claude(game: str, filename: str, filesize: int) -> dict:
    if not ANTHROPIC_API_KEY:
        raise ValueError("Clé API Anthropic manquante dans le fichier .env")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""Tu es MVP.coaching, un coach IA expert en jeux compétitifs.
Un joueur vient de t'envoyer une rediffusion de sa partie sur {game}.
Fichier : {filename} ({round(filesize / 1024 / 1024, 1)} Mo)

Génère un rapport de coaching complet et réaliste en JSON avec EXACTEMENT cette structure :

{{
  "score_global": <entier entre 40 et 85>,
  "jeu": "{game}",
  "resume": "<2 phrases résumant la performance globale>",
  "modules": [
    {{
      "id": "decision",
      "label": "Prise de décision",
      "score": <entier 0-100>,
      "niveau": "<Faible|Moyen|Bon|Excellent>",
      "points": [
        {{"type": "critique", "texte": "<observation précise>"}},
        {{"type": "alerte", "texte": "<observation précise>"}},
        {{"type": "info", "texte": "<observation précise>"}}
      ]
    }},
    {{
      "id": "mecanique",
      "label": "Mécanique et Aim",
      "score": <entier 0-100>,
      "niveau": "<Faible|Moyen|Bon|Excellent>",
      "points": [
        {{"type": "critique", "texte": "<observation précise>"}},
        {{"type": "alerte", "texte": "<observation précise>"}},
        {{"type": "info", "texte": "<observation précise>"}}
      ]
    }},
    {{
      "id": "placement",
      "label": "Placement et Map",
      "score": <entier 0-100>,
      "niveau": "<Faible|Moyen|Bon|Excellent>",
      "points": [
        {{"type": "critique", "texte": "<observation précise>"}},
        {{"type": "alerte", "texte": "<observation précise>"}},
        {{"type": "info", "texte": "<observation précise>"}}
      ]
    }},
    {{
      "id": "awareness",
      "label": "Lecture de l'environnement",
      "score": <entier 0-100>,
      "niveau": "<Faible|Moyen|Bon|Excellent>",
      "points": [
        {{"type": "critique", "texte": "<observation précise>"}},
        {{"type": "alerte", "texte": "<observation précise>"}},
        {{"type": "info", "texte": "<observation précise>"}}
      ]
    }}
  ],
  "plan_semaine": [
    "<exercice concret 1>",
    "<exercice concret 2>",
    "<exercice concret 3>"
  ],
  "point_fort": "<la meilleure chose que le joueur a faite>",
  "priorite": "<la chose la plus importante a corriger>"
}}

Reponds UNIQUEMENT avec le JSON valide, sans texte avant ou apres, sans markdown, sans apostrophes dans les cles."""

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
    )

    raw = response.content[0].text.strip()

    # Nettoie et parse le JSON
    raw = re.sub(r'```json|```', '', raw).strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    raw = raw[start:end]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Corrige les virgules en trop
        raw = re.sub(r',\s*}', '}', raw)
        raw = re.sub(r',\s*]', ']', raw)
        return json.loads(raw)
