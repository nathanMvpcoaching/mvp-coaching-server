"""
MVP.coaching — Serveur FastAPI avec analyse vidéo réelle
Lance avec : uvicorn main:app --reload --port 8000
"""

import os
import uuid
import json
import re
import base64
import asyncio
from pathlib import Path
from typing import Optional

import anthropic
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

app = FastAPI(title="MVP.coaching API", version="2.0.0")

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
    return {"status": "ok", "service": "MVP.coaching API v2 — Vision activée"}


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


def extract_frames(filepath: str, num_frames: int = 8) -> list:
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo : {filepath}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    print(f"[INFO] Vidéo : {duration:.1f}s, {total_frames} frames, {fps:.1f} fps")

    positions = [
        int(total_frames * (0.05 + 0.90 * i / (num_frames - 1)))
        for i in range(num_frames)
    ]

    frames_b64 = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if w > 800:
            scale = 800 / w
            frame = cv2.resize(frame, (800, int(h * scale)))
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buffer).decode('utf-8')
        frames_b64.append(b64)

    cap.release()
    print(f"[INFO] {len(frames_b64)} captures extraites")
    return frames_b64


async def run_analysis(aid: str):
    job = analyses[aid]
    try:
        job["status"] = "processing"
        job["progress"] = 10

        loop = asyncio.get_event_loop()

        job["progress"] = 25
        frames = await loop.run_in_executor(
            None,
            lambda: extract_frames(job["filepath"], num_frames=8)
        )

        job["progress"] = 50
        report = await call_claude_vision(
            game=job["game"],
            filename=job["filename"],
            frames_b64=frames
        )

        job["progress"] = 90
        await asyncio.sleep(0.5)

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
        import traceback
        traceback.print_exc()


async def call_claude_vision(game: str, filename: str, frames_b64: list) -> dict:
    if not ANTHROPIC_API_KEY:
        raise ValueError("Clé API Anthropic manquante")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    content = []

    content.append({
        "type": "text",
        "text": f"""Tu es MVP.coaching, un coach IA expert en jeux compétitifs spécialisé sur {game}.

Je vais te montrer {len(frames_b64)} captures d'écran extraites d'une replay de {game} ({filename}).
Ces captures sont réparties uniformément sur toute la durée de la partie.

Tu es un coach sévère mais bienveillant. Tu NE décris PAS ce que tu vois.
Tu identifies directement CE QUE LE JOUEUR FAIT MAL et comment le corriger.

Pour chaque capture, pose-toi ces questions :
- Quelle erreur précise le joueur commet-il ici ?
- Quelle aurait été la bonne décision dans cette situation ?
- Quel exercice concret va corriger cette erreur ?

INTERDIT : décrire la scène, raconter ce qui se passe
OBLIGATOIRE : identifier l'erreur, expliquer la correction, donner un exercice
Pour les points "info" uniquement : cite un truc que le joueur fait BIEN
et pourquoi c'est une bonne habitude à garder.
Les points "critique" et "alerte" restent focalisés sur les corrections.

Voici les captures :"""
    })

    for i, frame_b64 in enumerate(frames_b64):
        content.append({
            "type": "text",
            "text": f"\n--- Capture {i+1}/{len(frames_b64)} ---"
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame_b64
            }
        })

    content.append({
        "type": "text",
        "text": f"""

Basé sur ces captures réelles, génère un rapport de coaching en JSON.
Sois SPECIFIQUE et cite ce que tu as vraiment observé dans les images.

Réponds UNIQUEMENT avec ce JSON valide :

{{
  "score_global": <entier entre 40 et 85>,
  "jeu": "{game}",
  "resume": "<2 phrases basées sur ce que tu as observé>",
  "modules": [
    {{
      "id": "decision",
      "label": "Prise de décision",
      "score": <entier 0-100>,
      "niveau": "<Faible|Moyen|Bon|Excellent>",
      "points": [
        {{"type": "critique", "texte": "<observation précise basée sur les images>"}},
        {{"type": "alerte", "texte": "<observation précise>"}},
        {{"type": "info", "texte": "<point positif observé>"}}
      ]
    }},
    {{
      "id": "mecanique",
      "label": "Mécanique et Aim",
      "score": <entier 0-100>,
      "niveau": "<Faible|Moyen|Bon|Excellent>",
      "points": [
        {{"type": "critique", "texte": "<observation>"}},
        {{"type": "alerte", "texte": "<observation>"}},
        {{"type": "info", "texte": "<observation>"}}
      ]
    }},
    {{
      "id": "placement",
      "label": "Placement et Map",
      "score": <entier 0-100>,
      "niveau": "<Faible|Moyen|Bon|Excellent>",
      "points": [
        {{"type": "critique", "texte": "<observation>"}},
        {{"type": "alerte", "texte": "<observation>"}},
        {{"type": "info", "texte": "<observation>"}}
      ]
    }},
    {{
      "id": "awareness",
      "label": "Lecture de l'environnement",
      "score": <entier 0-100>,
      "niveau": "<Faible|Moyen|Bon|Excellent>",
      "points": [
        {{"type": "critique", "texte": "<observation>"}},
        {{"type": "alerte", "texte": "<observation>"}},
        {{"type": "info", "texte": "<observation>"}}
      ]
    }}
  ],
  "plan_semaine": [
    "<exercice concret basé sur les faiblesses observées>",
    "<exercice concret>",
    "<exercice concret>"
  ],
  "point_fort": "<ce que tu as vraiment vu de positif>",
  "priorite": "<la correction la plus urgente basée sur les images>"
}}"""
    })

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": content}]
        )
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r'```json|```', '', raw).strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    raw = raw[start:end]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw = re.sub(r',\s*}', '}', raw)
        raw = re.sub(r',\s*]', ']', raw)
        return json.loads(raw)
