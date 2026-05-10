"""
MVP.coaching — Serveur FastAPI avec Riot API + Claude Vision
"""

import os
import uuid
import json
import re
import base64
import asyncio
import httpx
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
RIOT_API_KEY = os.getenv("RIOT_API_KEY", "")

app = FastAPI(title="MVP.coaching API", version="3.0.0")

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
    return {"status": "ok", "service": "MVP.coaching API v3 — Riot + Vision"}


# ── RIOT API ──────────────────────────────────────────────────────────────────

REGIONS = {
    "eu": "euw1",
    "euw": "euw1",
    "eune": "eun1",
    "na": "na1",
    "kr": "kr",
    "br": "br1",
    "jp": "jp1",
    "lan": "la1",
    "las": "la2",
    "oce": "oc1",
    "tr": "tr1",
    "ru": "ru",
}

ROUTING = {
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "kr": "asia",
    "jp1": "asia",
    "oc1": "sea",
}

RANK_ORDER = ["IRON","BRONZE","SILVER","GOLD","PLATINUM","EMERALD","DIAMOND","MASTER","GRANDMASTER","CHALLENGER"]
TIER_LABELS = {"I":"1","II":"2","III":"3","IV":"4"}


async def get_riot_stats(riot_id: str, region: str = "euw") -> dict:
    """
    Récupère les stats Valorant d'un joueur via l'API Riot.
    riot_id format : "NomJoueur#TAG" ou "NomJoueur"
    """
    if not RIOT_API_KEY:
        return {}

    platform = REGIONS.get(region.lower(), "euw1")
    routing = ROUTING.get(platform, "europe")

    headers = {"X-Riot-Token": RIOT_API_KEY}

    try:
        async with httpx.AsyncClient(timeout=10) as client:

            # Parse game name + tagline
            if "#" in riot_id:
                game_name, tag_line = riot_id.split("#", 1)
            else:
                game_name = riot_id
                tag_line = "EUW"

            # 1. Récupère le PUUID via le Riot Account API
            account_url = f"https://{routing}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
            account_resp = await client.get(account_url, headers=headers)

            if account_resp.status_code != 200:
                print(f"[RIOT] Account not found: {account_resp.status_code}")
                return {}

            account = account_resp.json()
            puuid = account["puuid"]

            # 2. Récupère les matchs récents (Valorant)
            matches_url = f"https://{routing}.api.riotgames.com/val/match/v1/matchlists/by-puuid/{puuid}"
            matches_resp = await client.get(matches_url, headers=headers)

            stats = {
                "riot_id": riot_id,
                "puuid": puuid,
                "game_name": game_name,
                "tag_line": tag_line,
                "region": platform,
            }

            if matches_resp.status_code == 200:
                matches_data = matches_resp.json()
                match_ids = [m["matchId"] for m in matches_data.get("history", [])[:10]]
                stats["recent_matches_count"] = len(match_ids)

                # 3. Analyse les 5 derniers matchs
                kills_total = 0
                deaths_total = 0
                assists_total = 0
                wins = 0
                agents = {}
                hs_pct_total = 0
                match_count = 0

                for match_id in match_ids[:5]:
                    match_url = f"https://{routing}.api.riotgames.com/val/match/v1/matches/{match_id}"
                    match_resp = await client.get(match_url, headers=headers)

                    if match_resp.status_code != 200:
                        continue

                    match_data = match_resp.json()
                    players = match_data.get("players", {}).get("allPlayers", [])

                    for player in players:
                        if player.get("puuid") == puuid:
                            stats_p = player.get("stats", {})
                            kills_total += stats_p.get("kills", 0)
                            deaths_total += stats_p.get("deaths", 1)
                            assists_total += stats_p.get("assists", 0)

                            agent = player.get("characterId", "Unknown")
                            agents[agent] = agents.get(agent, 0) + 1

                            hs = stats_p.get("headshots", 0)
                            bs = stats_p.get("bodyshots", 0)
                            ls = stats_p.get("legshots", 0)
                            total_shots = hs + bs + ls
                            if total_shots > 0:
                                hs_pct_total += (hs / total_shots) * 100

                            # Check win
                            teams = match_data.get("teams", {})
                            player_team = player.get("teamId", "")
                            for team_id, team_data in teams.items():
                                if team_id == player_team and team_data.get("won"):
                                    wins += 1
                            match_count += 1
                            break

                if match_count > 0:
                    kda = round((kills_total + assists_total) / max(deaths_total, 1), 2)
                    stats["kda"] = kda
                    stats["kills_avg"] = round(kills_total / match_count, 1)
                    stats["deaths_avg"] = round(deaths_total / match_count, 1)
                    stats["assists_avg"] = round(assists_total / match_count, 1)
                    stats["winrate"] = round((wins / match_count) * 100, 1)
                    stats["hs_percent"] = round(hs_pct_total / match_count, 1)
                    stats["matches_analyzed"] = match_count
                    stats["top_agents"] = list(agents.keys())[:3]

            return stats

    except Exception as e:
        print(f"[RIOT ERROR] {e}")
        return {}


# ── ANALYSES ─────────────────────────────────────────────────────────────────

@app.post("/v1/analyses", response_model=AnalysisStatus)
async def create_analysis(
    file: UploadFile = File(...),
    game: str = Form(default="Valorant"),
    riot_id: str = Form(default=""),
    region: str = Form(default="euw"),
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
        "riot_id": riot_id,
        "region": region,
    }

    asyncio.create_task(run_analysis(aid))
    return AnalysisStatus(**analyses[aid])


@app.get("/v1/analyses/{aid}", response_model=AnalysisStatus)
async def get_analysis(aid: str):
    if aid not in analyses:
        raise HTTPException(404, "Analyse introuvable")
    return AnalysisStatus(**analyses[aid])


# ── Route pour chercher un joueur Riot ───────────────────────────────────────

@app.get("/v1/riot/player")
async def get_player(riot_id: str, region: str = "euw"):
    """Vérifie et récupère les infos d'un joueur Riot."""
    if not RIOT_API_KEY:
        raise HTTPException(400, "Clé Riot API manquante")
    stats = await get_riot_stats(riot_id, region)
    if not stats:
        raise HTTPException(404, "Joueur introuvable")
    return stats


# ── ANALYSE ──────────────────────────────────────────────────────────────────

async def run_analysis(aid: str):
    job = analyses[aid]
    try:
        job["status"] = "processing"
        job["progress"] = 8

        # Récupère les stats Riot si un riot_id est fourni
        riot_stats = {}
        if job.get("riot_id") and job["game"] == "Valorant":
            job["progress"] = 15
            print(f"[INFO] Récupération stats Riot pour {job['riot_id']}")
            riot_stats = await get_riot_stats(job["riot_id"], job.get("region", "euw"))
            print(f"[INFO] Stats Riot: {riot_stats}")

        # Extraction des frames
        job["progress"] = 25
        loop = asyncio.get_event_loop()
        frames = await loop.run_in_executor(
            None,
            lambda: extract_frames(job["filepath"], num_frames=8)
        )

        # Analyse Claude Vision
        job["progress"] = 50
        report = await call_claude_vision(
            game=job["game"],
            filename=job["filename"],
            frames_b64=frames,
            riot_stats=riot_stats
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


async def call_claude_vision(game: str, filename: str, frames_b64: list, riot_stats: dict = {}) -> dict:
    if not ANTHROPIC_API_KEY:
        raise ValueError("Clé API Anthropic manquante")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Construction du contexte Riot
    riot_context = ""
    if riot_stats:
        riot_context = f"""
DONNÉES RÉELLES DU JOUEUR (API Riot) :
- Pseudo : {riot_stats.get('game_name', 'Inconnu')}#{riot_stats.get('tag_line', '')}
- KDA moyen (5 dernières parties) : {riot_stats.get('kda', 'N/A')}
- Kills/Deaths/Assists moyens : {riot_stats.get('kills_avg', 'N/A')}/{riot_stats.get('deaths_avg', 'N/A')}/{riot_stats.get('assists_avg', 'N/A')}
- Win rate (5 dernières parties) : {riot_stats.get('winrate', 'N/A')}%
- Headshot % moyen : {riot_stats.get('hs_percent', 'N/A')}%
- Agents joués récemment : {', '.join(riot_stats.get('top_agents', [])) or 'N/A'}
- Nombre de parties analysées : {riot_stats.get('matches_analyzed', 'N/A')}

Utilise ces données réelles pour personnaliser le rapport. Par exemple :
- Si le KDA est faible (<1.5), focus sur la survie et les décisions
- Si le HS% est faible (<20%), focus sur la visée et le crosshair placement
- Si le winrate est faible (<45%), focus sur les rotations et le jeu d'équipe
"""

    content = []

    content.append({
        "type": "text",
        "text": f"""Tu es un coach professionnel esport spécialisé sur {game} avec 10 ans d'experience.
Tu analyses une replay d'un joueur qui veut progresser.
Tu vas recevoir {len(frames_b64)} captures d'ecran extraites de la partie.

{riot_context}

REGLES STRICTES DE COACHING :

1. JAMAIS de description — tu ne racontes pas ce que tu vois, tu coaches.
   Mauvais : "On voit le joueur en position mid"
   Bon : "Ta position mid t'expose a 3 angles — recule derriere le pilier et jiggle peek"

2. Chaque critique = erreur precise + correction immediate + pourquoi c'est important
   Format : "Tu fais X, fais Y a la place parce que Z"

3. Chaque alerte = mauvaise habitude + exercice concret pour la corriger
   Format : "Tu as tendance a X, entraine-toi a Y pendant tes deathmatch"

4. Chaque info = point fort observe + comment l'exploiter encore plus
   Format : "Ton X est solide, exploite-le davantage en faisant Y"

5. Si tu as des donnees Riot reelles, utilise-les pour personnaliser chaque module.
   Ex: "Ton HS% de {riot_stats.get('hs_percent', '?')}% montre que..."

6. Utilise le vocabulaire specifique de {game} :
   Valorant : spike, site, eco, full buy, ult, flash, jiggle peek, crosshair placement, off-angle

Voici les captures de la replay :"""
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

Genere le rapport de coaching. Reponds UNIQUEMENT avec ce JSON valide :

{{
  "score_global": <entier entre 35 et 90>,
  "jeu": "{game}",
  "resume": "<2 phrases de bilan direct basees sur les vraies stats et la replay>",
  "riot_stats": {{
    "kda": "{riot_stats.get('kda', 'N/A')}",
    "winrate": "{riot_stats.get('winrate', 'N/A')}%",
    "hs_percent": "{riot_stats.get('hs_percent', 'N/A')}%",
    "matches": "{riot_stats.get('matches_analyzed', 'N/A')}"
  }},
  "modules": [
    {{
      "id": "decision",
      "label": "Prise de decision",
      "score": <entier 0-100>,
      "niveau": "<Faible|Moyen|Bon|Excellent>",
      "points": [
        {{"type": "critique", "texte": "<Tu fais X, fais Y parce que Z>"}},
        {{"type": "alerte", "texte": "<Tu as tendance a X, corrige en faisant Y>"}},
        {{"type": "info", "texte": "<Ton X est solide, exploite-le en faisant Y>"}}
      ]
    }},
    {{
      "id": "mecanique",
      "label": "Mecanique et Aim",
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
    "<Exercice 1 : action concrete + duree + objectif precis>",
    "<Exercice 2 : action concrete + duree + objectif precis>",
    "<Exercice 3 : action concrete + duree + objectif precis>"
  ],
  "point_fort": "<ce que tu as vraiment vu de positif>",
  "priorite": "<l'erreur qui te coute le plus et comment la corriger>"
}}"""
    })

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2500,
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
