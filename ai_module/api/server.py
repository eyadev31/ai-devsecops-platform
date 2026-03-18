"""
Hybrid Intelligence Portfolio System — FastAPI Server
=====================================================
REST API that exposes the 4-agent ML/LLM pipeline as a web service.

Usage:
  uvicorn api.server:app --reload --port 8000
"""

import logging
import os
import sys

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.auth import router as auth_router
from api.routes.daq import router as daq_router
from api.routes.portfolio import router as portfolio_router
from api.routes.dashboard import router as dashboard_router
from api.routes.rebalance import router as rebalance_router
from api.routes.news import router as news_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="Hybrid Intelligence Portfolio System",
    description=(
        "AI-powered portfolio management with 4-agent ML/LLM pipeline. "
        "Connects to Binance for real portfolio tracking and trading."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Next.js dev
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://localhost:8081",   # Expo dev server
        "http://localhost:19006",  # Expo web
        "exp://192.168.*.*:*",     # Expo Go on local network
        "*",  # Allow all in dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routes ──────────────────────────────────────────
app.include_router(auth_router)
app.include_router(daq_router)
app.include_router(portfolio_router)
app.include_router(dashboard_router)
app.include_router(rebalance_router)
app.include_router(news_router)


# ── Root ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "system": "Hybrid Intelligence Portfolio System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/api/docs",
            "auth": "/api/auth",
            "daq": "/api/daq",
            "portfolio": "/api/portfolio",
            "dashboard": "/api/dashboard",
            "news": "/api/news",
            "rebalance": "/api/portfolio/rebalance",
        },
    }


@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "portfolio-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
