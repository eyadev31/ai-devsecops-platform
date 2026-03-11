from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="portfolio_service")

class Asset(BaseModel):
    name: str
    percentage: int

class PortfolioResponse(BaseModel):
    user_id: int
    assets: List[Asset]

portfolios_db = {
    1: {
        "user_id": 1,
        "assets": [
            {"name": "Actions US", "percentage": 40},
            {"name": "ETF", "percentage": 20},
            {"name": "Crypto", "percentage": 20},
            {"name": "Cash", "percentage": 10},
            {"name": "Or", "percentage": 10}
        ]
    }
}

recommendations_db = {
    1: {
        "summary": "Portefeuille diversifié, profil modéré, exposition équilibrée entre actions, ETF, crypto et cash.",
        "agents": [
            {
                "name": "Agent 1 - Macro Context Analyzer",
                "description": "Analyse du contexte macroéconomique et du marché"
            },
            {
                "name": "Agent 2 - User Behavior Profiler",
                "description": "Analyse du profil investisseur"
            },
            {
                "name": "Agent 3 - Portfolio Allocation Optimizer",
                "description": "Optimisation de l’allocation du portefeuille"
            },
            {
                "name": "Agent 4 - Risk Oversight & Validation",
                "description": "Contrôle du risque et validation"
            },
            {
                "name": "Agent 5 - News Sentiment Agent",
                "description": "Analyse des actualités et du sentiment"
            }
        ]
    }
}

@app.get("/health")
def health():
    return {"status": "portfolio service running"}

@app.get("/user/{user_id}")
def get_portfolio_by_user(user_id: int):
    portfolio = portfolios_db.get(user_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portefeuille non trouvé")
    return portfolio

@app.get("/recommendation/{user_id}")
def get_recommendation(user_id: int):
    recommendation = recommendations_db.get(user_id)
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommandation non trouvée")
    return recommendation

