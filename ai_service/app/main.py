from fastapi import FastAPI
from app.services.orchestrator import run_all_agents

app = FastAPI(title="Portfolio Mind AI Service")


@app.get("/health")
def health():
    return {"status": "ai service running"}


@app.get("/recommendation/{user_id}")
def get_ai_recommendation(user_id: int):
    profile = {
        "id": 1,
        "name": "Eya Khalfallah",
        "risk_level": "Modérée",
        "horizon": "Long terme",
        "objective": "Croissance et diversification",
    }

    portfolio = {
        "user_id": 1,
        "assets": [
            {"name": "Actions US", "percentage": 40},
            {"name": "ETF", "percentage": 20},
            {"name": "Crypto", "percentage": 20},
            {"name": "Cash", "percentage": 10},
            {"name": "Or", "percentage": 10},
        ],
    }

    return run_all_agents(profile, portfolio)
