from fastapi import FastAPI

app = FastAPI(
    title="portfolio_service",
    version="1.0.0",
    root_path="/portfolio"
)

@app.get("/health")
def health():
    return {"status": "portfolio service running"}

@app.get("/portfolios")
def portfolios():
    return [
        {"id": 1, "user_id": 1, "asset": "AAPL", "amount": 10},
        {"id": 2, "user_id": 1, "asset": "BTC", "amount": 0.5},
    ]
