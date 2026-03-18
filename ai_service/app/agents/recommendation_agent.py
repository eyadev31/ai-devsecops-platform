def run_recommendation(profile_result: dict, portfolio: dict) -> dict:
    assets = portfolio.get("assets", [])
    recommendations = []

    crypto = next((a["percentage"] for a in assets if "crypto" in a["name"].lower()), 0)
    cash = next((a["percentage"] for a in assets if "cash" in a["name"].lower()), 0)
    etf = next((a["percentage"] for a in assets if "etf" in a["name"].lower()), 0)

    if crypto > 15:
        recommendations.append("Réduire légèrement l’exposition crypto pour limiter la volatilité.")
    if etf < 25:
        recommendations.append("Augmenter les ETF pour renforcer la diversification.")
    if cash < 8:
        recommendations.append("Conserver une poche de liquidité plus confortable.")
    if not recommendations:
        recommendations.append("Maintenir l’allocation actuelle avec rééquilibrage périodique.")

    return {
        "agent_name": "Recommendation Agent",
        "score": 8.0,
        "summary": "Recommandations générées selon le profil et la composition actuelle.",
        "details": recommendations,
        "recommendations": recommendations,
    }
