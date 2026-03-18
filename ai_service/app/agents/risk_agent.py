def run_risk_analysis(portfolio: dict) -> dict:
    assets = portfolio.get("assets", [])
    alerts = []
    risk_score = 5.5

    crypto = next((a["percentage"] for a in assets if "crypto" in a["name"].lower()), 0)
    actions = next((a["percentage"] for a in assets if "action" in a["name"].lower()), 0)

    if crypto >= 20:
        alerts.append("Exposition crypto élevée : volatilité potentiellement importante.")
        risk_score += 1.5

    if actions >= 45:
        alerts.append("Exposition actions significative : sensibilité au marché.")
        risk_score += 1.0

    if not alerts:
        alerts.append("Pas d’alerte critique détectée.")

    risk_score = min(10.0, round(risk_score, 1))

    return {
        "agent_name": "Risk Agent",
        "score": risk_score,
        "summary": f"Score de risque global estimé à {risk_score}/10.",
        "details": alerts,
        "global_risk_score": risk_score,
        "alerts": alerts,
    }
