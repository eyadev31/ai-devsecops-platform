def run_explanation(investor_type: str, diversification_score: float, risk_score: float, recommendations: list[str]) -> dict:
    explanation = (
        f"L’utilisateur présente un profil {investor_type}. "
        f"Le portefeuille montre un score de diversification de {diversification_score}/10 "
        f"et un score de risque global de {risk_score}/10. "
        f"Les recommandations prioritaires sont : {' ; '.join(recommendations)}"
    )

    return {
        "agent_name": "Explanation Agent",
        "score": 9.0,
        "summary": "Explication finale générée pour l’utilisateur.",
        "details": [explanation],
        "explanation": explanation,
    }
