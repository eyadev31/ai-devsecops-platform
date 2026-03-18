def run_portfolio_analysis(portfolio: dict) -> dict:
    assets = portfolio.get("assets", [])
    num_assets = len(assets)

    max_weight = max((asset["percentage"] for asset in assets), default=0)
    diversification_score = min(10.0, num_assets * 1.8)

    details = [
        f"Nombre d'actifs : {num_assets}",
        f"Poids maximal observé : {max_weight}%",
    ]

    if max_weight >= 50:
        diversification_score -= 3
        details.append("Concentration élevée détectée.")
    elif max_weight >= 40:
        diversification_score -= 1.5
        details.append("Concentration modérée détectée.")
    else:
        details.append("Répartition globalement équilibrée.")

    diversification_score = max(1.0, round(diversification_score, 1))

    return {
        "agent_name": "Portfolio Analysis Agent",
        "score": diversification_score,
        "summary": f"Score de diversification estimé à {diversification_score}/10.",
        "details": details,
        "diversification_score": diversification_score,
    }
