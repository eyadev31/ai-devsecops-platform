def run_profile_analysis(profile: dict) -> dict:
    risk = profile.get("risk_level", "").lower()
    horizon = profile.get("horizon", "").lower()
    objective = profile.get("objective", "").lower()

    score = 5.0
    investor_type = "Modéré"

    if "faible" in risk or "prudent" in risk:
        score = 3.0
        investor_type = "Prudent"
    elif "mod" in risk:
        score = 6.0
        investor_type = "Modéré"
    elif "élev" in risk or "agress" in risk:
        score = 8.5
        investor_type = "Agressif"

    details = [
        f"Niveau de risque détecté : {profile.get('risk_level', 'N/A')}",
        f"Horizon détecté : {profile.get('horizon', 'N/A')}",
        f"Objectif détecté : {profile.get('objective', 'N/A')}",
    ]

    if "long" in horizon:
        details.append("Horizon long terme compatible avec une stratégie de croissance.")
    if "divers" in objective:
        details.append("Objectif orienté diversification.")
    if "croissance" in objective:
        details.append("Objectif orienté rendement / croissance.")

    return {
        "agent_name": "Profiling Agent",
        "score": score,
        "summary": f"Profil investisseur classé comme {investor_type}.",
        "details": details,
        "investor_type": investor_type,
    }
