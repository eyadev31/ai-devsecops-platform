from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="user_service")

class UserProfile(BaseModel):
    id: int
    name: str
    risk_level: str
    horizon: str
    objective: str

users_db = {
    1: {
        "id": 1,
        "name": "Eya Khalfallah",
        "risk_level": "Modérée",
        "horizon": "Long terme",
        "objective": "Croissance et diversification"
    }
}

@app.get("/health")
def health():
    return {"status": "user service running"}

@app.get("/profile/{user_id}")
def get_profile(user_id: int):
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    return user

@app.post("/profile")
def create_profile(profile: UserProfile):
    users_db[profile.id] = profile.dict()
    return {
        "message": "Profil créé avec succès",
        "profile": profile
    }

