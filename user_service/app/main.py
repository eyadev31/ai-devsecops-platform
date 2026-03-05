from fastapi import FastAPI

app = FastAPI(
    title="user_service",
    version="1.0.0",
    root_path="/users"
)

@app.get("/health")
def health():
    return {"status": "user service running"}

