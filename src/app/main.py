import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Resolve project paths and append to python search paths dynamically
api_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(api_dir, "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from app.services.registry import services
from app.api.routes import router as api_router

app = FastAPI(
    title="TasteFinder AI - Zomato",
    description="Zero-Scroll, Single-Screen Context-Aware Zomato Restaurant Discovery Engine.",
    version="1.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(api_router)

# Serve static frontend web assets
frontend_dir = os.path.join(project_root, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def read_root():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "TasteFinder AI frontend index.html missing."}

# Startup Event to pre-warm the service registry
@app.on_event("startup")
def startup_event():
    print("FastAPI application is starting up...")
    services.initialize()

if __name__ == "__main__":
    import uvicorn
    # Start ASGI server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
