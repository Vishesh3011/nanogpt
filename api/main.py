from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routers import stories, code
from api.inference import load_all_models, is_loaded
from api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: load models once into memory
    print("[startup] loading models...")
    load_all_models()
    print("[startup] ready")
    yield
    # shutdown: nothing to clean up (models released with process)
    print("[shutdown] bye")


app = FastAPI(
    title="nanogpt",
    description="Story generation and Go code generation via two fine-tuned GPT models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(stories.router)
app.include_router(code.router)

# Health check


@app.get("/health", response_model=HealthResponse)
def health():
    stories_ok = is_loaded("stories")
    code_ok = is_loaded("code")
    return HealthResponse(
        status="ok" if (stories_ok and code_ok) else "degraded",
        stories_loaded=stories_ok,
        code_loaded=code_ok,
    )


# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def root():
    return FileResponse("frontend/index.html")