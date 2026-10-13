from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api.routes import router
from app.core.config import get_settings, BASE_DIR
from app.core.logging import configure_logging
from app.services.audit import init_audit_db

configure_logging()
settings = get_settings()
init_audit_db()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise HR Policy & Employee Support Copilot powered by LangGraph, Pinecone, and Groq.",
)

app.include_router(router)

# Mount static assets and Jinja2 templates if directories exist
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "app_name": settings.app_name,
                "version": settings.app_version,
            },
        )
