from fastapi import APIRouter
from app.api.routes import agent
from app.api.routes.rag import document

api_router = APIRouter()

api_router.include_router(agent.router, prefix="/messages")
api_router.include_router(document.router, prefix="/document")
