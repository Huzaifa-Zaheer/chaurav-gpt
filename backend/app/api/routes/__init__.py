from fastapi import APIRouter

from . import documents, system, vectors

api_router = APIRouter()
api_router.include_router(vectors.router)
api_router.include_router(documents.router)
api_router.include_router(system.router)

__all__ = ["api_router"]
