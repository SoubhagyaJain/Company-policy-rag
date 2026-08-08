from backend.api.routes.admin import router as admin_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.documents import router as documents_router
from backend.api.routes.health import router as health_router
from backend.api.routes.models import router as models_router

__all__ = [
    "admin_router",
    "chat_router",
    "documents_router",
    "health_router",
    "models_router",
]
