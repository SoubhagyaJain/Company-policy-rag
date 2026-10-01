from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.api.dependencies import get_storage_service
from backend.services.storage_service import (
    StorageBusyError,
    StorageService,
    UnknownStorageActionError,
)

router = APIRouter(tags=["Admin Storage"])


class StorageActionRequest(BaseModel):
    older_than_days: int | None = Field(default=None, ge=1)
    target: str | None = Field(default=None, max_length=200)


@router.get("/api/admin/storage")
def get_storage_summary(
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """Every store the app writes to or holds in memory, with sizes and available cleanups."""
    return storage_service.summary()


@router.get("/api/admin/storage/history")
def get_storage_history(
    limit: int = Query(default=500, ge=1, le=2000),
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """Size snapshots over time and the log of cleanups that were run."""
    return storage_service.history(limit=limit)


@router.post("/api/admin/storage/cleanup")
def clean_safe_storage(
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """Remove orphaned data and compact databases; nothing the app still uses is deleted."""
    return storage_service.clean_safe()


@router.post("/api/admin/storage/{store_id}/{action_id}")
def run_storage_action(
    store_id: str,
    action_id: str,
    request: StorageActionRequest | None = None,
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """Run one cleanup action on one store."""
    body = request or StorageActionRequest()
    try:
        return storage_service.run_action(
            store_id,
            action_id,
            older_than_days=body.older_than_days,
            target=body.target,
        )
    except UnknownStorageActionError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except StorageBusyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage action failed: {exc!s}",
        ) from exc
