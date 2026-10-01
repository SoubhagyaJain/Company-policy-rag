from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.api.dependencies import get_storage_service
from backend.services.storage_service import (
    StorageBusyError,
    StorageService,
    UnknownStorageActionError,
)

router = APIRouter(tags=["Admin Storage"])

# Stores and actions are addressed by logical id only; the service owns every path.
_ID = r"^[a-z0-9_]{1,40}$"


class StorageActionRequest(BaseModel):
    older_than_days: int | None = Field(default=None, ge=1)
    target: str | None = Field(default=None, max_length=200)


class CleanupItem(BaseModel):
    store_id: str = Field(pattern=_ID)
    action_id: str = Field(pattern=_ID)
    older_than_days: int | None = Field(default=None, ge=1)


class CleanupRequest(BaseModel):
    items: list[CleanupItem] = Field(max_length=50)


def _run(call: Any) -> dict[str, Any]:
    """Map the service's errors onto HTTP statuses."""
    try:
        return call()
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


@router.get("/api/admin/storage")
def get_storage_summary(
    refresh: bool = Query(default=False, description="Drop cached directory sizes and scan again."),
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """Every store the app writes to or holds in memory, with sizes and available cleanups."""
    return storage_service.summary(refresh=refresh)


@router.get("/api/admin/storage/live")
def get_storage_live(
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """RAM, VRAM, loaded models, cache counters and running operations. Cheap enough to poll."""
    return storage_service.live()


@router.get("/api/admin/storage/documents")
def get_storage_documents(
    deep: bool = Query(default=False, description="Also scan the vector rows for orphaned chunks."),
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """Storage footprint per document and the artifacts whose document is gone."""
    return storage_service.documents(deep=deep)


@router.get("/api/admin/storage/history")
def get_storage_history(
    limit: int = Query(default=500, ge=1, le=6000),
    window: Literal["24h", "7d", "30d"] | None = Query(default=None, alias="range"),
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """Size snapshots over time and the log of cleanups; with ``range`` also chart series, events and a forecast."""
    return storage_service.history(limit=limit, window=window)


@router.get("/api/admin/storage/cleanup/plan")
def get_cleanup_plan(
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """Every cleanup on offer, with what it would reclaim and what it costs."""
    return storage_service.cleanup_plan()


@router.post("/api/admin/storage/cleanup")
def clean_storage(
    request: CleanupRequest | None = None,
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """Run the chosen cleanups. Without a body: remove orphaned data and compact the databases."""
    items = [item.model_dump() for item in request.items] if request is not None else None
    return _run(lambda: storage_service.clean_safe(items))


@router.get("/api/admin/storage/{store_id}/inspect")
def inspect_store(
    store_id: str,
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """Internals of one store: collections, tables, largest files and where it lives."""
    return _run(lambda: storage_service.inspect(store_id))


@router.post("/api/admin/storage/{store_id}/{action_id}/preview")
def preview_storage_action(
    store_id: str,
    action_id: str,
    request: StorageActionRequest | None = None,
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """What an action would remove, without running it."""
    body = request or StorageActionRequest()
    return _run(lambda: storage_service.preview(store_id, action_id, older_than_days=body.older_than_days))


@router.post("/api/admin/storage/{store_id}/{action_id}")
def run_storage_action(
    store_id: str,
    action_id: str,
    request: StorageActionRequest | None = None,
    storage_service: StorageService = Depends(get_storage_service),
) -> dict[str, Any]:
    """Run one cleanup action on one store."""
    body = request or StorageActionRequest()
    return _run(
        lambda: storage_service.run_action(
            store_id,
            action_id,
            older_than_days=body.older_than_days,
            target=body.target,
        )
    )
