"""Single-host administrative boundary, with an optional operator bearer key."""
import hmac
import ipaddress

from fastapi import HTTPException, Request

from src.config import settings


def require_admin(request: Request) -> None:
    if settings.admin_token:
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {settings.admin_token}"):
            raise HTTPException(status_code=403, detail="An operator key is required.")
        return
    try:
        peer = ipaddress.ip_address(request.client.host)
        if any(peer in ipaddress.ip_network(network) for network in settings.admin_trusted_networks):
            return
    except (ValueError, AttributeError):
        pass
    raise HTTPException(status_code=403, detail="Administration is restricted to the local deployment.")
