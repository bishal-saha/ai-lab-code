"""A thin client for the Kestrel Outfitters mock API.

Both MCP servers go through this client and nothing else, so neither tool design
sees data the other can't: the only difference between them is the tool surface.
"""

from __future__ import annotations

from typing import Any

import httpx


class ApiNotFound(Exception):
    """The API answered 404; the message is the API's own error text."""


class KestrelAPI:
    def __init__(self, base_url: str, *, timeout: float = 10.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def get(self, path: str, **params: Any) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        response = self._client.get(path, params=clean)
        if response.status_code == 404:
            raise ApiNotFound(response.json().get("error", f"not found: {path}"))
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._client.close()
