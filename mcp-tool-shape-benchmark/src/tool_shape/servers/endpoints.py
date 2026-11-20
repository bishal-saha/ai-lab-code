"""Server A, "endpoints": one MCP tool per REST route, returning the API's raw JSON.

This is the design most teams ship first, often generated straight from an OpenAPI
spec: every route becomes a tool, descriptions say what the route returns, and the
model is left to chain calls, join records and apply business rules itself.
The descriptions are accurate and reasonably written, so the comparison is with a
good endpoint wrapper, not a strawman.
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from tool_shape.kestrel_api import ApiNotFound, KestrelAPI


def build_server(api: KestrelAPI) -> MCPServer:
    server = MCPServer(
        "kestrel-endpoints",
        instructions="Read-only access to the Kestrel Outfitters internal API. Each tool is one API route.",
    )

    def call(path: str, **params: object) -> str:
        try:
            return json.dumps(api.get(path, **params))
        except ApiNotFound as err:
            raise ToolError(f"404: {err}") from None

    @server.tool()
    def get_order(order_id: str) -> str:
        """GET /orders/{order_id}. Returns the order record: order_id, customer_id, channel
        ("store" or "trade"), placed_at, status, currency, total and promised_date."""
        return call(f"/orders/{order_id}")

    @server.tool()
    def get_order_shipment(order_id: str) -> str:
        """GET /orders/{order_id}/shipment. Returns the shipment for an order: shipment_id,
        carrier, warehouse, shipped_at, promised_by, delivered_at, status and late.
        404 if the order has not shipped."""
        return call(f"/orders/{order_id}/shipment")

    @server.tool()
    def list_shipments(
        carrier: str | None = None, late: bool | None = None, limit: int = 50
    ) -> str:
        """GET /shipments. Lists shipments, optionally filtered by carrier name and late flag.
        limit defaults to 50 (max 500)."""
        return call(
            "/shipments",
            carrier=carrier,
            late=None if late is None else str(late).lower(),
            limit=limit,
        )

    @server.tool()
    def get_stock(sku: str) -> str:
        """GET /stock/{sku}. Returns sku, product name and on_hand quantity per warehouse."""
        return call(f"/stock/{sku}")

    @server.tool()
    def get_ticket(ticket_id: str) -> str:
        """GET /tickets/{ticket_id}. Returns the support ticket: ticket_id, customer_id, order_id
        (may be null), channel, priority, status, subject and messages."""
        return call(f"/tickets/{ticket_id}")

    @server.tool()
    def list_tickets(status: str | None = None, limit: int = 50) -> str:
        """GET /tickets. Lists tickets, optionally filtered by status (open, pending, solved).
        limit defaults to 50 (max 500)."""
        return call("/tickets", status=status, limit=limit)

    @server.tool()
    def list_policies() -> str:
        """GET /policies. Returns the names of all policy documents."""
        return call("/policies")

    @server.tool()
    def get_policy(name: str) -> str:
        """GET /policies/{name}. Returns one policy document as Markdown."""
        return call(f"/policies/{name}")

    return server
