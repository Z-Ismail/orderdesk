"""How the rest of orderdesk talks to the ERP.

ERPReader / ERPWriter are the interfaces (the "adapter" boundary). HttpERPClient implements them
for our simulated ERP; a real SAP or Sage connector would implement the same methods. The
investigator agent is only ever handed an ERPReader, so it cannot create orders even if
it is tricked into trying.
"""

import time
from typing import Protocol

import httpx

from orderdesk.erp_sim.app import DraftOrder, DraftOrderIn, Price, Product, StockLevel

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ERPError(RuntimeError):
    """The ERP refused the request or stayed unavailable after retries."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ERPReader(Protocol):
    def get_product(self, sku: str) -> Product: ...
    def get_stock(self, sku: str) -> StockLevel: ...
    def get_price(self, customer_id: str, sku: str) -> Price: ...
    def get_draft_order(self, order_id: str) -> DraftOrder: ...


class ERPWriter(ERPReader, Protocol):
    def create_draft_order(self, order: DraftOrderIn, idempotency_key: str) -> DraftOrder: ...


class HttpERPClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 5.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        http: httpx.Client | None = None,
    ) -> None:
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout)
        self._headers = {"X-API-Key": api_key}
        self._max_attempts = max_attempts
        self._backoff = backoff_seconds

    def _request(self, method: str, path: str, **kwargs) -> dict | list:
        """Send a request, retrying timeouts and temporary server errors with backoff.

        Retrying a POST is only safe because create_draft_order always sends an
        Idempotency-Key: the ERP returns the original order instead of making a second one.
        """
        headers = {**self._headers, **kwargs.pop("headers", {})}
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._http.request(method, path, headers=headers, **kwargs)
            except httpx.TransportError as exc:  # timeouts, refused connections
                last_error = exc
            else:
                if response.status_code not in RETRYABLE_STATUS:
                    if response.is_error:
                        detail = response.json().get("detail", response.text)
                        raise ERPError(f"{method} {path}: {detail}", response.status_code)
                    return response.json()
                last_error = ERPError(
                    f"{method} {path}: HTTP {response.status_code}", response.status_code
                )
            if attempt < self._max_attempts:
                time.sleep(self._backoff * 2 ** (attempt - 1))  # 0.5s, 1s, 2s...
        raise ERPError(f"{method} {path} failed after {self._max_attempts} attempts: {last_error}")

    def get_product(self, sku: str) -> Product:
        return Product.model_validate(self._request("GET", f"/products/{sku}"))

    def get_stock(self, sku: str) -> StockLevel:
        return StockLevel.model_validate(self._request("GET", f"/stock/{sku}"))

    def get_price(self, customer_id: str, sku: str) -> Price:
        return Price.model_validate(
            self._request("GET", "/prices", params={"customer_id": customer_id, "sku": sku})
        )

    def get_draft_order(self, order_id: str) -> DraftOrder:
        return DraftOrder.model_validate(self._request("GET", f"/draft-orders/{order_id}"))

    def create_draft_order(self, order: DraftOrderIn, idempotency_key: str) -> DraftOrder:
        return DraftOrder.model_validate(
            self._request(
                "POST",
                "/draft-orders",
                json=order.model_dump(mode="json"),
                headers={"Idempotency-Key": idempotency_key},
            )
        )
