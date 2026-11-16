"""Custom middleware."""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_BODY_METHODS = {"POST", "PUT", "PATCH"}


class BodySizeLimitMiddleware:
    """Reject requests whose declared body size is missing or too large,
    before the body is read into memory."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        error: tuple[int, str] | None = None
        raw_length = dict(scope["headers"]).get(b"content-length")

        if raw_length is None:
            if scope["method"] in _BODY_METHODS:
                error = (411, "Content-Length header is required.")
        else:
            try:
                length = int(raw_length)
            except ValueError:
                length = -1
            if length < 0:
                error = (400, "Invalid Content-Length header.")
            elif length > self.max_bytes:
                error = (413, "Request body too large.")

        if error is not None:
            response = JSONResponse(status_code=error[0], content={"detail": error[1]})
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)