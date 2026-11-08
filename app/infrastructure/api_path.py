from starlette.types import ASGIApp, Receive, Scope, Send


class NormalizeApiPathMiddleware:
    """У /api-запроса срезает хвостовой слэш до выбора маршрута.

    Иначе адрес вроде /api/save-chapter/ забирает статика и отвечает 405.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope.get("path") or ""
            if path.startswith("/api/") and path.endswith("/"):
                trimmed = path.rstrip("/")
                if trimmed:
                    scope = dict(scope)
                    scope["path"] = trimmed
                    raw_path = scope.get("raw_path")
                    if isinstance(raw_path, (bytes, bytearray)):
                        text = raw_path.decode("latin1")
                        _, separator, query = text.partition("?")
                        rewritten = trimmed + (separator + query if separator else "")
                        scope["raw_path"] = rewritten.encode("latin1")
        await self.app(scope, receive, send)
