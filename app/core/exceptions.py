from fastapi import HTTPException


class ServiceError(Exception):
    """Base for external-dependency failures that map cleanly onto HTTP errors.

    Carries a user-safe ``detail``; technical context belongs in the logs, not
    in the response body.
    """

    def __init__(self, message: str, *, status_code: int = 502, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail or message

    def to_http_exception(self) -> HTTPException:
        return HTTPException(status_code=self.status_code, detail=self.detail)


class OllamaServiceError(ServiceError):
    """Raised when the Ollama service is unavailable or returns an error."""


class ComfyUIServiceError(ServiceError):
    """Raised when ComfyUI is unreachable, times out, or rejects a workflow."""


class WorkflowError(ServiceError):
    """Raised when a workflow file is missing, malformed, or badly mapped."""

    def __init__(self, message: str, *, status_code: int = 400, detail: str | None = None) -> None:
        super().__init__(message, status_code=status_code, detail=detail)


class AIToolkitError(ServiceError):
    """Raised when the Ostris AI Toolkit is unconfigured or a training run fails."""

    def __init__(self, message: str, *, status_code: int = 503, detail: str | None = None) -> None:
        super().__init__(message, status_code=status_code, detail=detail)


class LoRAProjectError(ServiceError):
    """Raised for invalid LoRA project / dataset operations."""

    def __init__(self, message: str, *, status_code: int = 400, detail: str | None = None) -> None:
        super().__init__(message, status_code=status_code, detail=detail)
