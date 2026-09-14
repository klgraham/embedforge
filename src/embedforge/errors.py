"""User-facing EmbedForge errors."""


class EmbedForgeError(Exception):
    """An error that the CLI prints without a traceback."""


class SecretInConfigError(EmbedForgeError):
    """Raised when a caller tries to store credentials in config."""

    def __init__(self, env_name: str) -> None:
        self.env_name = env_name
        super().__init__(
            "API credentials must be supplied through environment variables.\n"
            f"Set {env_name} in your shell environment."
        )
