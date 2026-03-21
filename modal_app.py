"""Modal deployment entrypoint for OptiPrompt FastAPI service."""

import modal

from app.main import app as fastapi_app

image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements(
    "requirements.txt"
)

app = modal.App("optiprompt")


@app.function(image=image)
@modal.asgi_app()
def fastapi_app_entry():
    """Expose the FastAPI app through Modal ASGI integration."""
    return fastapi_app
