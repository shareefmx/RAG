"""Hugging Face Spaces Entrypoint.

Launches the PDF Knowledge Assistant Gradio interface.
"""

from app.config import get_settings, setup_logging
from app.ui.gradio_app import create_ui


if __name__ == "__main__":
    settings = get_settings()
    setup_logging(settings.log_level)
    demo = create_ui()
    demo.launch(
        server_name=settings.app_host,
        server_port=7860,  # Standard Hugging Face Spaces port
        share=False,
    )

