import logging
import os
import sys
from typing import Any

import litellm
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARN,  # Change to INFO level to show more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.DEBUG and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)


# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(
            ColorizedFormatter("%(asctime)s - %(levelname)s - %(message)s")
        )

app = FastAPI()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


@app.post("/v1/messages")
async def create_message(request: dict[str, Any], raw_request: Request) -> Any:
    # Get the display name for logging, just the model name without provider prefix
    display_model = request.get("model", "unknown")
    if "/" in display_model:
        display_model = display_model.split("/")[-1]

    logger.debug(
        "📊 PROCESSING REQUEST: Model=%s, Stream=%s",
        request.get("model"),
        request.get("stream", False),
    )

    # Add API key to request
    request["api_key"] = ANTHROPIC_API_KEY
    logger.debug("Using Anthropic API key for model: %s", request.get("model"))

    # Only log basic info about the request, not the full details
    logger.debug(
        "Request for model: %s, stream: %s",
        request.get("model"),
        request.get("stream", False),
    )

    num_tools = len(request.get("tools", []))
    num_messages = len(request.get("messages", []))

    log_request_beautifully(
        "POST",
        raw_request.url.path,
        display_model,
        num_messages,
        num_tools,
        200,  # Assuming success at this point
    )

    # Use LiteLLM's native Anthropic format support
    response = await litellm.anthropic.messages.acreate(**request)

    # Handle streaming responses
    if request.get("stream", False):
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )

    return response


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Anthropic Proxy for LiteLLM"}


# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"


def log_request_beautifully(
    method: str,
    path: str,
    claude_model: str,
    num_messages: int,
    num_tools: int,
    status_code: int,
) -> None:
    """Log requests in a beautiful, twitter-friendly format."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"

    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]

    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"

    # Format status code
    status_str = (
        f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}"
        if status_code == 200
        else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    )

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} {tools_str} {messages_str}"

    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)

    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
