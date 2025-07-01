import asyncio
import json
import logging
import os
import re
import sys
from typing import Any, AsyncGenerator, Union

from fastapi.responses import StreamingResponse
import litellm
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langfuse import Langfuse


PROMPTS_TO_SKIP = [
    "Analyze this message and come up with a single positive, cheerful and delightful verb in gerund form that's "
    "related to the message. Only include the word with no other text or punctuation. The word should have the first "
    "letter capitalized. Add some whimsy and surprise to entertain the user. Ensure the word is highly relevant to "
    "the user's message. Synonyms are welcome, including obscure words. Be careful to avoid words that might look "
    "alarming or concerning to the software engineer seeing it as a status notification, such as Connecting, "
    "Disconnecting, Retrying, Lagging, Freezing, etc. NEVER use a destructive word, such as Terminating, Killing, "
    "Deleting, Destroying, Stopping, Exiting, or similar. NEVER use a word that may be derogatory, offensive, or "
    "inappropriate in a non-coding context, such as Penetrating.",
]


# Load environment variables from .env file
load_dotenv()

# Set Langfuse environment variables (will be read from .env if present)
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Initialize Langfuse client if credentials are available
langfuse_client = None
langfuse_trace = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    langfuse_client = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)

    # Create a trace
    langfuse_trace = langfuse_client.trace(name="proxy_session")


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
        handler.setFormatter(ColorizedFormatter("%(asctime)s - %(levelname)s - %(message)s"))

app = FastAPI()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


async def capture_streaming_output(stream: AsyncGenerator) -> tuple[AsyncGenerator, asyncio.Future]:
    """Tee the streaming response to capture output while allowing passthrough."""
    captured_chunks = []
    future = asyncio.Future()

    async def tee_generator():
        try:
            async for chunk in stream:
                captured_chunks.append(chunk)
                yield chunk
        finally:
            # Set the future result when streaming is complete
            future.set_result(captured_chunks)

    return tee_generator(), future


def merge_anthropic_streaming_dicts_recursively(target: dict, source: dict, dict_path: tuple = ()) -> None:
    """Merge source dict into target dict recursively with Anthropic streaming protocol support."""

    for key, value in source.items():
        current_path = dict_path + (key,)

        # Special Anthropic streaming protocol handling
        if current_path == ("type",):
            continue
        elif current_path == ("index",):
            continue
        elif current_path == ("content_block",):
            # Initialize content block structure as dict with int indices
            if "content" not in target:
                target["content"] = {}

            index = source.get("index", 0)
            # Set the initial content block at this index
            target["content"][index] = value

            input_value = target["content"][index].get("input")
            if input_value == {}:
                # For some reason when the start of the content_block is streamed an empty dict is put in the input
                # field. We need to replace it with an empty string, so it is possible to accumulate partial_json
                # strings there.
                target["content"][index]["input"] = ""
            continue
        elif current_path == ("delta",):
            # Handle delta updates for streaming content
            if isinstance(value, dict):
                delta_type = value.get("type")
                index = source.get("index", 0)

                if delta_type == "text_delta":
                    # Ensure content dict exists and has this index
                    if "content" not in target:
                        target["content"] = {}
                    if index not in target["content"]:
                        target["content"][index] = {"type": "text", "text": ""}

                    # Append text delta
                    if "text" in value:
                        target["content"][index]["text"] += value["text"]

                elif delta_type == "input_json_delta":
                    # Handle tool input JSON delta
                    if "content" not in target:
                        target["content"] = {}
                    if index not in target["content"]:
                        target["content"][index] = {"type": "tool_use", "input": ""}

                    # Append partial JSON
                    if "partial_json" in value:
                        if "input" not in target["content"][index]:
                            target["content"][index]["input"] = ""
                        try:
                            target["content"][index]["input"] += value["partial_json"]
                        except:
                            from pprint import pprint

                            pprint(target["content"][index])
                            pprint(value)
                            raise
            continue

        # Regular merging for non-Anthropic streaming keys
        if key in target:
            existing = target[key]
            if isinstance(existing, dict) and isinstance(value, dict):
                merge_anthropic_streaming_dicts_recursively(existing, value, current_path)
            elif isinstance(existing, list) and isinstance(value, list):
                existing.extend(value)
            elif isinstance(existing, list):
                existing.append(value)
            else:
                target[key] = [existing, value]
        else:
            target[key] = value


def reconstruct_message_from_chunks(captured_output: list[bytes]) -> dict[str, Any]:
    """Reconstruct a complete message from streaming chunks by merging all JSON data."""
    merged_data = {}

    for chunk in captured_output:
        chunk_str = chunk.decode("utf-8")

        # Parse each event in the chunk
        events = re.findall(r"event:\s*\w+\s*\ndata:\s*({.*?})\s*\n", chunk_str, re.DOTALL)

        for data_str in events:
            try:
                data = json.loads(data_str.strip())
                merge_anthropic_streaming_dicts_recursively(merged_data, data)
            except json.JSONDecodeError as e:
                logger.error("JSONDecodeError: %s\n\nFULL CHUNK JSON: %s", e, data_str)
                continue

    # Post-process: Convert content dict to sorted list and parse JSON strings
    if "content" in merged_data and isinstance(merged_data["content"], dict):
        # Convert content dict to sorted list
        content_blocks = []
        for index in sorted(merged_data["content"].keys()):
            content_block = merged_data["content"][index]

            # Parse tool input JSON if it's a string
            if content_block.get("type") == "tool_use" and "input" in content_block:
                if isinstance(content_block["input"], str):
                    try:
                        content_block["input"] = json.loads(content_block["input"])
                    except json.JSONDecodeError as e:
                        logger.error("Failed to parse tool input JSON: %s\n\nJSON: %s", e, content_block["input"])

            content_blocks.append(content_block)

        merged_data["content"] = content_blocks

    return merged_data


async def trace_to_langfuse(
    request_data: dict[str, Any],
    response_data: Union[dict[str, Any], asyncio.Future],
) -> None:
    """Trace request and response to Langfuse asynchronously."""
    if not langfuse_client:
        return

    # Prepare metadata with nested structure preserved
    # Create a copy of request data for Langfuse logging
    langfuse_request = request_data.copy()

    # Remove sensitive data before logging
    langfuse_request.pop("api_key", None)

    # Handle system message for Langfuse - move it to the beginning of messages
    if "system" in langfuse_request and langfuse_request["system"]:
        system_content = langfuse_request.pop("system")
        if (
            isinstance(system_content, (list, tuple))
            and "text" in system_content[0]  # We already checked if it's not empty with the outer if-statement
            and system_content[0]["text"] in PROMPTS_TO_SKIP
        ):
            # Let's unclutter the logs by skipping non-useful prompts
            return

        messages = langfuse_request.get("messages", []).copy()

        # Prepend system message to the messages array
        system_message = {"role": "system", "content": system_content}
        messages.insert(0, system_message)
        langfuse_request["messages"] = messages

    metadata = {"request": {k: v for k, v in langfuse_request.items() if k != "messages"}}

    is_streaming = isinstance(response_data, asyncio.Future)

    # Handle response data based on streaming mode
    if is_streaming:
        # Await for the captured streaming chunks
        response_data = await response_data  # It's a Future and it will contain all the captured chunks in a list

        # Reconstruct the complete message from chunks
        langfuse_response = reconstruct_message_from_chunks(response_data)
    else:
        langfuse_response = response_data.copy()

    trace_output = {
        "role": langfuse_response.pop("role", None),
        "content": langfuse_response.pop("content", None),
    }
    metadata["response"] = langfuse_response

    # Create generation span
    langfuse_trace.generation(
        name="litellm_anthropic_call",
        model=request_data.get("model", "unknown"),
        input=langfuse_request.get("messages", []),
        output=trace_output,
        metadata=metadata,
    )

    # Flush to ensure data is sent
    langfuse_client.flush()


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

    # Handle streaming responses with capture for Langfuse
    is_streaming = request.get("stream", False)
    if is_streaming:
        # Create tee to capture streaming output
        tee_stream, captured_chunks_future = await capture_streaming_output(response)

        # Start Langfuse tracing in background task with captured output future
        if langfuse_client:
            asyncio.create_task(trace_to_langfuse(request, captured_chunks_future))

        return StreamingResponse(
            tee_stream,
            media_type="text/event-stream",
        )
    else:
        # For non-streaming, trace immediately
        if langfuse_client:
            asyncio.create_task(trace_to_langfuse(request, response))

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
