"""
FastAPI backend for BookClaw LangChain Agent.

Usage:
    uv run python server.py
    or
    uv run uvicorn server:app --reload
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from agent.graph import AgentGraph
from agent.tools import GetWeatherTool, CurrentTimeTool


# Paths
BOOKS_DIR = Path(__file__).parent / "books"


# Global agent instance
agent: AgentGraph = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manage agent lifecycle."""
    global agent
    # Startup: initialize agent
    agent = AgentGraph(
        mcp_config_path="mcp_config.yaml",
        native_tools=[GetWeatherTool(), CurrentTimeTool()],
    )
    await agent.initialize()
    print("Agent initialized successfully")
    yield
    # Shutdown: close agent
    await agent.close()
    print("Agent closed")


app = FastAPI(lifespan=lifespan)


# Request/Response models
class ChatRequest(BaseModel):
    messages: list[dict] = []
    user_input: str
    selected_book: str = None  # 可选，限定在某本书内搜索


class ChatResponse(BaseModel):
    response: str
    messages: list[dict]


def serialize_message(msg) -> dict:
    """Convert LangChain message to dict for JSON serialization."""
    if isinstance(msg, HumanMessage):
        return {"type": "human", "content": msg.content}
    elif isinstance(msg, AIMessage):
        return {"type": "ai", "content": msg.content}
    elif isinstance(msg, SystemMessage):
        return {"type": "system", "content": msg.content}
    elif isinstance(msg, ToolMessage):
        return {"type": "tool", "content": msg.content, "name": msg.name}
    else:
        return {"type": "unknown", "content": str(msg.content)}


def deserialize_messages(messages: list[dict]) -> list:
    """Convert dict messages to LangChain message objects."""
    result = []
    for msg in messages:
        msg_type = msg.get("type", "human")
        content = msg.get("content", "")
        if msg_type == "human":
            result.append(HumanMessage(content=content))
        elif msg_type == "ai":
            result.append(AIMessage(content=content))
        elif msg_type == "system":
            result.append(SystemMessage(content=content))
        # Skip tool messages from client
    return result


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle chat request.

    Args:
        request: Contains messages history and new user input

    Returns:
        response: Agent's response text
        messages: Updated messages list
    """
    # Deserialize messages from client
    messages = deserialize_messages(request.messages)

    # Add new user message
    messages.append(HumanMessage(content=request.user_input))

    # If a book is selected, add context hint for the agent
    if request.selected_book:
        from langchain_core.messages import SystemMessage
        book_hint = SystemMessage(
            content=f"[Context] 用户当前选中的图书: {request.selected_book}。如果需要检索知识库，优先在此书范围内搜索。"
        )
        messages.append(book_hint)

    # Run agent
    response, updated_messages = await agent.run(messages)

    # Remove the book hint from returned messages (client doesn't need it)
    updated_messages = [m for m in updated_messages if not (
        isinstance(m, SystemMessage) and m.content.startswith("[Context]")
    )]

    # Serialize messages for response
    serialized = [serialize_message(m) for m in updated_messages]

    return ChatResponse(response=response, messages=serialized)


@app.get("/books")
async def list_books():
    """List all books in the books directory."""
    books = []
    if BOOKS_DIR.exists():
        for file in BOOKS_DIR.iterdir():
            if file.is_file() and file.suffix.lower() == ".pdf":
                # Format file size
                size = file.stat().st_size
                if size >= 1024 * 1024 * 1024:
                    size_str = f"{size / (1024*1024*1024):.1f} GB"
                elif size >= 1024 * 1024:
                    size_str = f"{size / (1024*1024):.1f} MB"
                else:
                    size_str = f"{size / 1024:.1f} KB"

                books.append({
                    "name": file.stem,  # filename without extension
                    "filename": file.name,
                    "size": size_str,
                })
    return {"books": books}


# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Redirect to static index page."""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)