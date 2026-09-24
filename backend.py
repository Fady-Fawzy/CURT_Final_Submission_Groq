"""FastAPI backend for the CURT inventory assistant."""

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from data_access import get_all_parts
from groq_client import MODEL_NAME, create_groq_client
from tools import AVAILABLE_TOOLS, GROQ_TOOLS


app = FastAPI(title="CURT Inventory Assistant")

# Each key holds a short list of messages for that conversation.
sessions = {}


class ChatRequest(BaseModel):
    """The message and conversation ID sent to the chat endpoint."""

    message: str
    session_id: str

    @field_validator("message", "session_id")
    @classmethod
    def must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("This field must not be blank.")
        return value.strip()


class ChatResponse(BaseModel):
    """The assistant's reply returned by the chat endpoint."""

    reply: str


def _run_requested_tool(function_call):
    """Run a model-requested function only if it is in our allowlist."""
    function = function_call.function
    tool_name = function.name
    tool_function = AVAILABLE_TOOLS.get(tool_name)
    if tool_function is None:
        return {"error": "That tool is not available."}

    argument_name = "category" if tool_name == "list_by_category" else "item_name"
    try:
        arguments = json.loads(function.arguments or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"error": "The tool request had invalid arguments."}
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {argument_name}
        or not isinstance(arguments[argument_name], str)
        or not arguments[argument_name].strip()
    ):
        return {"error": "The tool request had invalid arguments."}

    try:
        return tool_function(**arguments)
    except TypeError:
        return {"error": "The tool request had invalid arguments."}


def _answer_with_groq(message, session_id):
    """Ask Groq, execute approved tool calls in the backend, then get a reply."""
    try:
        client = create_groq_client()
    except RuntimeError as error:
        if "GROQ_API_KEY" not in str(error):
            raise
        raise HTTPException(
            status_code=503,
            detail="Groq is not configured. Set GROQ_API_KEY in the local .env file.",
        ) from error
    messages = [
        {
            "role": "system",
            "content": (
                "You are the CURT inventory assistant. Use tools for inventory facts. "
                "Never guess quantities, locations, or category contents. Use flag_shortage "
                "only when the user asks to report or flag a shortage. If asked whether "
                "stock is enough, report the current quantity and ask how many are needed "
                "before judging."
            ),
        }
    ]
    messages.extend(sessions.get(session_id, []))
    messages.append({"role": "user", "content": message})

    # A tool may lead to another tool request, so allow a few short rounds.
    reply = None
    for _ in range(3):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=GROQ_TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        assistant_message = response.choices[0].message
        function_calls = assistant_message.tool_calls or []
        if not function_calls:
            reply = assistant_message.content or (
                "I couldn't create a reply for that question."
            )
            break

        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": function_call.id,
                        "type": "function",
                        "function": {
                            "name": function_call.function.name,
                            "arguments": function_call.function.arguments,
                        },
                    }
                    for function_call in function_calls
                ],
            }
        )
        for function_call in function_calls:
            print(f"Groq requested tool: {function_call.function.name}")
            result = _run_requested_tool(function_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": function_call.id,
                    "content": json.dumps({"result": result}),
                }
            )

    if reply is None:
        reply = "I couldn't finish processing that question."

    messages = sessions.setdefault(session_id, [])
    messages.append({"role": "user", "content": message})
    messages.append({"role": "assistant", "content": reply})
    return reply


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Answer a chat message using Groq and approved inventory tools."""
    try:
        reply = _answer_with_groq(request.message, request.session_id)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="The assistant could not complete the request. Please try again later.",
        ) from error
    return ChatResponse(reply=reply)


@app.get("/inventory")
def inventory():
    """Return the current inventory as a JSON list."""
    return get_all_parts()
