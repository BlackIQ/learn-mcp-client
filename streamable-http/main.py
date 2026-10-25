# Libs
import sys  # System
import asyncio  # AsyncIO
import json  # JSON

from ollama import chat, ChatResponse, Tool, Message  # Ollama

from mcp import Client  # MCP Client
from mcp_types import TextContent  # Text Content

# Application
from core.settings import settings  # Core: Settings

messages: list[Message] = []


async def process_query(client: Client, query: str) -> str | None:
    tool_list = await client.list_tools()

    available_tools = [tool.name for tool in tool_list.tools]

    available_functions = [
        Tool(
            type="function",
            function=Tool.Function(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.input_schema,
            ),
        )
        for tool in tool_list.tools
    ]

    messages.append(
        Message(
            role="user",
            content=query,
        )
    )

    response: ChatResponse = chat(
        model=settings.MODEL,
        messages=messages,
        tools=available_functions,
    )

    assistant_message = response.message
    messages.append(assistant_message)

    tool_calls = assistant_message.tool_calls or []

    if not tool_calls:
        return assistant_message.content

    if not tool_calls:
        return assistant_message.content or ""

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments or {}

        if tool_name not in available_tools:
            tool_content = json.dumps(
                {
                    "success": False,
                    "error": f"Tool {tool_name} not found",
                }
            )
        else:
            result = await client.call_tool(tool_name, arguments)

            text_parts = [
                block.text for block in result.content if isinstance(block, TextContent)
            ]

            tool_content = json.dumps(
                {
                    "success": not result.is_error,
                    "content": "\n".join(text_parts),
                    "structured_content": getattr(result, "structured_content", None),
                },
                ensure_ascii=False,
                default=str,
            )

        messages.append(
            Message(
                role="tool",
                tool_name=tool_name,
                content=tool_content,
            )
        )

    final_response: ChatResponse = chat(
        model=settings.MODEL,
        messages=messages,
        tools=available_functions,
    )

    messages.append(final_response.message)

    return final_response.message.content or ""


async def chat_loop(client: Client) -> None:
    print("\nMCP Client Started!")
    print("Type your queries or 'quit' to exit.")

    while True:
        try:
            query = input("\nUser: ").strip()
        except EOFError:
            break

        if query.lower() == "quit":
            break

        try:
            response = await process_query(client, query)

            print(response)
        except Exception as e:
            print(f"\nError: {e}")


async def main():
    if len(sys.argv) < 1:
        print("Usage: uv run main.py <url_to_server>")

        sys.exit(1)

    server_url = sys.argv[1]

    async with Client(server_url) as client:
        await chat_loop(client)


if __name__ == "__main__":
    asyncio.run(main())
