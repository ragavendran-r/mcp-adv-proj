import asyncio
from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    LoggingMessageNotificationParams,
    TextContent,
    SamplingMessage,
)

from dotenv import load_dotenv

load_dotenv()
anthropic_client = AsyncAnthropic()
model = "claude-sonnet-4-5"

server_params = StdioServerParameters(
    command="uv",
    args=["run", "server.py"],
)


async def chat(input_messages: list[SamplingMessage], max_tokens=4000):
    messages = []
    for msg in input_messages:
        content = msg.content if isinstance(msg.content, list) else [msg.content]
        content_block = content[0]
        if msg.role == "user" and content_block.type == "text":
            content = content_block.text if hasattr(content_block, "text") else str(content_block)
            messages.append({"role": "user", "content": content})
        elif msg.role == "assistant" and content_block.type == "text":
            content = content_block.text if hasattr(content_block, "text") else str(content_block)
            messages.append({"role": "assistant", "content": content})

    response = await anthropic_client.messages.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    text = "".join([p.text for p in response.content if p.type == "text"])
    return text


async def sampling_callback(context: RequestContext, params: CreateMessageRequestParams):
    # Call Claude using the Anthropic SDK
    text = await chat(params.messages)

    return CreateMessageResult(
        role="assistant",
        model=model,
        content=TextContent(type="text", text=text),
    )


async def logging_callback(params: LoggingMessageNotificationParams):
    print(params.data)


async def print_progress_callback(progress: float, total: float | None, message: str | None):
    if total is not None:
        percentage = (progress / total) * 100
        print(f"Progress: {progress}/{total} ({percentage:.1f}%)")
    else:
        print(f"Progress: {progress}")


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=sampling_callback) as session:
            await session.initialize()

            result = await session.call_tool(
                name="summarize",
                arguments={"text_to_summarize": "lots of text"},
            )
            print(result.content)


async def run2():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, logging_callback=logging_callback) as session:
            await session.initialize()

            await session.call_tool(
                name="add",
                arguments={"a": 1, "b": 3},
                progress_callback=print_progress_callback,
            )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run2())
    # asyncio.run(run())
