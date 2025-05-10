import os
import asyncio
from anthropic import Anthropic

client = Anthropic(
    base_url="http://47.88.17.95:1003",
    default_headers={
       "anthropic-version": "2023-06-01",
       "content-type": "application/json"
    },
    api_key="asdf"
)






# usage
message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello, Claude",
        }
    ],
    model="claude-3-7-sonnet-20250219",
)
print(message.content)


# 异步使用
async def main() -> None:
    message = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello, Claude",
            }
        ],
        model="claude-3-5-sonnet-latest",
    )
    print(message.content)


asyncio.run(main())






# def get_thinking_and_response(prompt):
    
#     client = Anthropic(
#     base_url="http://47.88.17.95:1003",
#     default_headers={
#        "anthropic-version": "2023-06-01",
#        "content-type": "application/json"
#     },
#     api_key="asdf"
# )

#     think = ""
#     text = ""
#     with client.messages.stream(
#         model="claude-3-7-sonnet-20250219",

#         max_tokens=64000,
#         thinking={"type": "enabled", "budget_tokens": 60000},
#         messages=[{"role": "user", "content": f"{prompt}"}],
#     ) as stream:
#         thinking = "not-started"

#         for event in stream:
#             if event.type == "thinking":
#                 if thinking == "not-started":

#                     thinking = "started"
#                 # breakpoint()
#                 # print(event.thinking, end="", flush=True)
#                 think += event.thinking
#             elif event.type == "text":
#                 if thinking != "finished":

#                     thinking = "finished"

#                 # print(event.text, end="", flush=True)
#                 text += event.text
#     return think, text