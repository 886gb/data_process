import time
import json
import anthropic
import requests



def get_thinking_and_response(prompt):
    
    client = anthropic.Anthropic(
    base_url="http://47.88.17.95:1007",
    default_headers={
       "anthropic-version": "2023-06-01",
       "content-type": "application/json"
    },
    api_key="asdf"
)

    think = ""
    text = ""
    with client.messages.stream(
        model="claude-3-7-sonnet-20250219",

        max_tokens=64000,
        thinking={"type": "enabled", "budget_tokens": 60000},
        messages=[{"role": "user", "content": f"{prompt}"}],
    ) as stream:
        thinking = "not-started"
        for event in stream:
            if event.type == "thinking":
                if thinking == "not-started":
                    thinking = "started"
                # print(event.thinking, end="", flush=True)
                think += event.thinking
            elif event.type == "text":
                if thinking != "finished":
                    thinking = "finished"

                # print(event.text, end="", flush=True)
                text += event.text
    return think, text




def claude_post_request(prompt):
    api_base = "http://47.88.17.95:1003/v1/messages"
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": f"{prompt}"}
        ]
    }
    try:
        response = requests.post(api_base, headers=headers, data=json.dumps(data), proxies={"http": None, "https": None})
        response.raise_for_status()
        completion = response.json()['content'][0]['text']
        return completion
    except requests.exceptions.HTTPError as e:
        print(f"Http error occurred: {e}")
    except Exception as e:
        print(f"An requests error occurred: {e}")
        

if __name__ == "__main__":
    prompt = "什么是人工智能？"
    think, text = get_thinking_and_response(prompt)
    print("Thinking:\n----------------------------------")
    print("\nFinal think:\n", think)
    print("\n\nText:\n==================================")
    print("\nFinal text:\n", text)
    