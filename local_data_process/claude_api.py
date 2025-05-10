import time
import json
import anthropic
import requests


client = anthropic.Anthropic(
base_url="http://47.88.17.95:6888",
default_headers={
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
},
api_key="sk-lkzGUYtv1iXIQyV8gta3oRLMm7xpqmhFOksp6jG7FqiH55H4"
)




"""
curl http://47.88.17.95:6888/v1/messages \
     --header "x-api-key: sk-lkzGUYtv1iXIQyV8gta3oRLMm7xpqmhFOksp6jG7FqiH55H4" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-3-7-sonnet-20250219",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, Claude"}
    ]
}'

"""

def claude_api_cot(prompt):
    retries = 20
    for attempt in range(retries):
        try:
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
        except anthropic.APIStatusError as e:
            print(e)
            time.sleep(20)

    raise RuntimeError("All retry attempts failed due to API overload.")



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
    think, text = claude_api_cot(prompt)
    print("Thinking:\n----------------------------------")
    print("\nFinal think:\n", think)
    print("\n\nText:\n==================================")
    print("\nFinal text:\n", text)
    
