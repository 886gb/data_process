from openai import OpenAI
import time

openai_api_base = "https://ark.cn-beijing.volces.com/api/v3"
key = "06aabd85-66ff-481a-8c2e-81813f64ab4a"
client = OpenAI(base_url=openai_api_base, api_key=key,timeout=1800)

def deepseek_api_cot(prompt):
    for attempt in range(10):
        thinking = ""
        answer = ""
        try:
            stream = client.chat.completions.create(
                model="deepseek-r1-250120",
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": f"{prompt}"}
                ],
                temperature=0,
                max_tokens=16000,
                stream = True
            )
            for chunk in stream:
                # breakpoint()
                if not chunk.choices:
                    continue
                # breakpoint()
                if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                    thinking += chunk.choices[0].delta.reasoning_content
                answer += chunk.choices[0].delta.content
            return thinking, answer
        except Exception as e:
            time.sleep(10)
            print(f"An error occurred: {e}")
    
    
if __name__ == "__main__":
    prompt = "介绍一下人工智能？"
    thinking ,response= deepseek_api_cot(prompt)
    print(thinking)
    print("#"*100)
    print(response)
    
    
    
"""
curl https://ark.cn-beijing.volces.com/api/v3/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 06aabd85-66ff-481a-8c2e-81813f64ab4a" \
  -d '{
    "model": "deepseek-r1-250120",
    "messages": [
        {
            "role": "user",
            "content": "我要有研究推理模型与非推理模型区别的课题，怎么体现我的专业性"
        }
    ]
  }'
"""