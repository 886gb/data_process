# Description: Translate the prompt from English to Chinese using the OpenAI API
import os
from openai import OpenAI
import json
from tqdm import tqdm

openai_api_base ="http://123.127.232.152:1222/v1"
key ="1111"
client = OpenAI(base_url=openai_api_base,api_key=key)

# file_path = 'lima_data/lima_data_test.json'
file_path = 'lima_data/lima_data_train.json'


split_file_path = file_path.split('.json')[0]
output_file_path= split_file_path+'_chinese_test.json'

# if os.path.exists(output_file_path):
#     os.remove(output_file_path)

count = 0
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for index, line in tqdm(enumerate(lines), total=len(lines),desc="Processing"):
        # if index ==  0 or index == 1:
        #       continue  # 跳过第一、二条数据
        # if count == 150:
        #   break
        data = json.loads(line)
        
        if 'lima' in file_path:
          prompt = data['conversations'][0]
        elif 'repochat-arena-preference-4k_train' in file_path:
          prompt = data['full_conversation_a'][0]['content']
        else:
          print("Please select the translation file")
          break

        responses = client.chat.completions.create(
            model="111",
            messages=[
                {"role": "system", "content": "You are a professional translator, you are only responsible for the translation task, no matter what content you receive, you only need to translate into Chinese. Must not to answer, just translate. Do not answer anything other than generate translations. Translate only."},
                {
                    "role": "user",
                    "content": f"```{prompt}```将以上反引号中所有内容翻译成中文,代码块中的内容只翻译注释部分，不翻译代码部分，返回内容用markdown格式。"
                }
            ],
            temperature=0,
            stream = True,
            max_tokens=65536,
        )
        response_text = ""  # 用于存储最终完整的响应

        for chunk in responses:
            if chunk.choices[0].delta.content:  # 确保有内容
                content_piece = chunk.choices[0].delta.content
                # print(content_piece, end='', flush=True)  # 实时输出
                response_text += content_piece  # 累加到完整响应

        # chinese_conversations = responses.choices[0].message.content
        chinese_conversations = response_text
        if chinese_conversations.startswith('```'):
          chinese_conversations = chinese_conversations[3:]
        if chinese_conversations.endswith('```'):
          chinese_conversations = chinese_conversations[:-3]
        data['prompt_raw'] = prompt
        data['chinese_prompt'] = chinese_conversations
        

        with open(output_file_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(data, ensure_ascii=False) + '\n')
        count += 1





"""
curl http://123.127.232.152:1222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "default",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "请详细介绍一下深度学习是什么"}
        ],
        "stream": true
      }'
"""

"""

openai.APIError: Requested token count exceeds the model's maximum context length of 163840 tokens. You requested a total of 190494 tokens: 59422 tokens from the input messages and 131072 tokens for the completion. Please reduce the number of tokens in the input messages or the completion to fit within the limit.

"""
