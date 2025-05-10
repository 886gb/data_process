# Description: Translate the prompt from English to Chinese using OpenAI API
import os
from openai import OpenAI
import json
from tqdm import tqdm

openai_api_base ="http://123.127.232.152:1222/v1"
key ="1111"
client = OpenAI(base_url=openai_api_base,api_key=key)

# file_path = 'lima_data/lima_data_test.json'
# file_path = 'lima_data/lima_data_train.json'
file_path = 'repochat-arena-preference-4k/repochat-arena-preference-4k_train.json'

split_file_path = file_path.split('.json')[0]
output_file_path= split_file_path+'_chinese_test.json'

# if os.path.exists(output_file_path):
#     os.remove(output_file_path)

count = 0
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for index, line in tqdm(enumerate(lines), total=len(lines),desc="Processing"):
        if index < 8:
              continue  # 跳过条数据
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
        
        # 检查输入 token 数量是否超过限制
        if len(prompt) > 65536:
            print(f"Input tokens exceed the limit, skipping 第{index+1}条.")
            continue
        
        # responses = client.chat.completions.create(
        #     model="111",
        #     messages=[
        #         {"role": "system", "content": "你是一个专业的翻译聊天机器人，你只负责翻译任务，当用户发给你对话时，你只需要把它翻译为中文。用户代码块中的内容只翻译注释部分，不翻译代码部分，返回翻译后的完整内容用Markdown格式。下面是用户发给你的需要翻译为中文的内容："},
        #         {
        #             "role": "user",
        #             "content": f"```{prompt}```"
        #         }
        #     ],
        #     temperature=0,
        #     stream = True,
        #     max_tokens=65536,
        # )
        
        
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
            # if chunk.choices[0].delta.content:  # 确保有内容
                content_piece = chunk.choices[0].delta.content
                response_text += content_piece  # 累加到完整响应

        
        
        chinese_conversations = response_text
        
        # import pdb;pdb.set_trace()
        if chinese_conversations.startswith('```'):
          chinese_conversations = chinese_conversations[3:]
        if chinese_conversations.endswith('```'):
          chinese_conversations = chinese_conversations[:-3]
        # import pdb;pdb.set_trace()
        translated_data ={}
        translated_data['conv_ids'] = data['conv_ids']
        translated_data['prompt_raw'] = prompt
        translated_data['chinese_prompt'] = chinese_conversations
        

        with open(output_file_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(translated_data, ensure_ascii=False) + '\n')
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



'''

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

'''