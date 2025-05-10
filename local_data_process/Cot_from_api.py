import requests
import json
from openai import OpenAI
import tqdm
import argparse
import os
from claude_api import claude_api_cot
from  deepseek_api import deepseek_api_cot

def ds_post_request(prompt):
    openai_api_base ="http://123.127.232.152:1222/v1"
    key ="1111"
    client = OpenAI(base_url=openai_api_base,api_key=key)
    responses = client.chat.completions.create(
        model="111",
        messages=[
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": f"{prompt}"
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
    return response_text





def choose_model(prompt):
    if model_name == 'claude':
        thinking, response = claude_api_cot(prompt)
    elif model_name == 'deepseek':
        thinking, response = deepseek_api_cot(prompt)
    return thinking, response



def read_data(file_path, read_limit_num=None):
    prompts = []
    domains = []
    with open(file_path,"r",encoding="utf-8") as file:
        lines = file.readlines() if read_limit_num is None else file.readlines()[:read_limit_num]
        for index, line in tqdm.tqdm(enumerate(lines),total= len(lines),desc="Load data..."):
            data  = json.loads(line)
            prompts.append(data['problem'])
            domains.append(data['domain'])
        return domains, prompts
            
def read_lima_data(file_path, read_limit_num=None):
    Eng_prompts = []
    Chinese_prompts = []
    with open(file_path,"r",encoding="utf-8") as file:
        lines = file.readlines() if read_limit_num is None else file.readlines()[:read_limit_num]
        for index, line in tqdm.tqdm(enumerate(lines),total= len(lines),desc="Load data..."):
            data  = json.loads(line)
            Eng_prompts.append(data['prompt_raw'])
            Chinese_prompts.append(data['chinese_prompt'])
        return Eng_prompts, Chinese_prompts

def read_repochat_arena_data(file_path, read_limit_num=None):
    Eng_prompts = []
    Chinese_prompts = []
    conv_ids = []
    with open(file_path,"r",encoding="utf-8") as file:
        lines = file.readlines() if read_limit_num is None else file.readlines()[:read_limit_num]
        for index, line in tqdm.tqdm(enumerate(lines),total= len(lines),desc="Load data..."):
            data  = json.loads(line)
            conv_ids.append(data['conv_ids'])
            if "full_conversation_a" not in data:
                Eng_prompts.append(data['prompt_raw'])
            else:
                Eng_prompts.append(data['full_conversation_a'][0]['content'])
            Chinese_prompts.append(data['chinese_prompt'])
            
    return conv_ids, Eng_prompts, Chinese_prompts



def read_EricLu_data(start,read_limit_num,file_path):
    prompts = []
    domains = []
    with open(file_path, "r", encoding="utf-8") as file:
        if read_limit_num == -1:
            lines = file.readlines()[start:]
        else:
            # 读取指定行数
            lines = file.readlines()[start:read_limit_num]
        for index, line in tqdm.tqdm(enumerate(lines), total= len(lines), desc="Load data..."):
            data  = json.loads(line)
            prompts.append(data['problem'])
            domains.append(data['domain'])
        return domains, prompts

def save_EricLu_result(domain, prompt, thinking, response, result_file_path):
    result = {
        "domain": domain,
        "prompt": prompt,
        "thinking": thinking,
        "response": response
        
    }

    with open(result_file_path,"a",encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")
        # 先计算已有行数
    try:
        with open(result_file_path, "r", encoding="utf-8") as read_file:
            total_lines = len(read_file.readlines())
            print(f"推理了{total_lines}条")
    except FileNotFoundError:
        total_lines = 0


        
def save_lima_data_result(Eng_prompts, Chinese_prompts, thinking,  response, result_file_path):
    result = {
        "raw_prompt": Eng_prompts,
        "Chinese_prompt": Chinese_prompts,
        "thinking": thinking, 
        "response": response
    }
    with open(result_file_path,"a",encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        
def save_repochat_arena_result(conv_ids, Eng_prompts, Chinese_prompts, thinking, response, result_file_path):
    result = {
        "conv_id": conv_ids,
        "Eng_prompt": Eng_prompts,
        "Chinese_prompt": Chinese_prompts,
        "thinking":thinking,
        "response": response
    }
    with open(result_file_path,"a",encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
        if 'lima_data' in input_file_path :
            Eng_prompts, Chinese_prompts = read_lima_data(input_file_path, read_limit_num)
            print("Start infra...")
            for Eng_prompts, Chinese_prompts in zip(Eng_prompts, Chinese_prompts):
                thinking, response = choose_model(Chinese_prompts)
                save_lima_data_result(Eng_prompts, Chinese_prompts, thinking, response, result_file_path)
            
        elif 'repochat-arena' in result_file_path:
            conv_ids, Eng_prompts, Chinese_prompts = read_repochat_arena_data(input_file_path, read_limit_num)
            print("Start infra...")
            for conv_ids, Eng_prompts, Chinese_prompts in zip(conv_ids, Eng_prompts, Chinese_prompts):
                thinking, response = choose_model(Chinese_prompts)
                save_repochat_arena_result(conv_ids, Eng_prompts, Chinese_prompts, thinking, response, result_file_path)

        elif 'EricLu' in result_file_path:
            domains, prompts = read_EricLu_data(start, read_limit_num, input_file_path)
            print("Start infra...")
            for domain, prompt in zip(domains, prompts):
                thinking, response = choose_model(prompt)

                save_EricLu_result(domain, prompt, thinking, response, result_file_path)
                 

    
    
if __name__ == "__main__":
    
    paser = argparse.ArgumentParser()
    paser.add_argument("--input_file_path", type=str, default="", help="The input file path")
    paser.add_argument("--result_file_path", type=str, default="", help="The result file path")
    paser.add_argument("--model", type=str, default="deepseek", help="The model name:claude or deepseek")
    paser.add_argument("--data_end", type=int, default= -1, help="The number of data to be processed")    
    paser.add_argument("--resume", type=bool, default=False, help="是否从头开始处理数据")
    args = paser.parse_args()
    
    input_file_path = args.input_file_path
    result_file_path = args.result_file_path 
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    model_name = args.model
    read_limit_num = args.data_end
    resume = args.resume
    if resume:
        with open(result_file_path, "r", encoding="utf-8") as read_file:
            total_lines = len(read_file.readlines())
            print(f"已经推理{total_lines}条")
            start = total_lines
    else:
        start = 0
    print(f"当前处理数据行数：{start+1}")
    main()
    
    
    
    
    
"""

启动命令

python Cot_from_api.py --input_file_path repochat-arena-preference-4k/China_repochat-arena-preference-4k/repochat-arena-preference-4k_train_chinese_50.json --result_file_path output/repochat-arena/claude/repochat-arena-preference-4k_train_chinese_50.json --model claude


python Cot_from_api.py --input_file_path lima_data/China_lima_data/lima_data_train_chinese_50.json --result_file_path output/lima_data/deepseek/lima_data_train_chinese_50_from_api.json --model deepseek


python Cot_from_api.py --input_file_path /Users/edy/Desktop/zt/zt_aixcoder/data/EricLu/SCP_116K_train.json --result_file_path ./output/EricLu/claude/SCP_116K_train_result.json --model claude >/Users/edy/Desktop/zt/zt_aixcoder/data/logs/SCP_116K_train_result_log.txt 3>&1

tail -f /Users/edy/Desktop/zt/zt_aixcoder/data/logs/SCP_116K_train_result_log.txt
    
    
    
"""






"""

curl http://47.88.17.95:1003/v1/messages \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-3-7-sonnet-20250219",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, world"}
    ]
}'

"""

