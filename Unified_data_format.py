import json

def exact_preference_data(prompt, prompt_source, metadata, source_prompt_id):
    if "Skywork" in prompt_source:
        chosen = metadata['chosen'][1]['content']
        rejected = metadata['rejected'][1]['content']
    elif "lmarena-ai" in prompt_source:
        if "arena-human-preference-100k" in source_prompt_id:
            chosen = []
            rejected = []
            if metadata['winner'] == "model_b":
                prompt = []
                for i in range(metadata['turn']):
                    prompt.append(metadata['conversation_b'][2*i]['content'])
                    chosen.append(metadata['conversation_b'][2*i+1]['content'])
                    rejected.append(metadata['conversation_a'][2*i+1]['content'])
            elif metadata['winner'] == "model_a":
                prompt = []
                for i in range(metadata['turn']):
                    prompt.append(metadata['conversation_a'][2*i]['content'])
                    chosen.append(metadata['conversation_a'][2*i+1]['content'])
                    rejected.append(metadata['conversation_b'][2*i+1]['content'])
            else:
                chosen = ""
                rejected = ""
        elif "arena-human-preference-55k" in source_prompt_id:
            if metadata['winner_model_a'] == 1:
                chosen = metadata['response_a'][2:-2]
                rejected = metadata['response_b'][2:-2]
            elif metadata['winner_model_b'] == 1:
                chosen = metadata['response_b'][2:-2]
                rejected = metadata['response_a'][2:-2]
            else:
                chosen = ""
                rejected = ""
        elif "repochat-arena-preference-4k" in source_prompt_id:
            if metadata['winner'] == "model_b":
                chosen = metadata['full_conversation_b'][1]['content']
                rejected = metadata['full_conversation_a'][1]['content']
            elif metadata['winner'] == "model_a":
                chosen = metadata['full_conversation_a'][1]['content']
                rejected = metadata['full_conversation_b'][1]['content']
            else:
                chosen = ""
                rejected = ""
        elif "webdev-arena-preference-10k" in source_prompt_id:
            if metadata['winner'] == "model_b":
                chosen = metadata['conversation_b']
                rejected = metadata['conversation_a']
            elif metadata['winner'] == "model_a":
                chosen = metadata['conversation_a']
                rejected = metadata['conversation_b']
            else:
                chosen = ""
                rejected = ""
    elif "OpenR1-Math-220k" in prompt_source:
        reasoning_verify = metadata['correctness_math_verify']
        if len(reasoning_verify) == 2:
            if reasoning_verify[0] == False and reasoning_verify[1] == True:
                chosen = metadata['generations'][1]
                rejected = metadata['generations'][0]
            elif reasoning_verify[0] == True and reasoning_verify[1] == False:
                chosen = metadata['generations'][0]
                rejected = metadata['generations'][1]
            else:
                chosen = ""
                rejected = ""
        else:
            chosen = ""
            rejected = ""
    elif "HelpSteer3" in prompt_source:
        if "HelpSteer3_preference" in source_prompt_id:
            overall_preference = metadata['overall_preference']
            if overall_preference == 0:
                chosen = ""
                rejected = ""
            elif overall_preference < 0:
                chosen = metadata['response1']
                rejected = metadata['response2']
            elif overall_preference > 0:
                chosen = metadata['response2']
                rejected = metadata['response1']
        elif "HelpSteer3_edit_quality" in source_prompt_id:
            chosen = metadata['good_edited_response']
            rejected = metadata['bad_edited_response']
        elif "HelpSteer3_edit" in source_prompt_id:
            chosen = metadata['edited_response']
            rejected = metadata['original_response']
    else:
        raise ValueError(f"Unknown preference prompt source: {prompt_source}")
    print(source_prompt_id)
    return prompt, chosen, rejected


def exact_sft_data(prompt_source, metadata, source_prompt_id):
    if "AI-MO" in prompt_source:
        answer = metadata['solution']
    elif "allenai" in prompt_source:
        if "tulu-3-sft-olmo-2-mixture-0225" in source_prompt_id:
            answer = metadata['messages'][1]['content']
        elif "tulu-3-sft-personas-math" in source_prompt_id:
            answer = metadata['messages'][1]['content']
        elif "tulu-3-sft-personas-code" in source_prompt_id:
            answer = metadata['messages'][1]['content']
            
    elif "GAIR" in prompt_source:
        answer = metadata['conversations'][1]
    elif "HuggingFaceH4" in prompt_source:
        answer = metadata['messages'][0]['content']
    elif "KodCode" in prompt_source:
        answer = metadata['solution']
    elif "m-a-p" in prompt_source:
        answer = metadata['answer']
    elif "opc-sft-stage1" in prompt_source:
        answer = metadata['output']
    elif "opc-sft-stage2" in prompt_source:
        answer = metadata['output']
    else:
        raise ValueError(f"Unknown sft_prompt source: {prompt_source}")
    print(source_prompt_id)
    return answer


def exact_cot_data(prompt_source, metadata):
    if "KodCode" in prompt_source:
        start = metadata['conversations'][1]['value'].find("<think>")
        end = metadata['conversations'][1]['value'].find("</think>")
        thinking = metadata['conversations'][1]['value'][start+7:end]
        answer = metadata['conversations'][1]['value'][end+8:]
    elif "open-r1" in prompt_source:
        start = metadata['messages'][1]['content'].find("<think>")
        end = metadata['messages'][1]['content'].find("</think>")
        thinking = metadata['messages'][1]['content'][start+7:end]
        answer = metadata['messages'][1]['content'][end+8:]
    # 不是cot(<think>  </think>)
    elif "GAIR" in prompt_source:
        thinking = metadata['solution']
        answer = ""
    elif "simplescaling" in prompt_source:
        thinking = metadata['thinking_trajectories'][0]
        answer = metadata['attempt']
    elif "open-thoughts" in prompt_source:
        thinking = metadata['deepseek_reasoning']
        answer = metadata['deepseek_solution']

    elif "GeneralReasoning" in prompt_source:
        thinking = metadata['model_reasoning']
        answer = metadata['model_answer']
        
    # AM-DeepSeek-R1-Distilled-1.4M  
    elif "a-m-team" in prompt_source:
        thinking = metadata['messages'][1]['info']['think_content']
        answer = metadata['messages'][1]['info']['answer_content']
    elif "Congliu" in prompt_source:
        thinking = metadata['reasoning_content']
        answer = metadata['content']
    elif "OpenR1-Math-220k" in prompt_source:
        start = metadata['messages'][1]['content'].find("<think>")
        end = metadata['messages'][1]['content'].find("</think>")
        thinking = metadata['messages'][1]['content'][start+7:end]
        answer = metadata['messages'][1]['content'][end+8:]
    elif "qihoo360" in prompt_source:
        start = metadata['conversations'][1]['value'].find("<think>")
        end = metadata['conversations'][1]['value'].find("</think>")
        thinking = metadata['conversations'][1]['value'][start+7:end]
        answer = metadata['conversations'][1]['value'][end+8:]
    # 不是cot
    elif "AI-MO" in prompt_source:
        thinking = metadata['solution']
        answer = ""
        
    # "nvidia/Llama-Nemotron-Post-Training-Dataset-v1" 
    elif "nvidia" in prompt_source:
        start = metadata['output'].find("<think>")
        end = metadata['output'].find("</think>")
        thinking = metadata['output'][start+7:end]
        answer = metadata['output'][end+8:]
    else:
        breakpoint()
        raise ValueError(f"Unknown cot prompt source: {prompt_source}")
    return thinking, answer



def process_preference_data(file_path, output_file):
    with open(file_path, 'r') as f:
        for line in f:
            # breakpoint()
            line = json.loads(line)
            metadata = line['metadata']
            prompt_source = line['source_prompt_id'].split('/')[-2]
            metadata = json.loads(metadata)
            prompt_raw = line['prompt']
            prompt, chosen, rejected = exact_preference_data(prompt_raw, prompt_source, metadata, line['source_prompt_id'])
            # if chosen == "" and rejected == "":
            #     continue
            data = {"all_prompt_id": line['all_prompt_id'],
                    "source_prompt_id": line['source_prompt_id'],
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": metadata}
            with open(output_file, 'a') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

def process_cot_data(file_path, output_file):
    with open(file_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            metadata = line['metadata']
            prompt_source = line['source_prompt_id'].split('/')[-2]
            metadata = json.loads(metadata)
            thinking, answer = exact_cot_data(prompt_source, metadata)
            data = {"all_prompt_id": line['all_prompt_id'],
                    "source_prompt_id": line['source_prompt_id'],
                    "prompt": line['prompt'],
                    "thinking": thinking,
                    "answer": answer,
                    "metadata": metadata}
            with open(output_file, 'a') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

def process_sft_data(file_path, output_file):
    with open(file_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            metadata = line['metadata']
            prompt_source = line['source_prompt_id'].split('/')[-2]
            metadata = json.loads(metadata)
            answer = exact_sft_data(prompt_source, metadata, line['source_prompt_id'])
            data = {"all_prompt_id": line['all_prompt_id'],
                    "source_prompt_id": line['source_prompt_id'],
                    "prompt": line['prompt'],
                    "answer": answer,
                    "metadata": metadata}
            with open(output_file, 'a') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')


def process_files(input_file, output_file):
    """
    递归读取指定目录及其子目录下的所有文件，为每条数据添加meta键，并保存到一个文件中
    每行一条JSON数据
    
    Args:
        input_file: 要处理的目录路径
        output_file: 输出文件路径
    """
    if "preference" in input_file:
        process_preference_data(input_file, output_file)
    elif "COT" in input_file:
        process_cot_data(input_file, output_file)
    
    elif "SFT" in input_file:
        process_sft_data(input_file, output_file)
    else:
        raise ValueError(f"Unknown prompt source: {input_file}")


if __name__ == "__main__":
    
    input_file = "data/test_SFT_all.json"
    
    output_file = "outputs/test/process/test_SFT_all.json"
    
    
    process_files(input_file, output_file)
    print("done,文件保存到", output_file)