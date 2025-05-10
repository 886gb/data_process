import jsonlines
import tqdm
import json

# def read_data(file_path, read_limit_num=None):
#     prompts = []
#     domains = []
#     with open(file_path,"r",encoding="utf-8") as file:
#         lines = file.readlines() if read_limit_num is None else file.readlines()[:read_limit_num]
#         for index, line in tqdm.tqdm(enumerate(lines),total= len(lines),desc="Load data..."):
#             data  = json.loads(line)
#             prompts.append(data['problem'])
#             domains.append(data['domain'])
#         return domains, prompts





def read_data(input_path, read_limit_num=None):
    # 加载 prompts
    if input_path.split("/")[2] == "prompt" or input_path.split("/")[2] == "test":
        if "BytedTsinghua-SIA" in input_path:
            raw_data_dict, prompts = read_BytedTsinghua_SIA(input_path)
        elif "dsrselfcorr" in input_path:
            raw_data_dict, prompts = read_dsrselfcorr(input_path)
        elif "fka" in input_path:
            raw_data_dict, prompts = read_fka(input_path)
        elif "pvduy" in input_path:
            raw_data_dict, prompts = read_pvduy(input_path)
        elif "trl-lib" in input_path:
            raw_data_dict, prompts = read_trl_lib(input_path)
            
    elif input_path.split("/")[2] == "SFT":

        if "AI-MO" in input_path:
            raw_data_dict, prompts = read_AI_MO(input_path, read_limit_num)
        elif "allenai" in input_path:
            if "tulu-3-sft-olmo-2-mixture-0225" in input_path:
                raw_data_dict, prompts = read_tulu3_sft_olmo_2_mixture_0225(input_path, read_limit_num)
            else:
                raw_data_dict, prompts = read_allenai(input_path, read_limit_num)
        elif "GAIR" in input_path:
            raw_data_dict, prompts = read_GAIR(input_path, read_limit_num)
        elif "HuggingFaceH4" in input_path:
            raw_data_dict, prompts = read_HuggingFaceH4(input_path, read_limit_num)
        elif "KodCode" in input_path:
            raw_data_dict, prompts = read_KodCode(input_path, read_limit_num)
        elif "m-a-p" in input_path:
            raw_data_dict, prompts = read_m_a_p(input_path, read_limit_num)
        elif "opc-sft-stage1" in input_path:
            raw_data_dict, prompts = read_opc_sft_stage1(input_path, read_limit_num)
        elif "opc-sft-stage2" in input_path:
            raw_data_dict, prompts = read_opc_sft_stage2(input_path, read_limit_num)
    
    
    elif input_path.split("/")[2] == "preference":
        
        if "Skywork" in input_path:
            raw_data_dict, prompts = read_Skywork(input_path, read_limit_num)
        elif "lmarena-ai" in input_path:
            if "arena-human-preference-100k" in input_path:
                raw_data_dict, prompts = read_arena_human_preference_100k(input_path, read_limit_num)
            elif "arena-human-preference-55k" in input_path:
                raw_data_dict, prompts = read_arena_human_preference_55k(input_path, read_limit_num)
            elif "repochat-arena-preference-4k" in input_path:
                raw_data_dict, prompts = read_repochat_arena_preference_4k(input_path, read_limit_num)
            elif "webdev-arena-preference-10k" in input_path:
                raw_data_dict, prompts = read_webdev_arena_preference_10k(input_path, read_limit_num)
        elif "OpenR1-Math-220k" in input_path:
            raw_data_dict, prompts = read_OpenR1_Math_220k(input_path, read_limit_num)
        elif "HelpSteer3" in input_path:
            raw_data_dict, prompts = read_HelpSteer3(input_path, read_limit_num)

            
    elif input_path.split("/")[2] == "COT":
        # breakpoint()
        if "KodCode_V1_SFT_R1_train" in input_path:
            raw_data_dict, prompts = read_KodCode_V1_SFT_R1_train(input_path, read_limit_num)
        elif "open-r1" in input_path:
            if "codeforces_cots_solutions_decontaminated" in input_path:
                raw_data_dict, prompts = read_codeforces_cots_solutions_decontaminated(input_path, read_limit_num)
            else:
                raw_data_dict, prompts = read_open_r1(input_path, read_limit_num)
        elif "GAIR" in input_path:
            raw_data_dict, prompts = read_COT_GAIR(input_path, read_limit_num)
        elif "simplescaling" in input_path:
            raw_data_dict, prompts = read_simplescaling(input_path, read_limit_num)
        elif "OpenThoughts-114k_metadata" in input_path:
            raw_data_dict, prompts = read_OpenThoughts_114k_metadata(input_path, read_limit_num)
        elif "Llama-Nemotron-Post-Training-Dataset-v1" in input_path:
            raw_data_dict, prompts = read_Llama_Nemotron_Post_Training_Dataset_v1(input_path, read_limit_num)
        elif "GeneralThought-430K" in input_path:
            raw_data_dict, prompts = read_GeneralThought_430K(input_path, read_limit_num)
        elif "AM-DeepSeek-R1-Distilled-1.4M" in input_path:
            raw_data_dict, prompts = read_AM_DeepSeek_R1_Distilled_1_4M(input_path, read_limit_num)
        elif "Chinese-DeepSeek-R1-Distill-data-110k" in input_path:
            raw_data_dict, prompts = read_Chinese_DeepSeek_R1_Distill_data_110k(input_path, read_limit_num)
        elif "OpenR1-Math-220k" in input_path:
            raw_data_dict, prompts = read_OpenR1_Math_220k(input_path, read_limit_num)
        elif "Light-R1-SFTData" in input_path:
            raw_data_dict, prompts = read_Light_R1_SFTData(input_path, read_limit_num)
        elif "NuminaMath-CoT" in input_path:
            raw_data_dict, prompts = read_NuminaMath_CoT(input_path, read_limit_num)
        elif "nvidia" in input_path:
            raw_data_dict, prompts = read_nvidia(input_path, read_limit_num)
        elif "a-m-team" in input_path:
            raw_data_dict, prompts = read_a_m_team(input_path, read_limit_num)
    else:
        print("未选择输入文件")
        return    
    return raw_data_dict, prompts


def read_a_m_team(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['messages'][0]['content'])
    return lines[:read_limit_num], prompts


def read_nvidia(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['input'][0]['content'])
    return lines[:read_limit_num], prompts



def read_NuminaMath_CoT(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['problem'])
    return lines[:read_limit_num], prompts




def read_Light_R1_SFTData(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['conversations'][0]['value'])
    return lines[:read_limit_num], prompts




def read_Chinese_DeepSeek_R1_Distill_data_110k(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['input'])
    return lines[:read_limit_num], prompts


def read_AM_DeepSeek_R1_Distilled_1_4M(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            # breakpoint()
            prompts.append(line['messages'][0]['content'])
    return lines[:read_limit_num], prompts





def read_GeneralThought_430K(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['question'])
    return lines[:read_limit_num], prompts

# 检查
def read_Llama_Nemotron_Post_Training_Dataset_v1(input_path, read_limit_num):
    breakpoint()
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            input = line['input']
            start = input.find("<|start_header_id|>user<|end_header_id|>")
            end = input.find("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
            prompt = input[start:end].replace("<|start_header_id|>user<|end_header_id|>", "").replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", "")
            prompts.append(prompt)
    return lines[:read_limit_num], prompts



def read_OpenThoughts_114k_metadata(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['problem'])
    return lines[:read_limit_num], prompts


def read_simplescaling(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['question'])
    return lines[:read_limit_num], prompts

def read_COT_GAIR(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            # breakpoint()
            prompts.append(line['question'])
    return lines[:read_limit_num], prompts

def read_open_r1(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            # breakpoint()
            prompts.append(line['prompt'])
    return lines[:read_limit_num], prompts


def read_codeforces_cots_solutions_decontaminated(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            # breakpoint()
            prompts.append(line['problem'])
    return lines[:read_limit_num], prompts


def read_KodCode_V1_SFT_R1_train(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            # breakpoint()
            prompts.append(line['question'])
    return lines[:read_limit_num], prompts



def read_HelpSteer3(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            
            if isinstance(line['context'], list):
                prompt_content = ""
                for item in line['context']:
                    if 'content' in item and item['role'] == 'user':
                        prompt_content += item['content'] + "\n"
                prompts.append(prompt_content.strip())
            # elif isinstance(line['context'], dict) and 'content' in line['context']:
            #     prompts.append(line['context']['content'])
            # else:
            #     prompts.append(line['prompt'])
    return lines[:read_limit_num], prompts



def read_Skywork(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['chosen'][0]['content'])
    return lines[:read_limit_num], prompts


def read_arena_human_preference_100k(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['conversation_a'][0]['content'])
    return lines[:read_limit_num], prompts


def read_arena_human_preference_55k(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompt_list = json.loads(line['prompt'])
            prompt_content = ""
            for item in prompt_list:
                prompt_content += item + "\n"
            prompts.append(prompt_content.strip())
    return lines[:read_limit_num], prompts


def read_repochat_arena_preference_4k(input_path, read_limit_num):

    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['full_conversation_a'][0]['content'])
    return lines[:read_limit_num], prompts


def read_webdev_arena_preference_10k(input_path, read_limit_num):

    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['conversation_a'][0]['content'][0]['text'])
    return lines[:read_limit_num], prompts



def read_OpenR1_Math_220k(input_path, read_limit_num):

    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['problem'])
    return lines[:read_limit_num], prompts



def read_AI_MO(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['problem'])
    return lines[:read_limit_num], prompts


def read_allenai(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['prompt'])  
    return lines[:read_limit_num], prompts   

def read_GAIR(input_path, read_limit_num):

    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['conversations'][0])  
    return lines[:read_limit_num], prompts   


def read_HuggingFaceH4(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['prompt'])  
    return lines[:read_limit_num], prompts   


def read_KodCode(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['question'])  
    return lines[:read_limit_num], prompts   

def read_m_a_p(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['query'])  
    return lines[:read_limit_num], prompts   


def read_opc_sft_stage1(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['instruction'])  
    return lines[:read_limit_num], prompts   


def read_opc_sft_stage2(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['instruction'])  
    return lines[:read_limit_num], prompts   


def read_tulu3_sft_olmo_2_mixture_0225(input_path, read_limit_num):
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:read_limit_num]:
            line = json.loads(line)
            prompts.append(line['messages'][0]['content'])
    return lines[:read_limit_num], prompts   





def read_BytedTsinghua_SIA(input_path):
    # raw_data_dict = {}
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            prompts.append(line['prompt'][0]['content'])
            # raw_data_dict[line['prompt'][0]['content']] = line
    return lines, prompts


def read_dsrselfcorr(input_path):
    # raw_data_dict = {}
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            prompts.append(line['prompt'][1]['content'])
            # raw_data_dict[line['prompt'][1]['content']] = line
    return lines, prompts


def read_fka(input_path):
    # raw_data_dict = []
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            prompts.append(line['prompt'])
            # raw_data_dict[line['prompt']] = line
    return lines, prompts
            
            
def read_pvduy(input_path):
    # raw_data_dict = {}
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            prompts.append(line['prompt'])
            # raw_data_dict[line['prompt']] = line
    return lines, prompts   
            
            
            
            
def read_trl_lib(input_path):
    # raw_data_dict = {}
    prompts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            prompts.append(line['prompt'][0]['content'])
            # raw_data_dict[line['prompt'][0]['content']] = line
    return lines, prompts

            

def read_OpenThoughts_114_prompts(input_path):
    raw_data_dict = {}
    prompts = []
    with jsonlines.open(input_path, 'r') as f:
        for line in f:
            # breakpoint()
            prompts.append(line['conversations'][0]['value'])
            raw_data_dict[line['conversations'][0]['value']] = line
            
    return raw_data_dict, prompts


    
def read_OpenThoughts_114k_prompts_solutions(input_path):
    raw_data_dict = {}
    prompts_solutions = [] 
    with jsonlines.open(input_path, 'r') as f:
        i = 0
        for line in f:
            prompt = line['problem']
            deepseek_solution = line['deepseek_solution']
            prompt_solution = prompt + deepseek_solution
            
            prompts_solutions.append(prompt_solution)
            # print(f"长度：{len(prompts_solutions)}")
            raw_data_dict[prompt_solution] = line
            # i += 1
    return raw_data_dict, prompts_solutions


def read_DAPO_Math_17k(input_path):
    raw_data_dict = {}
    prompts_solutions = [] 
    with jsonlines.open(input_path, 'r') as f:
        for line in f:
            prompt = line['problem']
            deepseek_solution = line['deepseek_solution']
            prompt_solution = prompt + deepseek_solution
            
            prompts_solutions.append(prompt_solution)
            # print(f"长度：{len(prompts_solutions)}")
            raw_data_dict[prompt_solution] = line
    return raw_data_dict, prompts_solutions

            
            
def read_all_prompt_data(input_path):
    prompt_key_data = {}
    id_key_data = {}
    prompts = []
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            prompt = line['prompt']
            if prompt not in prompt_key_data:
                prompts.append(prompt)
            prompt_key_data[line['prompt']] = line

            id_key_data[line['all_prompt_id']] = line
            
    return prompt_key_data, prompts, id_key_data





