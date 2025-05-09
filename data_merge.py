import os
import json
from pathlib import Path
from read_data import read_data

def process_files(directory_path, output_file):
    """
    递归读取指定目录及其子目录下的所有文件，为每条数据添加meta键，并保存到一个文件中
    每行一条JSON数据
    
    Args:
        directory_path: 要处理的目录路径
        output_file: 输出文件路径
    """
    # 确保目录存在
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在")
        return
    
    # 计数器
    count = 0
    
    # 打开输出文件，准备逐行写入
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".json") or file.endswith(".jsonl"):
                    file_path = os.path.join(root, file)
                    
                    print(file_path)
                    # breakpoint()
                    # if file_path != "./data/preference/HelpSteer3/HelpSteer3_edit.json":
                    #     continue
                    # if "./data/preference/HelpSteer3" not in file_path:
                    #     continue
                    # breakpoint()
                    
                    lines, prompts = read_data(file_path, read_limit_num=-1)
                    print(f"文件 {file_path} 共有 {len(lines)} 条数据, prompts 共有 {len(prompts)} 条")
                    print()
                    for i, (line, prompt) in enumerate(zip(lines, prompts)):
                        line = line.strip()
                        data_source = str(i) + "_" + file_path.split(".")[1] 
                        data_with_meta = {"all_prompt_id": count,
                                          "source_prompt_id": data_source,
                                          "prompt": prompt,
                                          "metadata": line}
                                
                                # 将JSON对象写入一行
                        out_f.write(json.dumps(data_with_meta, ensure_ascii=False) + '\n')
                        count += 1
                    
                # print(f"已处理文件: {file_path}")
    
    print(f"处理完成，共处理 {count} 条数据，已保存到 {output_file}")

if __name__ == "__main__":
    # 设置要处理的目录和输出文件
    # directory_to_process = "./data/test"
    # directory_to_process = "./data/SFT"
    directory_to_process = "./data/preference"
    # directory_to_process = "./data/COT"
    
    # 获取父目录并设置输出文件路径
    output_file_path = os.path.dirname(directory_to_process) +"/"+directory_to_process.split("/")[-1] + "_test.json"
    
    # 处理文件
    process_files(directory_to_process, output_file_path)
