import json
import csv
import os
from collections import defaultdict

def data_file_summary(file_path):
    # 使用字典来统计每个 source_prompt_id 的数量
    prompt_id_counts = defaultdict(int)
    total_count = 0

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            source_prompt_id = data['source_prompt_id']
            
            # 提取 "数字_/" 后面的部分
            if '_/' in source_prompt_id:
                extracted_path = source_prompt_id.split('_/', 1)[1]
            else:
                extracted_path = source_prompt_id
            
            # 按提取的路径统计
            prompt_id_counts[extracted_path] += 1
            total_count += 1
    
    # 打印总体统计信息
    print()
    file_type = "原始" if file_path.split("/")[0] == "data" else "去重后"
    print(f"{file_type}总数据量: {total_count}")
    
    # 打印按 source_prompt_id 的统计
    print(f"\n按source_prompt_id统计 ({file_path}):")

    for prompt_id, count in sorted(prompt_id_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{prompt_id}: {count}")
    
    return prompt_id_counts, total_count

def save_comparison_to_csv(raw_counts, dup_counts, raw_total, dup_total, output_path):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['source_prompt_id', '原始数量', '去重后数量', '差异', '原始占比(%)', '去重后占比(%)', '保留率(%)'])
        
        # 写入总计行
        writer.writerow(['总计', raw_total, dup_total, raw_total - dup_total, 100, 100, 
                         round(dup_total / raw_total * 100, 2) if raw_total > 0 else 0])
        
        # 写入各数据源的统计
        all_prompt_ids = sorted(set(raw_counts.keys()) | set(dup_counts.keys()))
        for prompt_id in all_prompt_ids:
            raw_count = raw_counts.get(prompt_id, 0)
            dup_count = dup_counts.get(prompt_id, 0)
            diff = raw_count - dup_count
            
            # 计算百分比
            raw_percent = round(raw_count / raw_total * 100, 2) if raw_total > 0 else 0
            dup_percent = round(dup_count / dup_total * 100, 2) if dup_total > 0 else 0
            retention_rate = round(dup_count / raw_count * 100, 2) if raw_count > 0 else 0
            
            writer.writerow([prompt_id, raw_count, dup_count, diff, raw_percent, dup_percent, retention_rate])
    
    print(f"\nCSV统计结果已保存至: {output_path}")

def compare_data_files(raw_file_path, dup_file_path, output_dir=None, print_diff=True):
    """
    比较原始数据和去重后数据的统计信息
    
    参数:
        raw_file_path: 原始数据文件路径
        dup_file_path: 去重后数据文件路径
        output_dir: CSV输出目录，默认为None（自动从dup_file_path提取）
        print_diff: 是否打印差异信息，默认为True
        
    返回:
        (raw_counts, dup_counts, raw_total, dup_total): 统计结果的元组
    """
    # 获取数据统计
    raw_counts, raw_total = data_file_summary(raw_file_path)
    dup_counts, dup_total = data_file_summary(dup_file_path)
    
    # 比较原始数据和去重后数据的差异
    if print_diff:
        print("\n原始数据与去重后数据的source_prompt_id对比:")
        all_prompt_ids = sorted(set(raw_counts.keys()) | set(dup_counts.keys()))
        for prompt_id in all_prompt_ids:
            raw_count = raw_counts.get(prompt_id, 0)
            dup_count = dup_counts.get(prompt_id, 0)
            diff = raw_count - dup_count
            if diff != 0:
                print(f"{prompt_id}: 原始 {raw_count}, 去重后 {dup_count}, 差异 {diff}")
    
    # 如果未指定输出目录，则从dup_file_path提取
    if output_dir is None:
        output_dir = os.path.dirname(dup_file_path)
    
    # 保存CSV统计结果
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "summary.csv")
    save_comparison_to_csv(raw_counts, dup_counts, raw_total, dup_total, csv_path)
    
    return raw_counts, dup_counts, raw_total, dup_total

if __name__ == '__main__':
    # 示例用法
    raw_file_path = "data/prompt_all.json"
    dup_file_path = "outputs/prompt_all/all_dedup_eps0.2/all_files_dedup_eps0.2.jsonl"
    
    # 调用函数进行比较，output_dir将自动从dup_file_path提取
    compare_data_files(raw_file_path, dup_file_path)
    
    # 如果需要比较其他文件，可以再次调用函数
    # dup_file_path2 = "/aix_datas/data/SFT_all.json"
    # compare_data_files(raw_file_path, dup_file_path2)