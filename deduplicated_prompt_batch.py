import numpy as np
import jsonlines
import random
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import os
import gc
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
# from read_data import read_all_prompt_data
from data_summary import compare_data_files


# 分批读取数据函数
def read_data_in_batches(file_path, batch_size=10000):
    """分批读取大型JSON文件"""
    total_count = 0
    
    # 先计算文件总行数（可选，用于进度显示）
    print(f"开始统计文件行数...")
    with open(file_path, 'r') as f:
        for _ in tqdm(f):
            total_count += 1
    print(f"文件总行数: {total_count}")
    
    # 分批读取
    with open(file_path, 'r') as f:
        batch_data = []
        for i, line in enumerate(tqdm(f, total=total_count, desc="读取数据")):
            try:
                data = json.loads(line)
                batch_data.append(data)
                
                if len(batch_data) >= batch_size:
                    yield batch_data
                    print(f"已处理 {(i+1)} 条数据")
                    batch_data = []
                    
            except json.JSONDecodeError:
                print(f"无法解析第 {i+1} 行的JSON数据")
                continue
        
        # 处理最后一批数据
        if batch_data:
            yield batch_data

# 批量生成嵌入函数
def encode_in_batches(prompts, model_path, batch_size=32, gpu_id=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id is not None else ""
    model = SentenceTransformer(model_path, trust_remote_code=True).cuda()
    embeddings = []
    process_bar = tqdm(total=len(prompts) // batch_size + 1, desc="processing embeddings")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_embeddings = model.encode(batch_prompts, task='text-matching', device='cuda')
        embeddings.append(batch_embeddings)
        process_bar.update(1)
    return np.vstack(embeddings)

def save_combined_top_cluster_path(clustered_prompts, prompts_nums, clustered_num, deduplicate_num, combined_top_cluster_path, total_records, all_raw_data_dict):
    # 按照key中prompt的个数降序排序
    sorted_data = sorted(clustered_prompts.items(), key=lambda item: len(item[1]), reverse=True)  
    top_500_cluster = sorted_data[:500]
    with open(combined_top_cluster_path, 'w') as f:
        f.write(f"数据总量: {total_records}"+ "\n")
        f.write(f"去掉完全相同prompt后的数据量: {prompts_nums}"+ "\n")
        f.write(f"聚类后类别总数: {clustered_num}"+ "\n")
        f.write(f"聚类去重数量: {deduplicate_num}"+ "\n")
        f.write(f"总去重量: {total_records - clustered_num}"+ "\n")
        for cluster in top_500_cluster:
            f.write("\n" + "="*50 + f"重复数: {len(cluster[1])}" + "="*50 + "\n")
            for prompt in cluster[1]:
                if prompt in all_raw_data_dict:
                    prompt = prompt +"\n" + \
                    f"all_prompt_id:{all_raw_data_dict[prompt]['all_prompt_id']}" + \
                    "   " + f"source_prompt_id:{all_raw_data_dict[prompt]['source_prompt_id']}" + '\n'
                    f.write(prompt)
                    f.write("-"*60 + "\n")

# 处理所有文件的函数 - 分批处理版本
def process_all_files_in_batches(input_file_path, gpu_id, output_dir, embedding_dir, eps, 
                                batch_size=10000, embedding_batch_size=2048,
                                model_path="/models/rag_models/jina-embeddings-v3"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)
    
    # 输出文件路径
    combined_output_path = os.path.join(output_dir, f"all_files_dedup_eps{eps}.jsonl")
    combined_embedding_path = os.path.join(embedding_dir, f"all_files_embeddings.npy")
    combined_labels_path = os.path.join(output_dir, f"all_files_labels_eps{eps}.npy")
    combined_top_cluster_path = os.path.join(output_dir, f"all_files_top_500_cluster_eps{eps}.json")
    
    
    # 断点续传检查点文件
    checkpoint_file = os.path.join(output_dir, "processing_checkpoint.json")
    
    # 检查是否有断点续传
    start_batch = 0
    all_prompts = []
    all_raw_data_dict = {}
    total_records = 0
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_batch = checkpoint.get('processed_batches', 0)
            total_records = checkpoint.get('total_records', 0)
            print(f"从断点继续: 已处理 {start_batch} 批, 共 {total_records} 条记录")
            
            # 如果有保存的中间结果，加载它们
            if os.path.exists(os.path.join(output_dir, "all_prompts_temp.npy")):
                all_prompts = np.load(os.path.join(output_dir, "all_prompts_temp.npy"), allow_pickle=True).tolist()
            if os.path.exists(os.path.join(output_dir, "all_raw_data_dict_temp.json")):
                with open(os.path.join(output_dir, "all_raw_data_dict_temp.json"), 'r') as f:
                    breakpoint()
                    for line in f:
                        print(line)
                    
                    all_raw_data_dict = json.load(f)
    
    # 分批读取和处理数据
    batch_idx = 0
    for batch_data in read_data_in_batches(input_file_path, batch_size):
        if batch_idx < start_batch:
            batch_idx += 1
            continue
            
        print(f"处理第 {batch_idx+1} 批数据 (每批 {batch_size} 条)")
        
        # 处理这一批数据
        batch_prompts = []
        batch_raw_data_dict = {}
        
        for item in batch_data:
            prompt = item.get('prompt', '')
            if prompt and prompt not in batch_raw_data_dict:
                batch_prompts.append(prompt)
                batch_raw_data_dict[prompt] = item
        
        # 合并到总数据中
        all_prompts.extend(batch_prompts)
        all_raw_data_dict.update(batch_raw_data_dict)
        total_records += len(batch_data)
        
        # 保存断点
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'processed_batches': batch_idx + 1,
                'total_records': total_records
            }, f)
            
        # 每隔一定批次保存中间结果
        if (batch_idx + 1) % 10 == 0:
            np.save(os.path.join(output_dir, "all_prompts_temp.npy"), np.array(all_prompts, dtype=object))
            with open(os.path.join(output_dir, "all_raw_data_dict_temp.json"), 'w') as f:
                json.dump(all_raw_data_dict, f)
            print(f"已保存中间结果，当前处理了 {total_records} 条记录")
        
        batch_idx += 1
        
        # 释放内存
        del batch_data, batch_prompts, batch_raw_data_dict
        gc.collect()
    
    # 数据读取完成，保存最终的中间结果
    np.save(os.path.join(output_dir, "all_prompts_temp.npy"), np.array(all_prompts, dtype=object))
    with open(os.path.join(output_dir, "all_raw_data_dict_temp.json"), 'w') as f:
        json.dump(all_raw_data_dict, f)
    
    prompts_nums = len(all_prompts)
    print(f"数据读取完成，共 {total_records} 条原始记录，完全相同prompt去重后 {prompts_nums} 条")
    
    # 分批生成嵌入
    if not os.path.exists(combined_embedding_path):
        print(f'开始为所有数据生成嵌入向量，使用GPU {gpu_id}')
        
        # 分批处理嵌入
        all_embeddings = []
        for i in range(0, len(all_prompts), embedding_batch_size*10):
            batch_end = min(i + embedding_batch_size*10, len(all_prompts))
            print(f"处理嵌入批次 {i//embedding_batch_size} 到 {batch_end//embedding_batch_size}")
            
            batch_prompts = all_prompts[i:batch_end]
            batch_embeddings = encode_in_batches(batch_prompts, model_path, batch_size=embedding_batch_size, gpu_id=gpu_id)
            
            # 保存这一批的嵌入
            batch_embedding_path = os.path.join(embedding_dir, f"embeddings_batch_{i//embedding_batch_size}.npy")
            np.save(batch_embedding_path, batch_embeddings)
            
            # 释放内存
            del batch_prompts, batch_embeddings
            gc.collect()
        
        # 合并所有批次的嵌入
        print("合并所有批次的嵌入...")
        embeddings_files = [os.path.join(embedding_dir, f) for f in os.listdir(embedding_dir) 
                           if f.startswith("embeddings_batch_") and f.endswith(".npy")]
        embeddings_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        all_embeddings = []
        for emb_file in tqdm(embeddings_files, desc="合并嵌入"):
            batch_emb = np.load(emb_file)
            all_embeddings.append(batch_emb)
            
        embeddings = np.vstack(all_embeddings)
        np.save(combined_embedding_path, embeddings)
        
        # 清理临时文件
        for emb_file in embeddings_files:
            os.remove(emb_file)
    else:
        print(f'加载已有的嵌入向量')
        embeddings = np.load(combined_embedding_path)

    # 检查并生成聚类标签
    if not os.path.exists(combined_labels_path):
        print(f'开始聚类处理，使用GPU {gpu_id}')
        start_time = time.time()
        # 使用更小的批次进行DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=1, n_jobs=32)
        labels = dbscan.fit_predict(embeddings)
        print("聚类完成，用时：", time.time() - start_time)
        np.save(combined_labels_path, labels)
        print(f"聚类完成，标签已保存至 {combined_labels_path}")
    else:
        print(f'加载已有的聚类标签')
        labels = np.load(combined_labels_path)
    """
    blue 计算相似度——————————————————————————————————————————————————————————————————————————
    """
    # 将 prompts 与标签关联
    prompts_with_labels = list(zip(all_prompts, labels))
    
    # 按照聚类标签分组 prompts
    clustered_prompts = defaultdict(list)
    for prompt, label in prompts_with_labels:
        clustered_prompts[label].append(prompt)
    clustered_num = len(clustered_prompts)
    
    # 从每个簇中选择一个代表性 prompt 并保留其原始数据
    dump_list = []
    for _, clu_prompt in tqdm(clustered_prompts.items(), desc="生成去重结果"):
        prompt = random.choice(clu_prompt)
        if prompt in all_raw_data_dict:
            dump_list.append(all_raw_data_dict[prompt])
# ————————————-------------------------------------------------------------
    # 保存去重后的所有 prompts
    with jsonlines.open(combined_output_path, 'w') as f:
        print(f"最终保留数据: {len(dump_list)}")
        f.write_all(dump_list)
    
    print(f"所有文件处理和去重完成，输出已保存至 {combined_output_path}")
    
    print(f"数据总量: {total_records}")
    print("去掉完全相同prompt后的数据量: ", prompts_nums)
    print("聚类后类别总数", clustered_num)
    deduplicate_num = prompts_nums - clustered_num
    print("聚类去重数量", deduplicate_num)
    print("总去重量", total_records - clustered_num)

    # 保存top500的簇
    save_combined_top_cluster_path(clustered_prompts, prompts_nums, clustered_num, deduplicate_num, 
                                  combined_top_cluster_path, total_records, all_raw_data_dict)
    
    print(f"top500的簇已保存至 {combined_top_cluster_path}")
    
    # 清理临时文件
    if os.path.exists(os.path.join(output_dir, "all_prompts_temp.npy")):
        os.remove(os.path.join(output_dir, "all_prompts_temp.npy"))
    if os.path.exists(os.path.join(output_dir, "all_raw_data_dict_temp.json")):
        os.remove(os.path.join(output_dir, "all_raw_data_dict_temp.json"))
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        
    print(f"开始统计数据")
    # final数据统计与对比
    compare_data_files(input_file_path, combined_output_path)
    

    
    print("处理完成！")


# 主程序  
if __name__ == "__main__":
    input_file_path = "./data/preference_all.json"
    eps = 0.1
    embedding_dir = 'outputs/preference_all/embeddings'
    output_path = 'outputs/preference_all/dedup_eps' + str(eps)
    
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    # 使用分批处理函数
    process_all_files_in_batches(
        input_file_path=input_file_path,
        gpu_id='2',
        output_dir=output_path,
        embedding_dir=embedding_dir,
        eps=eps,
        batch_size=50000,  # 每批读取的数据量
        embedding_batch_size=1024  # 每批处理的嵌入量
    )


# python deduplicated_all_batch.py > /aix_datas/logs/preference_all/preference_all_dedup0.1_log.txt 2>&1
# tail -f /aix_datas/logs/preference_all/preference_all_dedup0.1_log.txt