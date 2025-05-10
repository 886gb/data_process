import numpy as np
import jsonlines
import random
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import os
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor, as_completed
from read_data import read_OpenThoughts_114_prompts, read_data




# 批量生成嵌入函数
def encode_in_batches(prompts, model_path, batch_size=32, gpu_id=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id is not None else ""
    model = SentenceTransformer(model_path, trust_remote_code=True).cuda()
    embeddings = []
    process_bar = tqdm(total=len(prompts) // batch_size, desc="processing embeddings")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_embeddings = model.encode(batch_prompts, task='text-matching', device='cuda')
        embeddings.append(batch_embeddings)
        process_bar.update(1)
    return np.vstack(embeddings)




def data_summary(input_file_paths, file_original_counts, file_dedup_counts):
    print("\n" + "="*80)
    print("每个文件的数据统计:")
    print("="*80)
    for input_path in input_file_paths:
        original_count = file_original_counts[input_path]
        dedup_count = file_dedup_counts[input_path]
        reduction = original_count - dedup_count
        reduction_percent = (reduction / original_count) * 100 if original_count > 0 else 0
        
        print(f"文件: {input_path}")
        print(f"  原始数据量: {original_count}")
        print(f"  去重后数据量: {dedup_count}")
        print(f"  减少数据量: {reduction} ({reduction_percent:.2f}%)")
        print("-"*80)
    
    print("\n总体统计:")
    total_original = sum(file_original_counts.values())
    total_dedup = sum(file_dedup_counts.values())
    total_reduction = total_original - total_dedup
    total_reduction_percent = (total_reduction / total_original) * 100 if total_original > 0 else 0
    print(f"总原始数据量: {total_original}")
    print(f"总去重后数据量: {total_dedup}")
    print(f"总减少数据量: {total_reduction} ({total_reduction_percent:.2f}%)")
    print("="*80)



# 处理所有文件的函数
def process_all_files(input_file_paths, gpu_id, output_save_paths, embedding_dir, eps, output_dir, model_path="/models/rag_models/jina-embeddings-v3"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 创建一个目录来保存所有文件的合并结果
    combined_output_dir = embedding_dir
    os.makedirs(combined_output_dir, exist_ok=True)
    combined_output_path = os.path.join(combined_output_dir, f"all_files_dedup_eps{eps}.jsonl")
    combined_embedding_path = os.path.join(output_dir, f"all_files_embeddings.npy")
    combined_labels_path = os.path.join(combined_output_dir, f"all_files_labels_eps{eps}.npy")
    combined_top_cluster_path = os.path.join(combined_output_dir, f"all_files_top_1000_cluster_eps{eps}.json")
    
    # 收集所有文件的prompts和原始数据
    all_prompts = []
    all_raw_data_dict = {}
    prompt_to_ids = defaultdict(list)  # 用于存储相同prompt对应的不同ID
    id_to_data = {}  # 用于存储ID到原始数据的映射
    file_to_indices = {}  # 记录每个文件的prompts在合并列表中的索引范围
    file_original_counts = {}  # 记录每个文件的原始数据量
    
    start_idx = 0
    for input_path in tqdm(input_file_paths, desc="Reading files"):
        raw_data_dict, prompts = read_data(input_path)
        file_to_indices[input_path] = (start_idx, start_idx + len(prompts))
        start_idx += len(prompts)
        file_original_counts[input_path] = len(prompts)
        
        all_prompts.extend(prompts)
        
        # 更新数据字典，使用唯一ID作为键
        for prompt, data in raw_data_dict.items():
            prompt_id = data.get("prompt_id_source", None)
            if prompt_id:
                prompt_to_ids[prompt].append(prompt_id)
                id_to_data[prompt_id] = data
            else:
                # 如果没有ID，使用原始方式处理
                all_raw_data_dict[prompt] = data
    
    prompts_nums = len(all_prompts)
    print("总数据量", prompts_nums)
    
    # 检查并生成embedding
    if not os.path.exists(combined_embedding_path):
        print(f'Process embeddings for all files on GPU {gpu_id}')
        embeddings = encode_in_batches(all_prompts, model_path, batch_size=2048, gpu_id=gpu_id)
        np.save(combined_embedding_path, embeddings)
    else:
        print(f'Load embeddings for all files')
        embeddings = np.load(combined_embedding_path)

    # 检查并生成聚类标签
    if not os.path.exists(combined_labels_path):
        print(f'Process clustering for all files on GPU {gpu_id}')
        dbscan = DBSCAN(eps=eps, min_samples=1, n_jobs=-1)
        labels = dbscan.fit_predict(embeddings)
        np.save(combined_labels_path, labels)
        print(f"Clustering complete for all files, labels saved as {combined_labels_path}")
    else:
        print(f'Load cluster labels for all files')
        labels = np.load(combined_labels_path)

    # 将 prompts 与标签关联
    prompts_with_labels = list(zip(all_prompts, labels))

    # 按照聚类标签分组 prompts
    clustered_prompts = defaultdict(list)
    for prompt, label in prompts_with_labels:
        clustered_prompts[label].append(prompt)
    clustered_num = len(clustered_prompts)
    print("类别总数", clustered_num)
    deduplicate_num = prompts_nums - clustered_num
    print("去重数量", deduplicate_num)
    
    # 按照key中prompt的个数降序排序
    sorted_data = sorted(clustered_prompts.items(), key=lambda item: len(item[1]), reverse=True)  
    top_1000_cluster = sorted_data[:1000]
    with open(combined_top_cluster_path, 'w') as f:
        f.write(f"数据量: {prompts_nums}"+ "\n")
        f.write(f"cluster总数: {clustered_num}"+ "\n")
        f.write(f"去重数量: {deduplicate_num}"+ "\n")
        for cluster in top_1000_cluster:
            f.write("\n" + "="*50 + f"重复数: {len(cluster[1])}" + "="*50 + "\n")
            
            # 创建一个集合来跟踪已经写入的prompt
            written_prompts = set()
            
            for prompt in cluster[1]:
                # 如果这个prompt已经写入过，跳过
                if prompt in written_prompts:
                    continue
                
                # 添加来源信息
                source_info = ""
                
                # 检查这个prompt是否有关联的ID
                breakpoint()
                if prompt in prompt_to_ids and prompt_to_ids[prompt]:
                    # 获取所有关联的ID
                    all_ids = prompt_to_ids[prompt]
                    source_info = f" [prompt_id_source: {', '.join(all_ids)}]"
                elif prompt in all_raw_data_dict:
                    data_entry = all_raw_data_dict[prompt]
                    prompt_id = data_entry.get("prompt_id_source")
                    if prompt_id:
                        source_info = f" [prompt_id_source: {prompt_id}]"
                
                f.write(prompt +"\n" + source_info + '\n')
                f.write("-"*60 + "\n")
                
                # 将prompt添加到已写入集合
                written_prompts.add(prompt)
            
    # 从每个簇中选择一个代表性 prompt 并保留其原始数据
    dump_list = []
    for _, clu_prompt in clustered_prompts.items():
        prompt = random.choice(clu_prompt)
        
        # 检查这个prompt是否有关联的ID
        if prompt in prompt_to_ids and prompt_to_ids[prompt]:
            # 如果有多个ID，随机选择一个
            prompt_id = random.choice(prompt_to_ids[prompt])
            dump_list.append(id_to_data[prompt_id])
        elif prompt in all_raw_data_dict:
            # 如果没有ID，使用原始方式处理
            dump_list.append(all_raw_data_dict[prompt])

    # 保存去重后的所有 prompts
    with jsonlines.open(combined_output_path, 'w') as f:
        f.write_all(dump_list)
    
    print(f"Finished processing and deduplication for all files, output saved to {combined_output_path}")
    file_dedup_counts = {}  # 记录每个文件去重后的数据量
    
    # 为每个原始文件创建去重后的版本（基于全局去重结果）
    for input_path, output_save_path in zip(input_file_paths, output_save_paths):
        filename = os.path.splitext(os.path.basename(input_path))[0]
        file_output_path = os.path.join(output_save_path, f"{filename}_dedup_eps{eps}.jsonl")
        
        # 获取该文件在全局去重中保留的prompts
        start_idx, end_idx = file_to_indices[input_path]
        file_prompts_with_labels = prompts_with_labels[start_idx:end_idx]
        
        # 找出该文件中每个簇的代表性prompt
        file_clusters = defaultdict(list)
        for prompt, label in file_prompts_with_labels:
            file_clusters[label].append(prompt)
        
        file_dump_list = []
        for label, prompts in file_clusters.items():
            # 如果这个簇在全局中有代表，使用该文件中的一个prompt作为代表
            if label in clustered_prompts:
                prompt = random.choice(prompts)
                
                # 检查这个prompt是否有关联的ID
                if prompt in prompt_to_ids and prompt_to_ids[prompt]:
                    # 如果有多个ID，随机选择一个
                    prompt_id = random.choice(prompt_to_ids[prompt])
                    file_dump_list.append(id_to_data[prompt_id])
                elif prompt in all_raw_data_dict:
                    # 如果没有ID，使用原始方式处理
                    file_dump_list.append(all_raw_data_dict[prompt])
                    
        file_dedup_counts[input_path] = len(file_dump_list)  # 记录去重后的数据量
        # 保存该文件的去重结果
        with jsonlines.open(file_output_path, 'w') as f:
            f.write_all(file_dump_list)
        
        print(f"Saved deduplicated version of {input_path} to {file_output_path}")
        
    data_summary(input_file_paths, file_original_counts, file_dedup_counts)
# 主程序
if __name__ == "__main__":
    
    # input_path = "./data/COT"
    # input_path = "./data/SFT"
    input_path = "./data/test"
    # input_path = "./data/perference"

    # 定义路径列表
    input_file_paths = []

    output_dir = 'outputs/'+input_path.split("/")[-1]
    output_save_paths = []
    
    for root, dirs, files in os.walk(input_path):
        for file in files:

            if file.endswith(".json"):
                # if 'rl_prompt_train' in file:
                #     continue
                input_file = os.path.join(root, file)
                input_file_paths.append(input_file)
                
                dir_name = input_file.split("/")[-2]
                output_save_path = os.path.join(output_dir, dir_name, file.split(".")[0])+ "/"
                os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
                output_save_paths.append(output_save_path)
    
    eps = 0.1  # eps描述了某一样本的邻域距离阈值：越小簇越多，保存的样本数量越多
    embedding_dir = os.path.join(output_dir, "all_dedup_eps"+str(eps))
    os.makedirs(embedding_dir, exist_ok=True)

    
    # 使用新的函数处理所有文件
    process_all_files(input_file_paths, '3', output_save_paths, embedding_dir, eps, output_dir)
    
    # # 定义 GPU ID 列表
    # gpu_ids = [0, 1]  # 假设有两个 GPU
    # max_workers = len(gpu_ids)  # 最大同时处理的任务数
    
    # # 使用进程池动态调度
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     future_to_task = {}

    #     # 提交任务
    #     for i, input_path in enumerate(input_file_paths):
    #         gpu_id = gpu_ids[i % len(gpu_ids)]  # 轮询分配 GPU
    #         future = executor.submit(process_file, input_path, gpu_id, output_save_paths[i])
    #         future_to_task[future] = input_path

    #     # 动态监控任务完成情况
    #     for future in as_completed(future_to_task):
    #         input_path = future_to_task[future]
    #         try:
    #             future.result()
    #             print(f"Processing complete for {input_path}")
    #         except Exception as e:
    #             print(f"Error processing {input_path}: {e}")