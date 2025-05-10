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

# 处理文件的函数
def process_file(input_path, gpu_id, output_save_path, model_path="/models/rag_models/jina-embeddings-v3"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    
    # 生成输出文件路径
    filename = os.path.splitext(os.path.basename(input_path))[0]  # 获取文件名（不含扩展名）
    output_path = os.path.join(output_save_path, f"{filename}_dedup_eps{eps}.jsonl")
    embedding_path = os.path.join(embedding_dir, f"{filename}_embeddings.npy")
    labels_path = os.path.join(embedding_dir, f"{filename}_labels_eps{eps}.npy")
    top_10_cluster_path = os.path.join(embedding_dir, f"{filename}_top_1000_cluster_eps{eps}.json")
    
    raw_data_dict, prompts = read_data(input_path)

    
    prompts_nums = len(prompts)
    print("数据量", prompts_nums)
    # 检查并生成embedding
    if not os.path.exists(embedding_path):
        print(f'Process embeddings for {input_path} on GPU {gpu_id}')
        embeddings = encode_in_batches(prompts, model_path, batch_size=2048, gpu_id=gpu_id)
        np.save(embedding_path, embeddings)
    else:
        print(f'Load embeddings for {input_path}')
        embeddings = np.load(embedding_path)

    # 检查并生成聚类标签
    if not os.path.exists(labels_path):
        print(f'Process clustering for {input_path} on GPU {gpu_id}')
        dbscan = DBSCAN(eps=eps, min_samples=1, n_jobs=-1)  # min_samples描述了某一样本的距离为eps的邻域中样本个数的阈值
        labels = dbscan.fit_predict(embeddings)
        np.save(labels_path, labels)
        print(f"Clustering complete for {input_path}, labels saved as {labels_path}")
    else:
        print(f'Load cluster labels for {input_path}')
        labels = np.load(labels_path)

    # 将 prompts 与标签关联
    prompts_with_labels = list(zip(prompts, labels))

    # 按照聚类标签分组 prompts
    clustered_prompts = defaultdict(list)
    for prompt, label in prompts_with_labels:
        clustered_prompts[label].append(prompt)
    clustered_num = len(clustered_prompts)
    print("类别总数", clustered_num)
    deduplicate_num = prompts_nums - clustered_num
    print("去重数量", deduplicate_num)
    
    
    
    # 按照key中prompt的个数降序排序，可以据此判断每个簇中包含的prompt数量
    sorted_data = sorted(clustered_prompts.items(), key=lambda item: len(item[1]), reverse=True)  
    top_10_cluster = sorted_data[:1000]
    with open(top_10_cluster_path, 'w') as f:
        f.write(f"数据量: {prompts_nums}"+ "\n")
        f.write(f"cluster总数: {clustered_num}"+ "\n")
        f.write(f"去重数量: {deduplicate_num}"+ "\n")
        for cluster in top_10_cluster:
            f.write("\n" + "="*50 + f"重复数: {len(cluster[1])}" + "="*50 + "\n")
            for prompt in cluster[1]:
                f.write(prompt + '\n')
                f.write("-"*60 + "\n")


    # 从每个簇中选择一个代表性 prompt
    dump_list = []
    for _, clu_prompt in clustered_prompts.items():
        prompt = random.choice(clu_prompt)
        dump_list.append(raw_data_dict[prompt])

    # 保存去重后的 prompts
    with jsonlines.open(output_path, 'w') as f:
        f.write_all(dump_list)

    print(f"Finished processing and deduplication for {input_path}, output saved to {output_path}")

# 主程序
if __name__ == "__main__":
    
    # input_path = "./data/COT"
    # input_path = "./data/SFT"
    input_path = "./data/prompt"
    # input_path = "./data/perference"

    # 定义路径列表
    input_file_paths = []
    output_dir = 'outputs/'+input_path.split("/")[-1]
    output_save_paths = []
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".json"):
                input_file = os.path.join(root, file)
                input_file_paths.append(input_file)
                
                dir_name = input_file.split("/")[-2]
                output_save_path = os.path.join(output_dir, dir_name, file.split(".")[0])+ "/"
                os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
                output_save_paths.append(output_save_path)
                

    input_file_nums = len(input_file_paths)
    for j in range(3, 4):
        input_file = input_file_paths[j]
        output_save_path = output_save_paths[j]
        embedding_dir = output_save_path
        eps = 0.2  # eps描述了某一样本的邻域距离阈值：越小簇越多，保存的样本数量越多
        process_file(input_file, '3', output_save_path)
        
    
    
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