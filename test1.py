import numpy as np
import jsonlines
import random
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import os
from sklearn.cluster import DBSCAN
# from concurrent.futures import ProcessPoolExecutor, as_completed
from read_data import read_all_prompt_data




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

def save_combined_top_cluster_path(clustered_prompts, prompts_nums, clustered_num, deduplicate_num, combined_top_cluster_path, id_key_data, id_key_data_nums):
        # 按照key中prompt的个数降序排序
    sorted_data = sorted(clustered_prompts.items(), key=lambda item: len(item[1]), reverse=True)  
    top_1000_cluster = sorted_data[:1000]
    with open(combined_top_cluster_path, 'w') as f:
        f.write(f"数据总量: {id_key_data_nums}"+ "\n")
        f.write(f"去掉完全相同prompt后的数据量: {prompts_nums}"+ "\n")
        f.write(f"聚类后类别总数: {clustered_num}"+ "\n")
        f.write(f"聚类去重数量: {deduplicate_num}"+ "\n")
        f.write(f"总去重量: {id_key_data_nums- clustered_num}"+ "\n")
        for cluster in top_1000_cluster:
            f.write("\n" + "="*50 + f"重复数: {len(cluster[1])}" + "="*50 + "\n")
            # 创建一个集合来跟踪已经写入的prompt
            written_prompts = set()
            for prompt in cluster[1]:
                # 如果这个prompt已经写入过，跳过
                if prompt in written_prompts:
                    continue
                
                for key, value in id_key_data.items():
                    # breakpoint()
                    if value['prompt'] == prompt:
                        same_prompt = prompt +"\n" + f"all_prompt_id:{key}" + "   " + f"source_prompt_id:{value['source_prompt_id']}" + '\n'
                        f.write(same_prompt)
                        f.write("-"*60 + "\n")
                
                # 将prompt添加到已写入集合
                written_prompts.add(prompt)


# 处理所有文件的函数
def process_all_files(input_file_path, gpu_id, output_dir, embedding_dir, eps, model_path="/models/rag_models/jina-embeddings-v3"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 创建一个目录来保存所有文件的合并结果
    # combined_output_dir = embedding_dir
    os.makedirs(output_dir, exist_ok=True)
    combined_output_path = os.path.join(output_dir, f"all_files_dedup_eps{eps}.jsonl")
    combined_embedding_path = os.path.join(embedding_dir, f"all_files_embeddings.npy")
    combined_labels_path = os.path.join(output_dir, f"all_files_labels_eps{eps}.npy")
    combined_top_cluster_path = os.path.join(output_dir, f"all_files_top_1000_cluster_eps{eps}.json")
    

    all_raw_data_dict, all_prompts, id_key_data = read_all_prompt_data(input_file_path)
    prompts_nums = len(all_prompts)
    id_key_data_nums = len(id_key_data)



    
    
        
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
        
        try:
            import cudf
            import cuml
            from cuml.cluster import DBSCAN as cuDBSCAN
            
            print("使用GPU加速的DBSCAN聚类（分批处理）...")
            
            # 确定批处理大小，根据GPU内存调整
            batch_size = 100000  # 根据您的GPU内存调整此值
            total_samples = embeddings.shape[0]
            labels = np.full(total_samples, -1)
            
            for i in range(0, total_samples, batch_size):
                end_idx = min(i + batch_size, total_samples)
                batch_embeddings = embeddings[i:end_idx]
                
                # 将批次数据转换为GPU上的数据
                batch_embeddings_gpu = cuml.DataFrame.from_numpy(batch_embeddings.astype(np.float32))
                
                # 在GPU上运行DBSCAN
                dbscan_gpu = cuDBSCAN(eps=eps, min_samples=1)
                batch_labels = dbscan_gpu.fit_predict(batch_embeddings_gpu).to_numpy()
                
                # 如果不是第一批，调整标签
                if i > 0:
                    max_label = np.max(labels[:i])
                    positive_mask = batch_labels >= 0
                    batch_labels[positive_mask] += max_label + 1
                
                labels[i:end_idx] = batch_labels
                print(f"已处理 {end_idx}/{total_samples} 个样本")
            
            print("GPU聚类完成")
        except (ImportError, ModuleNotFoundError):
            print("RAPIDS库未安装，回退到CPU版本的DBSCAN")
            print("要使用GPU加速，请安装RAPIDS: https://rapids.ai/start.html")
            
            # 回退到CPU版本
            print("使用CPU版本的DBSCAN...")
            dbscan = DBSCAN(eps=eps, min_samples=1, n_jobs=-1)
            labels = dbscan.fit_predict(embeddings)
            print("CPU聚类完成")
        
        np.save(combined_labels_path, labels)
        print(f"聚类完成，标签保存为 {combined_labels_path}")
    else:
        print(f'加载已有的聚类标签')
        labels = np.load(combined_labels_path)

    # 将 prompts 与标签关联
    prompts_with_labels = list(zip(all_prompts, labels))
    # 按照聚类标签分组 prompts
    clustered_prompts = defaultdict(list)
    for prompt, label in prompts_with_labels:
        clustered_prompts[label].append(prompt)
    clustered_num = len(clustered_prompts)
    

    print(f"数据总量: {id_key_data_nums}")
    print("去掉完全相同prompt后的数据量: ", prompts_nums)
    print("聚类后类别总数", clustered_num)
    deduplicate_num = prompts_nums - clustered_num
    print("聚类去重数量", deduplicate_num)
    print("总去重量", id_key_data_nums- clustered_num)
    
            
    # 从每个簇中选择一个代表性 prompt 并保留其原始数据
    dump_list = []
    for _, clu_prompt in clustered_prompts.items():
        prompt = random.choice(clu_prompt)
        if prompt in all_raw_data_dict:
            # 如果没有ID，使用原始方式处理
            dump_list.append(all_raw_data_dict[prompt])

    # 保存去重后的所有 prompts
    with jsonlines.open(combined_output_path, 'w') as f:
        print(f"最终保留数据: {len(dump_list)}")
        f.write_all(dump_list)
    
    print(f"Finished processing and deduplication for all files, output saved to {combined_output_path}")

    save_combined_top_cluster_path(clustered_prompts, prompts_nums, clustered_num, deduplicate_num, combined_top_cluster_path, id_key_data, id_key_data_nums)
        
    # data_summary(input_file_paths, file_original_counts, file_dedup_counts)
# 主程序
if __name__ == "__main__":
    
    # input_path = "./data/COT"
    # input_path = "./data/SFT"
    # input_file = "data/prompt_all.json"
    input_file = "./data/SFT_all.json"
    # input_path = "./data/perference"

    eps = 0.2
    embedding_dir = 'outputs/SFT_all'
    os.makedirs(embedding_dir, exist_ok=True)
    output_path = embedding_dir + "/" + f"all_dedup_eps{str(eps)}"
    os.makedirs(output_path, exist_ok=True)
    

    
    # 使用新的函数处理所有文件
    process_all_files(input_file, '3', output_path, embedding_dir, eps)
