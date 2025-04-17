import numpy as np
import jsonlines
import random
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import os
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor, as_completed
from read_data import read_all_prompt_data
from data_summary import compare_data_files



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

def save_combined_top_cluster_path(clustered_prompts, prompts_nums, clustered_num, deduplicate_num, combined_top_cluster_path, id_key_data, id_key_data_nums, all_raw_data_dict):
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
            # written_prompts = set()
            for prompt in cluster[1]:
                if prompt in all_raw_data_dict:
                    prompt = prompt +"\n" + \
                    f"all_prompt_id:{all_raw_data_dict[prompt]['all_prompt_id']}" + \
                    "   " + f"source_prompt_id:{all_raw_data_dict[prompt]['source_prompt_id']}" + '\n'
                    f.write(prompt)
                    f.write("-"*60 + "\n")
                
                # 将prompt添加到已写入集合
                # written_prompts.add(prompt)

                
                
                # # 如果这个prompt已经写入过，跳过
                # if prompt in written_prompts:
                #     continue
                
                # for key, value in id_key_data.items():
                #     # breakpoint()
                #     if value['prompt'] == prompt:
                #         same_prompt = prompt +"\n" + f"all_prompt_id:{key}" + "   " + f"source_prompt_id:{value['source_prompt_id']}" + '\n'
                #         f.write(same_prompt)
                #         f.write("-"*60 + "\n")
                
                # # 将prompt添加到已写入集合
                # written_prompts.add(prompt)


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
        dbscan = DBSCAN(eps=eps, min_samples=1, n_jobs=-1)
        labels = dbscan.fit_predict(embeddings)
        print("聚类完成")
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
    
    print(f"数据总量: {id_key_data_nums}")
    print("去掉完全相同prompt后的数据量: ", prompts_nums)
    print("聚类后类别总数", clustered_num)
    deduplicate_num = prompts_nums - clustered_num
    print("聚类去重数量", deduplicate_num)
    print("总去重量", id_key_data_nums- clustered_num)


    # 保存top1000的簇
    save_combined_top_cluster_path(clustered_prompts, prompts_nums, clustered_num, deduplicate_num, combined_top_cluster_path, id_key_data, id_key_data_nums, all_raw_data_dict)
    
    # final数据统计与对比
    compare_data_files(input_file_path, combined_output_path)
        
    # data_summary(input_file_paths, file_original_counts, file_dedup_counts)
# 主程序
if __name__ == "__main__":
    

    # input_file = "data/prompt_all.json"
    # input_file = "./data/SFT_all.json"
    input_file = "./data/preference_all.json"

    eps = 0.25
    embedding_dir = 'outputs/preference_all'
    os.makedirs(embedding_dir, exist_ok=True)
    output_path = embedding_dir + "/" + f"all_dedup_eps{str(eps)}"
    os.makedirs(output_path, exist_ok=True)
    

    
    # 使用新的函数处理所有文件
    process_all_files(input_file, '3', output_path, embedding_dir, eps)
