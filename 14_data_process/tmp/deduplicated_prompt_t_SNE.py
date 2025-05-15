import numpy as np
import jsonlines
import random
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import os
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from concurrent.futures import ProcessPoolExecutor, as_completed
from read_data import read_OpenThoughts_114_prompts, read_data, read_all_prompt_data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境




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

def save_combined_top_cluster_path(clustered_prompts, prompts_nums, clustered_num, deduplicate_num, combined_top_cluster_path, id_key_data):
        # 按照key中prompt的个数降序排序
    sorted_data = sorted(clustered_prompts.items(), key=lambda item: len(item[1]), reverse=True)  
    top_1000_cluster = sorted_data[:1000]
    with open(combined_top_cluster_path, 'w') as f:
        f.write(f"总数据量: {prompts_nums}"+ "\n")
        f.write(f"cluster总数: {clustered_num}"+ "\n")
        f.write(f"聚类去重数量: {deduplicate_num}"+ "\n")
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
    os.makedirs(output_dir, exist_ok=True)
    combined_output_path = os.path.join(output_dir, f"all_files_dedup_eps{eps}.jsonl")
    combined_embedding_path = os.path.join(embedding_dir, f"all_files_embeddings.npy")
    combined_tsne_path = os.path.join(embedding_dir, f"all_files_tsne.npy")
    combined_labels_path = os.path.join(output_dir, f"all_files_labels_eps{eps}.npy")
    combined_top_cluster_path = os.path.join(output_dir, f"all_files_top_1000_cluster_eps{eps}.json")
    tsne_plot_path = os.path.join(output_dir, f"tsne_visualization_eps{eps}.png")
    
    all_raw_data_dict, all_prompts, id_key_data = read_all_prompt_data(input_file_path)
    prompts_nums = len(all_prompts)
    all_raw_data_dict_nums = len(all_raw_data_dict)
    print(f"all_raw_data_dict: {all_raw_data_dict_nums}")

    # print(f"id_key_data: {len(id_key_data)}")
    
    
        
    # 检查并生成embedding
    if not os.path.exists(combined_embedding_path):
        print(f'Process embeddings for all files on GPU {gpu_id}')
        embeddings = encode_in_batches(all_prompts, model_path, batch_size=2048, gpu_id=gpu_id)
        np.save(combined_embedding_path, embeddings)
    else:
        print(f'Load embeddings for all files')
        embeddings = np.load(combined_embedding_path)

    # 检查并生成t-SNE降维结果
    if not os.path.exists(combined_tsne_path):
        print(f'Performing t-SNE dimensionality reduction')
        tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, random_state=1)
        embeddings_2d = tsne.fit_transform(embeddings)
        np.save(combined_tsne_path, embeddings_2d)
        print(f"t-SNE complete, results saved as {combined_tsne_path}")
    else:
        print(f'Loading existing t-SNE results')
        embeddings_2d = np.load(combined_tsne_path)

    # 检查并生成聚类标签
    if not os.path.exists(combined_labels_path):
        print(f'Process clustering for all files on GPU {gpu_id}')
        dbscan = DBSCAN(eps=eps, min_samples=1, n_jobs=-1)
        labels = dbscan.fit_predict(embeddings_2d)
        np.save(combined_labels_path, labels)
        print(f"Clustering complete for all files, labels saved as {combined_labels_path}")
    else:
        print(f'Load cluster labels for all files')
        labels = np.load(combined_labels_path)
    breakpoint()
    
    # 提取所有prompt的ID
    all_prompt_ids = []
    for prompt in all_prompts:
        prompt_id = None
        for key, value in id_key_data.items():
            if value['prompt'] == prompt:
                prompt_id = key
                break
        all_prompt_ids.append(prompt_id)
    
    # 可视化t-SNE结果，传入prompt IDs
    visualize_tsne(embeddings_2d, labels, tsne_plot_path, all_prompt_ids)

    # 将 prompts 与标签关联
    prompts_with_labels = list(zip(all_prompts, labels))
    # 按照聚类标签分组 prompts
    clustered_prompts = defaultdict(list)
    for prompt, label in prompts_with_labels:
        clustered_prompts[label].append(prompt)
    clustered_num = len(clustered_prompts)
    

    
    
    print("总数据量: ", prompts_nums)
    print("类别总数", clustered_num)
    deduplicate_num = prompts_nums - clustered_num
    print("聚类去重数量", deduplicate_num)
    
    save_combined_top_cluster_path(clustered_prompts, prompts_nums, clustered_num, deduplicate_num, combined_top_cluster_path, id_key_data)
            
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

    
        
    # data_summary(input_file_paths, file_original_counts, file_dedup_counts)

# 添加可视化t-SNE结果的函数
def visualize_tsne(embeddings_2d, labels, output_path, all_prompt_ids=None):
    print("Generating t-SNE visualization...")
    
    # 创建一个大图
    plt.figure(figsize=(20, 16))
    
    # 获取唯一的标签
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # 为了更好的可视化效果，如果聚类数量太多，只显示前100个最大的聚类
    if n_clusters > 100:
        # 计算每个聚类的大小
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[label] = np.sum(labels == label)
        
        # 获取前100个最大的聚类
        top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:100]
        top_cluster_labels = [label for label, _ in top_clusters]
        
        # 只绘制这些聚类
        mask = np.isin(labels, top_cluster_labels)
        embeddings_2d_filtered = embeddings_2d[mask]
        labels_filtered = labels[mask]
        
        # 如果有prompt IDs，也需要过滤
        if all_prompt_ids is not None:
            all_prompt_ids_filtered = [all_prompt_ids[i] for i in range(len(mask)) if mask[i]]
            all_prompt_ids = all_prompt_ids_filtered
        
        # 更新标签和数据
        embeddings_2d = embeddings_2d_filtered
        labels = labels_filtered
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        print(f"Showing top 100 clusters out of {len(np.unique(labels))} total clusters")
    
    # 生成颜色映射
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    
    # 绘制散点图
    for i, label in enumerate(unique_labels):
        mask = labels == label
        scatter = plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            s=10,  # 点的大小
            c=[cmap(i % 20)],  # 颜色
            label=f'Cluster {label} (n={np.sum(mask)})',
            alpha=0.7  # 透明度
        )
        
        # 如果提供了prompt IDs，为每个点添加ID标签
        if all_prompt_ids is not None:
            # 为了避免过多标签导致图像混乱，只为每个簇中的少量点添加标签
            points_in_cluster = np.where(mask)[0]
            # 如果簇中的点超过5个，只标注5个
            if len(points_in_cluster) > 5:
                points_to_label = np.random.choice(points_in_cluster, 5, replace=False)
            else:
                points_to_label = points_in_cluster
                
            for idx in points_to_label:
                plt.annotate(
                    str(all_prompt_ids[idx]),
                    (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                    fontsize=8,
                    alpha=0.7
                )
    
    # 添加标题和标签
    plt.title(f't-SNE Visualization of Prompt Embeddings\n{n_clusters} clusters identified', fontsize=16)
    plt.xlabel('t-SNE dimension 1', fontsize=14)
    plt.ylabel('t-SNE dimension 2', fontsize=14)
    
    # 如果聚类数量不太多，添加图例
    if n_clusters <= 20:
        plt.legend(loc='best', fontsize=10)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"t-SNE visualization saved to {output_path}")

# 主程序
if __name__ == "__main__":
    
    # input_path = "./data/COT"
    # input_path = "./data/SFT"
    input_file = "data/test_prompt_all.json"
    # input_file = "./data/test_SFT_all.json"
    # input_path = "./data/perference"

    eps = 0.1
    embedding_dir = 'outputs/test/prompt_all'
    os.makedirs(embedding_dir, exist_ok=True)
    output_path = embedding_dir + "/" + f"all_dedup_eps{str(eps)}"
    os.makedirs(output_path, exist_ok=True)
    

    
    # 使用新的函数处理所有文件
    process_all_files(input_file, '3', output_path, embedding_dir, eps)
