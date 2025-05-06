import time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from torch.utils.cpp_extension import load
import torch

# 加载自定义的CUDA DBSCAN实现
cuDBSCAN = load(
    name='cuDBSCAN',
    sources=['dbscan/main.cc', 'dbscan/dbscan.cu'],
    extra_cuda_cflags=['-O2']
)

def test_dbscan_accuracy_and_speed(data, eps=0.5, min_samples=5, runs=5):
    """
    测试CUDA DBSCAN与sklearn DBSCAN的精度和速度
    
    参数:
        data: numpy数组，形状为(n_samples, n_features)
        eps: float, DBSCAN的eps参数
        min_samples: int, DBSCAN的min_samples参数
        runs: int, 运行次数(用于计算平均执行时间)
    
    返回:
        dict: 包含精度和速度比较结果的字典
    """
    # 确保数据是float32类型，这对CUDA计算很重要
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # 1. 运行sklearn的DBSCAN
    sklearn_times = []
    for _ in range(runs):
        start_time = time.time()
        sklearn_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        sklearn_labels = sklearn_dbscan.fit_predict(data)
        end_time = time.time()
        sklearn_times.append(end_time - start_time)
    
    # 2. 运行CUDA DBSCAN
    cuda_times = []
    for _ in range(runs):
        start_time = time.time()
        # 转换为CUDA tensor
        cuda_data = torch.from_numpy(data).cuda()
        # 注意：这里假设你的cuDBSCAN.forward接受min_samples参数，如果不是，请调整
        cuda_labels = cuDBSCAN.forward(cuda_data, eps, min_samples).cpu().numpy()
        end_time = time.time()
        cuda_times.append(end_time - start_time)
    
    # 3. 比较结果
    # 调整标签以处理可能的标签差异（DBSCAN中-1表示噪声点）
    # 计算调整兰德指数(ARI)来比较聚类结果的相似度
    ari_score = adjusted_rand_score(sklearn_labels, cuda_labels)
    
    # 计算标签完全匹配的百分比
    match_percentage = np.mean(sklearn_labels == cuda_labels) * 100
    
    # 统计每个实现找到的簇数量
    sklearn_clusters = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
    cuda_clusters = len(set(cuda_labels)) - (1 if -1 in cuda_labels else 0)
    
    # 4. 返回结果
    return {
        "accuracy": {
            "adjusted_rand_index": ari_score,
            "match_percentage": match_percentage,
            "sklearn_clusters": sklearn_clusters,
            "cuda_clusters": cuda_clusters
        },
        "speed": {
            "sklearn_avg_time": np.mean(sklearn_times),
            "cuda_avg_time": np.mean(cuda_times),
            "speedup_factor": np.mean(sklearn_times) / np.mean(cuda_times)
        }
    }

if __name__ == "__main__":
    # 生成测试数据
    # 1. 简单的2D数据
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=100, n_features=16, centers=5, random_state=42)
    X = X.astype(np.float32)
    
    print("测试简单的2D数据:")
    results_2d = test_dbscan_accuracy_and_speed(X, eps=1.0, min_samples=5)
    print(f"精度比较:")
    print(f"  调整兰德指数: {results_2d['accuracy']['adjusted_rand_index']:.4f}")
    print(f"  标签匹配百分比: {results_2d['accuracy']['match_percentage']:.2f}%")
    print(f"  sklearn找到的簇数量: {results_2d['accuracy']['sklearn_clusters']}")
    print(f"  CUDA找到的簇数量: {results_2d['accuracy']['cuda_clusters']}")
    print(f"速度比较:")
    print(f"  sklearn平均时间: {results_2d['speed']['sklearn_avg_time']:.4f}秒")
    print(f"  CUDA平均时间: {results_2d['speed']['cuda_avg_time']:.4f}秒")
    print(f"  加速比: {results_2d['speed']['speedup_factor']:.2f}x")
    
    # 2. 更大规模的高维数据
    X_large = np.random.randn(10000, 1024).astype(np.float32)
    
    print("\n测试大规模高维数据:")
    results_large = test_dbscan_accuracy_and_speed(X_large, eps=0.5, min_samples=10)
    print(f"精度比较:")
    print(f"  调整兰德指数: {results_large['accuracy']['adjusted_rand_index']:.4f}")
    print(f"  标签匹配百分比: {results_large['accuracy']['match_percentage']:.2f}%")
    print(f"  sklearn找到的簇数量: {results_large['accuracy']['sklearn_clusters']}")
    print(f"  CUDA找到的簇数量: {results_large['accuracy']['cuda_clusters']}")
    print(f"速度比较:")
    print(f"  sklearn平均时间: {results_large['speed']['sklearn_avg_time']:.4f}秒")
    print(f"  CUDA平均时间: {results_large['speed']['cuda_avg_time']:.4f}秒")
    print(f"  加速比: {results_large['speed']['speedup_factor']:.2f}x")