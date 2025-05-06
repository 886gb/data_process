#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA核函数：计算点之间的距离矩阵
__global__ void compute_distances_kernel(
    const float* data,
    float* distances,
    int n_samples,
    int n_features) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n_samples && j < n_samples) {
        float dist_sum = 0.0f;
        for (int k = 0; k < n_features; k++) {
            float diff = data[i * n_features + k] - data[j * n_features + k];
            dist_sum += diff * diff;
        }
        distances[i * n_samples + j] = sqrt(dist_sum);
    }
}

// CUDA核函数：标识核心点
__global__ void identify_core_points_kernel(
    const float* distances,
    bool* core_points,
    int n_samples,
    float eps,
    int min_pts) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_samples) {
        int count = 0;
        for (int j = 0; j < n_samples; j++) {
            if (distances[i * n_samples + j] <= eps) {
                count++;
            }
        }
        core_points[i] = (count >= min_pts);
    }
}

// CUDA核函数：扩展聚类
__global__ void expand_clusters_kernel(
    const float* distances,
    const bool* core_points,
    int* labels,
    bool* changed,
    int n_samples,
    float eps) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_samples && core_points[i] && labels[i] >= 0) {
        int my_label = labels[i];
        
        for (int j = 0; j < n_samples; j++) {
            if (distances[i * n_samples + j] <= eps) {
                if (labels[j] == -1 || (labels[j] > my_label && labels[j] != -2)) {
                    atomicExch(&labels[j], my_label);
                    *changed = true;
                }
            }
        }
    }
}

torch::Tensor forward(torch::Tensor data, float eps, int min_pts) {
    // 确保输入张量在CUDA上
    if (!data.is_cuda()) {
        data = data.cuda();
    }
    
    const at::cuda::CUDAGuard device_guard(data.device());
    
    int n_samples = data.size(0);
    int n_features = data.size(1);
    
    // 计算距离矩阵
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(data.device());
    auto distances = torch::empty({n_samples, n_samples}, options);
    
    // 为大数据集分批计算距离矩阵
    const int batch_size = std::min(1024, n_samples);
    dim3 block_dim(16, 16);
    dim3 grid_dim((batch_size + block_dim.x - 1) / block_dim.x, 
                  (batch_size + block_dim.y - 1) / block_dim.y);
    
    for (int batch_i = 0; batch_i < n_samples; batch_i += batch_size) {
        int size_i = std::min(batch_size, n_samples - batch_i);
        
        for (int batch_j = 0; batch_j < n_samples; batch_j += batch_size) {
            int size_j = std::min(batch_size, n_samples - batch_j);
            
            compute_distances_kernel<<<grid_dim, block_dim>>>(
                data.data_ptr<float>(),
                distances.data_ptr<float>(),
                n_samples,
                n_features);
        }
    }
    
    // 标识核心点
    auto core_points = torch::empty({n_samples}, options.dtype(torch::kBool));
    
    int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;
    
    identify_core_points_kernel<<<blocks, threads>>>(
        distances.data_ptr<float>(),
        core_points.data_ptr<bool>(),
        n_samples,
        eps,
        min_pts);
    
    // 初始化标签：-1表示噪声，-2表示未处理
    auto labels = torch::full({n_samples}, -2, options.dtype(torch::kInt32));
    
    // 标记非核心点为噪声点
    auto non_core_points = ~core_points;
    labels.masked_fill_(non_core_points, -1);
    
    // 初始化聚类ID
    int cluster_id = 0;
    
    // 处理每个核心点
    for (int i = 0; i < n_samples; i++) {
        // 跳过已处理和非核心点
        if (labels[i].item<int>() != -2) {
            continue;
        }
        
        // 为新的聚类分配ID
        labels[i] = cluster_id;
        
        // 扩展聚类
        auto changed = torch::ones({1}, options.dtype(torch::kBool));
        
        while (changed.item<bool>()) {
            changed.fill_(false);
            
            expand_clusters_kernel<<<blocks, threads>>>(
                distances.data_ptr<float>(),
                core_points.data_ptr<bool>(),
                labels.data_ptr<int>(),
                changed.data_ptr<bool>(),
                n_samples,
                eps);
        }
        
        cluster_id++;
    }
    
    return labels;
}