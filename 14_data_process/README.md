# post_training_data_process

## 1. 下载数据
### 方式一：
登陆huggingface下载数据
```bash
python 14_data_process/load_data.py
```
### 方式二：
使用 hfd 下载数据

首先，进入14_data_process/down_load目录，给予脚本执行权限
```bash
cd 14_data_process/down_load
chmod a+x hfd.sh
```
然后，创建命令行别名
```bash
alias hfd="$PWD/hfd.sh"
```
最后下载数据, 例如：
```bash
 hfd open-r1/OpenR1-Math-220k --dataset --local-dir /aix_datas/data/COT/OpenR1-Math-220k --include "*default*" "*extended*"
```

## 2. 数据合并
将下载的数据统一格式合并到同一个文件中，以便第3步数据去重使用，保存格式为
```json
{"all_prompt_id": 合并后的数据新编号,
"source_prompt_id": 数据来源（包括所在数据集的编号和数据集名称）,
"prompt": 从原始数据中提取的prompt,
"metadata": 数据元信息}
```
例如：
```json
{"all_prompt_id": 17, "source_prompt_id": "7_/data/test/fka/awesome-chatgpt-prompts", "prompt": "I want you to act as an English pronunciation assistant for Turkish speaking people. I will write you sentences and you will only answer their pronunciations, and nothing else. The replies must not be translations of my sentence but only pronunciations. Pronunciations should use Turkish Latin letters for phonetics. Do not write explanations on replies. My first sentence is \"how the weather is in Istanbul?\"", "metadata": "{\"act\": \"English Pronunciation Helper\", \"prompt\": \"I want you to act as an English pronunciation assistant for Turkish speaking people. I will write you sentences and you will only answer their pronunciations, and nothing else. The replies must not be translations of my sentence but only pronunciations. Pronunciations should use Turkish Latin letters for phonetics. Do not write explanations on replies. My first sentence is \\\"how the weather is in Istanbul?\\\"\"}"}
```
注意：
- 完全相同的prompt在这里会被删除只保留一个
- 可以改进的点：对于同一类型数据集（Prompt SFT COT Preference），可以考虑在这一步分别统一格式保存，则不需要进行第4步数据统一格式步骤

## 3. 数据去重
去重思路：使用 jina-embeddings-v3 模型进行embedding，DBSCAN聚类，调参eps,保存聚类中top500的簇,观察聚类效果，确定eps
### 根据prompt去重。
#### 数据量较多时
参考14_data_process/deduplicated_prompt_batch.py，断点续传时，指定相同参数即可

必须指定以下参数
```json
{
    "input_file_path": 去重文件路径,
    "eps": 聚类阈值，越大去重数据越多，保留数据越少，经验值0.2左右,
    "embedding_dir": embedding文件保存路径,
    "output_path": 去重后文件保存路径,
    "gpu_id": GPU编号
}
```
运行并保存日志
```bash
python deduplicated_all_batch.py > /aix_datas/logs/preference_all/preference_all_dedup0.1_log.txt 2>&1
```
查看日志
```bash
tail -f /aix_datas/logs/preference_all/preference_all_dedup0.1_log.txt
```
最终生成以下文件

14_data_process/
└── outputs/
    └── preference_all/
        ├── dedup_eps0.3/
        │   ├── all_files_dedup_eps0.3.jsonl      # 去重后的数据文件，用于第4步数据统一格式
        │   ├── all_files_labels_eps0.3.npy       # 去重后的标签数据
        │   ├── all_files_top_500_cluster_eps0.3.json  # 前500个聚类结果，用于观察聚类效果
        │   └── summary.csv                       # 各个数据集去重前后的统计信息
        └── embeddings/
            └── all_files_embeddings.npy          # 所有文件的嵌入向量，可以重复使用


#### 数据量较少时
参考14_data_process/deduplicated_prompt_less.py


### 根据promt+solution去重。
参考14_data_process/deduplicated_prompt_solution_test.py

## 4. 数据统一格式
将第3步去重后的数据(SFT COT Preference)分别统一格式保存

```bash
python 14_data_process/Unified_data_format.py
```
指定参数，例如：
```json
{
    "input_file": "outputs/SFT_all/all_dedup_eps0.2/all_files_dedup_eps0.2.jsonl",
    "output_file": "outputs/SFT_all/all_dedup_eps0.2/all_files_unified_eps0.2.jsonl",
    "log_file": "logs/Unified_data_format/SFT_all/all_files_dedup_eps0.2.log"
}
```

### SFT数据
保存格式为
```json
{"all_prompt_id": 合并后的数据编号,
"source_prompt_id": 数据来源（包括所在数据集的编号和数据集名称）,
"prompt": 从原始数据中提取的prompt,
"answer": 从原始数据中提取的answer,
"metadata": 数据元信息}
```

### cot数据
保存格式为
```json
{"all_prompt_id": 合并后的数据编号,
"source_prompt_id": 数据来源（包括所在数据集的编号和数据集名称）,
"prompt": 从原始数据中提取的prompt,
"thinking": 从原始数据中提取的thinking,
"answer": 从原始数据中提取的answer,
"metadata": 数据元信息}
```
注意：
- GAIR/LIMO 和 AI-MO/NuminaMath-CoT 数据集，元数据中推理和答案无法分开保存，放在了 thinking 中，answer 是空

### preference数据
保存格式为
```json
{"all_prompt_id": 合并后的数据编号,
"source_prompt_id": 数据来源（包括所在数据集的编号和数据集名称）,
"prompt": 从原始数据中提取的prompt,
"chosen": 从原始数据中提取的chosen,
"rejected": 从原始数据中提取的rejected,
"metadata": 数据元信息}
```

注意：
- HelpSteer3 metadata 原始数据集规则：
    - -3：response 1 比 response 2 好得多
    - -2：response 1 优于 response 2
    - -1：response 1 比 response 2 稍好
    - 0：response 1 与 response 2 大致相同
    - 1：response 2 比 response 1 略好
    - 2：response 2 优于 response 1
    - 3：response 2 比 response 1 好得多

        - 当score为-3，-2，-1时，表示chosen = response 1    rejected = response 2
        - 当score为0时，表示平局  chosen = ""    rejected = ""
        - 当score为1，2，3时，表示chosen = rejected = ""

- 当是平局时，chosen和rejected都为空，例如：lmarena-ai 和 webdev-arena-preference-10k数据集

- arena-human-preference-100k 包含多轮对话，prompt  chosen  rejected  列表形式存储多轮对话内容

- OpenR1-Math-220k包含2-4条推理轨迹 如果chosen和rejected都为空，可能是平局也可能是少于两条推理或4条推理的平局