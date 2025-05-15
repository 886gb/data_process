#  1. 翻译:英文翻译为中文
指定
```json
{
    "file_path": 输入文件路径,
    "output_file_path": 输出文件路径
}
```
```bash
python ./data/translate_from_ds_api_arena.py    or 
python ./data/translate_from_ds_api_lima.py
```
# 使用ds or claude API 获取 COT（ thinking 和 response） 数据
指定
```json
{
    "input_file_path": 输入文件路径,
    "result_file_path": 输出文件路径,
    "model": 调用的API模型名称
}
```
```bash
python Cot_from_api.py --input_file_path data/EricLu/top10000_SCP_116K_train.json --result_file_path ./output/EricLu/claude/SCP_116K_train_result.json --model claude > data/logs/SCP_116K_train_result_log.txt 3>&1

tail -f data/logs/SCP_116K_train_result_log.txt
```
--resume 支持断点续传
