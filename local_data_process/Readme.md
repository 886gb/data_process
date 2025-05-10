#  1. 翻译:英文翻译为中文
```
python ./data/translate_from_ds_api_arena.py    or 
python ./data/translate_from_ds_api_lima.py
```
# 使用ds or claude API 获取 COT（ thinking 和 response） 数据

```
python Cot_from_api.py --input_file_path /Users/edy/Desktop/zt/zt_aixcoder/data/EricLu/top10000_SCP_116K_train.json --result_file_path ./output/EricLu/claude/SCP_116K_train_result.json --model claude >/Users/edy/Desktop/zt/zt_aixcoder/data/logs/SCP_116K_train_result_log.txt 3>&1

tail -f /Users/edy/Desktop/zt/zt_aixcoder/data/logs/SCP_116K_train_result_log.txt
```
--resume 支持断点续传
