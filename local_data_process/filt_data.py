# 将原始数据集中的数据进行筛选，保留需要的数据
import json

def filter_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    for item in lines:
        print("当前数据行:")
        print(json.dumps(item, indent=4, ensure_ascii=False))
        print()
        user_input = input("是否保留此行? (y/n): ").strip().lower()
        if user_input == 'y':
            with open(output_file, 'a', encoding='utf-8')as outfile:
                outfile.write(item)

if __name__ == "__main__":
    input_file = 'repochat-arena-preference-4k/repochat-arena-preference-4k_train_chinese_test.json'  # 输入文件路径
    output_file = 'repochat-arena-preference-4k/repochat-arena-preference-4k_train_chinese_test_50.json'  # 输出文件路径
    filter_json(input_file, output_file)