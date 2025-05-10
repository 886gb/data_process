import json

# with open("/Users/edy/Desktop/zt/zt_aixcoder/data/output/EricLu/claude/SCP_116K_train_result.json","r",encoding="utf-8") as file:
with open("/Users/edy/Desktop/zt/zt_aixcoder/data/output/EricLu/claude/SCP_116K_train_result.json","r",encoding="utf-8") as file:
    lines = file.readlines()
    breakpoint()
    for line in lines[8000:]:
        line = json.loads(line)
        breakpoint()
        with open("/Users/edy/Desktop/zt/zt_aixcoder/data/EricLu/post5000_SCP_116K_train.json","a",encoding="utf-8") as file:
            file.write(json.dumps(line,ensure_ascii=False)+"\n")