import json

with open("/Users/edy/Desktop/zt/zt_aixcoder/data/EricLu/SCP_116K_train.json","r",encoding="utf-8") as file:
# with open("/Users/edy/Desktop/zt/zt_aixcoder/data/EricLu/top10000_SCP_116K_train.json","r",encoding="utf-8") as file:
    lines = file.readlines()[:10000]
    # breakpoint()
    for line in lines:
        line = json.loads(line)
        # breakpoint()
        with open("/Users/edy/Desktop/zt/zt_aixcoder/data/EricLu/top10000_SCP_116K_train.json","a",encoding="utf-8") as file:
            file.write(json.dumps(line,ensure_ascii=False)+"\n")