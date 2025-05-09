import json
file_path = "/aix_datas/outputs/test/process/test_preference_all.json"
with open(file_path,"r") as f :
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        breakpoint()