from collections import defaultdict
import pandas as pd
import glob

folder ='ral-4-4-3'
path = './yue_ral/%s/*.csv'%folder
all_files = glob.glob(path)
print(all_files)

def format_line(line):
    group = line.split("\t")
    group = group[1].split(" ")
    last_index = len(group)-1
    group = group[last_index].split("\n")
    return group[0]

summary = defaultdict(list)
for filename in all_files:
    data = []
    with open(filename) as file:
        for i,line in enumerate(file):
            if(i>0 and i<5):
                data.append(format_line(line))
            elif(i>2):
                break
    assert len(data)==4
    summary["elapsed_time"].append(data[0])
    summary["reward"].append(data[1])
    summary["discounted_reward"].append(data[2])
    summary["sat"].append(data[3])



df = pd.DataFrame(summary)
df.to_csv('./yue_ral/%s/summary.csv'%folder)
