import pandas as pd
import numpy as np
from functools import reduce

filename = "./map-14-4-6_3000.csv"
candidate_reward = None
def sat_filter(data):
    '''
    given the dataset get the true indices
    :param data: pandas dataframe
    :return: indexes that are true
    '''
    sat = data['sat'].tolist()
    return [i for i in range(len(sat)) if sat[i]==True]

def main():
    print(f"max reward {max_reward} len {len(sat_index)}\n")
    print("_"*50)
    assert (sat_index.__len__() > 1)
    data =[]
    for r in max_loc:
        avg = 0
        i = 0
        while (sat_index[i] <= r):
            avg += time[i]
            i += 1
            if (i >= len(sat_index) - 1):
                break
        print(f"{time[i]:.3f} -> {avg:.3f} | {r}")
        data.append(avg)
    print("_" * 50)
    data = np.array(data)
    print(f" mean {np.mean(data):.3f} std {np.std(data):.3f}")


df = pd.read_csv(filename,header=0, index_col=0)
sat_index = sat_filter(df)
sat_data = df.loc[df['sat'] == True]
max_reward = sat_data['reward'].max()
candidate_reward = candidate_reward if(candidate_reward) else max_reward
max_loc = sat_data.loc[(sat_data['reward'] == candidate_reward)]
max_loc = max_loc.index.tolist()
time = sat_data['time'].tolist()
sat_reward = sat_data['reward'].tolist()
sat_reward = np.sort(sat_reward)
print(sat_reward)


if __name__ == '__main__':
    main()
