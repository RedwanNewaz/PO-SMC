import pandas as pd
import matplotlib.pyplot as plt

TIMEOUT = 10000
def stat(filename):
    dtf = pd.read_csv(filename, header=0)
    filter_frame = dtf[dtf['elapsed_time'] < TIMEOUT]
    mean = filter_frame['elapsed_time'].mean()
    std = filter_frame['elapsed_time'].std()
    total = filter_frame['elapsed_time'].std() + filter_frame['elapsed_time'].mean()
    confidence_interval = 1.96*std/len(filter_frame)
    ratio = mean/std
    print(f"filename {filename} | mean {mean:.3f} |std {std:.3f}| total {total:.3f} ratio: {ratio:.3f}")

if __name__ == '__main__':

    stat('thread-map-14-4-7_max_particles.csv')
    stat('thread-map-14-4-6_max_particles.csv')
    stat('thread-ral-3-12-1.csv')
    stat('thread-ral-3-12-2_max_particles.csv')
    stat('thread-ral-3-12-3_max_particles.csv')
    for id in range(3):
        stat(filename="%d.csv" % id)


    # dtf["elapsed_time"].plot()
    # plt.show()

