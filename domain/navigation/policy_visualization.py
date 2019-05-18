import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from threading import Thread
import numpy as np
from asyncio import Queue
import time
import logging


class DataClass():
    isSimFinish = False
    OBS=0
    CAUTION = 1
    FREE = 2
    GT = 3
    def __init__(self,n_row,n_col):
        self.data = np.zeros((4, n_row, n_col))
        self.color_bar =None
        self.roboX, self.roboY=0,0

    def __getitem__(self, item):
        return self.data[item]
    def __call__(self, *args, **kwargs):
        # assert args.__len__()<2, " not provided index and data"
        index,data = args
        self.data[index]=data

class AnimationClass(Thread):
    color_bar =None
    def __init__(self,dataClass):
        Thread.__init__(self)
        self.__dataClass=dataClass
        self.roboX, self.roboY =0,0
        self.__fig, self.__ax = plt.subplots()
        self.__ani = FuncAnimation(self.__fig, self.__view__, interval=100, repeat=True)
    def run(self):
        while not self.__dataClass.isSimFinish:
            time.sleep(0.1)

        logging.warning(" animation finish")
        time.sleep(3)
        plt.close()

    def init(self):
        ax1 = plt.subplot(1, 4, 1)
        plt.imshow(self.__dataClass[0])

        # if not self.color_bar:
        #     self.color_bar = plt.colorbar()
        # else:
        #     self.color_bar.remove()
        #     self.color_bar = plt.colorbar()

        ax1.set_xlabel('obstacle map')
        ax1 = plt.subplot(1, 4, 2)
        plt.imshow(self.__dataClass[1])
        ax1.set_xlabel('caution map')
        ax1 = plt.subplot(1, 4, 3)
        ax1.set_xlabel('free map')
        plt.imshow(self.__dataClass[2])
        ax1 = plt.subplot(1, 4, 4)
        ax1.set_xlabel('ground truth map')
        plt.imshow(self.__dataClass[3])



    def robot_update(self,next_position):
        self.roboX, self.roboY=next_position.j,next_position.i

    def trajectory(self,traj):
        ax1 = plt.subplot(1, 4, 4)
        # ax1.remove()
        plt.imshow(self.__dataClass[3])
        ax1.set_xlabel('ground truth map')
        X=[[x.j,x.i]for x in traj]
        X=np.array(X)
        plt.scatter(X[:,0],X[:,1])


    def __view__(self,i):
        self.init()
if __name__ == '__main__':
    data_class = DataClass()
    viz_thread = AnimationClass(data_class)
    viz_thread.start()
    plt.show()
    viz_thread.join()