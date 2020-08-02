from keras.utils import Sequence
import pandas as pd
import numpy as np
import random
import math
import os

class Triplet_DataGenerator(Sequence):
    def __init__(self, dataset, entityNum, posKGTriplet_Train, batchSize=32, negRatio=1):
            # 初始化參數
            self.dataset = dataset
            self.datasetList = dataset.values.tolist()
            self.entityNum = entityNum
            self.entityList = range(self.entityNum)
            self.posKGTriplet_Train = posKGTriplet_Train
            self.batchSize = batchSize
            self.negRatio = negRatio

    def __len__(self):
        # 計算一個epoch需要多少batch
        self.indexes = np.arange(len(self.dataset))
        return int(np.floor(len(self.dataset) / self.batchSize))

    def __getitem__(self, index):
        # 第幾個batch 要拿index多少到多少的資料
        indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]

        # 產生 label
        Y = []

        # 產生 traing batch
        H = []
        R = []
        T = []

        # pos data
        pos = np.array([self.datasetList[k] for k in indexes])

        # trainData
        traindata = []
        # generate neg data
        for d in pos:  
            # sample negative
            negSample = random.sample(self.entityList, 20)
            # random to replace head/tail
            if random.uniform(0,1) <0.5: # head
                # 找到 (h, r, ...) 在dataset中有接過的 entity list => tailSet
                tailSet = set(self.posKGTriplet_Train['replaceTail'][str(d[1])][str(d[0])])
                negTailList = list(set(negSample) - tailSet)
                nt = random.choice(negTailList)
                traindata.append([d[0],d[2],d[0],nt,d[1]])
            else:
                # 找到 (..., r, t) 在dataset中有接過的 entity list => headSet
                headSet = set(self.posKGTriplet_Train['replaceHead'][str(d[1])][str(d[2])])
                negHeadList = list(set(negSample) - headSet)
                nh = random.choice(negHeadList)
                traindata.append([d[0],d[2],nh,d[2],d[1]])
                
        traindata = np.array(traindata)                          
        X = [traindata[:,0], traindata[:,1], traindata[:,2], traindata[:,3], traindata[:,4]]
        Y = np.array([0 for i in range(len(traindata[:,0]))])
        return X, np.array(Y)
    
class SimplE_DataGenerator(Sequence):
    def __init__(self, dataset, entityNum, posKGTriplet_Train, batchSize=32, negRatio=1):
        # 初始化參數
        self.dataset = dataset
        self.datasetList = dataset.values.tolist()
        self.entityNum = entityNum
        self.entityList = range(self.entityNum)
        self.posKGTriplet_Train = posKGTriplet_Train
        self.batchSize = batchSize
        self.negRatio = negRatio

    def __len__(self):
        # 計算一個epoch需要多少batch
        self.indexes = np.arange(len(self.dataset))
        return int(np.floor(len(self.dataset) / self.batchSize))

    def __getitem__(self, index):
        # 第幾個batch 要拿index多少到多少的資料
        indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]

        # 產生 label
        Y = []

        # 產生 traing batch
        H = []
        R = []
        T = []

        # pos data
        pos = np.array([self.datasetList[k] for k in indexes])
        H.extend(pos[:,0])
        R.extend(pos[:,1])
        T.extend(pos[:,2])
        Y.extend([1 for k in indexes])
        
        # generate neg data
        for d in pos:
            # 抽負樣本 negRatio*2
            negSample = random.sample(self.entityList, self.negRatio*2)
            # 找到 (h, r, ...) 在dataset中有接過的 entity list => tailSet
            tailSet = set(self.posKGTriplet_Train['replaceTail'][str(d[1])][str(d[0])])
            # 找到 (..., r, t) 在dataset中有接過的 entity list => headSet
            headSet = set(self.posKGTriplet_Train['replaceHead'][str(d[1])][str(d[2])])
            # 刪掉存在於資料裡的
            tailList = list(set(negSample[0:self.negRatio]) - tailSet)
            headList = list(set(negSample[self.negRatio:self.negRatio*2]) - headSet)
            # 大於negRatio 拿前negRatio個
            if len(tailList)>math.ceil(self.negRatio/2) :
                tailList = tailList[0:int(math.ceil(self.negRatio/2))]
            if len(headList)>math.floor(self.negRatio/2) :
                headList = headList[0:int(math.floor(self.negRatio/2))]
                
            # H
            H.extend([d[0] for i in tailList])
            H.extend(headList)
            # R
            R.extend([d[1] for i in range(len(tailList)+len(headList))])
            # T
            T.extend(tailList)
            T.extend(d[2] for i in headList)
            # Y
            Y.extend([-1 for i in range(len(tailList)+len(headList))])
              
        X = [np.array(H), np.array(R), np.array(T)]
        return X, np.array(Y)