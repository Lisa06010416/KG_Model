from keras import Model
import keras.backend as K
from keras.layers import Embedding, Reshape, Input, Dot, Lambda
from keras import regularizers, activations, initializers
import pandas as pd
import numpy as np
import os
import random
import time
import json


class SimplE():
    def SimplE_simScore(self, vets):
        r, h, t, ri, hi, ti = vets
        score = (K.sum(h*r*t, axis=2)+K.sum(hi*ri*ti, axis=2))/2
        score = K.clip(score,-20,20)
        return score

    def SimplE_calLoss(self, yture, ypredic):
        loss = K.mean(K.softplus(-yture*ypredic))
        return loss

    def SimplE_calAccuracy(self, yture, ypredic):
        return K.mean(yture*ypredic > 0)
    
    def buildSimplE(self):
        # input
        intputshape = (1,)
        h = Input(shape=intputshape, name="h", dtype="int32")
        r = Input(shape=intputshape, name="r", dtype="int32")
        t = Input(shape=intputshape, name="t", dtype="int32")
        
        # get vects
        r_v = self.R1_E(r)
        h_v = self.Eh_E(h)
        t_v = self.Et_E(t)
        ri_v = self.R2_E(r)
        hi_v = self.Et_E(h)
        ti_v = self.Eh_E(t)
        
        # score
        score = Lambda(self.SimplE_simScore, name="SimplE_simScore")([r_v, h_v, t_v, ri_v, hi_v, ti_v])
        # build simplE model
        self.SimplE = Model([h, r, t], score, name='simplE')       
        # compile
        self.SimplE.compile(optimizer='adam', loss=self.SimplE_calLoss, metrics=[self.SimplE_calAccuracy])
        
class baseModel:
    def __init__(self, pars, Data):
        # get parameters
        self.pars = pars
        self.parsePar(self.pars) 
        self.Data = Data
        self.historyRecord = {}
            
    def parsePar(self, pars):
        if 'lr' in pars.keys():
            self.learning_rate = pars['lr']
        if 'alpha' in pars.keys():    
            self.alpha = pars['alpha']
        if 'emSize' in pars.keys():
            self.emSize = pars['emSize']
        if 'epoch' in pars.keys():
            self.epoch = pars['epoch']
        if 'batchSize' in pars.keys():
            self.batchSize = pars['batchSize']
        if 'negRatio' in pars.keys():
            self.negRatio = pars['negRatio']
        if 'mulitPreprocess' in pars.keys():
            self.mulitPreprocess = pars['mulitPreprocess']
        if 'numWorks' in pars.keys():
            self.numWorks = pars['numWorks']
        if 'isLoad' in pars.keys():
            self.isLoad = pars['isLoad']
        if 'savePath' in pars.keys():
            self.savePath = pars['savePath']
        if 'loadPath' in pars.keys():
            self.loadPath = pars['loadPath']
        if 'gamma' in pars.keys():
            self.gamma = pars['gamma']
                
    def getHistory(self, test, history, modelName, recordList):
        # init
        if modelName not in self.historyRecord.keys():
            self.historyRecord[modelName] = {}
        for eva in recordList:
            if eva not in self.historyRecord[modelName].keys():
                self.historyRecord[modelName][eva] = []
        # record history
        for eva in recordList:
            try :
                self.historyRecord[modelName][eva].append(history.history[eva][0])
            except :
                try:
                    self.historyRecord[modelName][eva].append(test[eva])
                except:
                    print("history parser error!")
                    pass
        
    def saveDic(self, path, data):
        with open(path, 'w') as hisfile:
            json.dump(str(data), hisfile)
    
    def avgEmbedding(self,vets):
        h,t = vets
        return (h+t)/2
    
    def meanVector(self,vets):
        return K.mean(vets,axis=-1)
    
    def sumVector(self, vets):
        return K.sum(vets,axis=-1)
    
    def Train(self, isPreTrain = 0):
        # create dir to save model
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        print("model file at :"+str(timestamp))
        dirPath = self.savePath+timestamp
        if not os.path.isdir(dirPath):
            os.mkdir(dirPath)
            
        # save par
        with open(dirPath+"/parSet.txt", 'w') as file:
            json.dump(self.pars, file)
            
        # start tain
        for i in range(1, self.epoch+1+isPreTrain):
            print("~~~~~~~~~~~~~~~~~~~~~~ epoch:"+str(i)+" ~~~~~~~~~~~~~~~~~~~~~~")
            # train model
            if i<isPreTrain :
                model = self.preTrain()
            else:
                model = self.training()
            # save model
            model.save_weights(dirPath+ "/" + str(i)+'.h5')
            # save result
            self.saveDic(dirPath + '/result.txt', self.historyRecord)
            
    def test(self, predicModel, modelName, testData):
        hit1 = hit5 = hit10 = avgRank =  0
        # 迴圈讀每個 test set
        for test in testData:
            # 對每個 leave one out test set 都跑模型知道每個triplet的分數
            test = test.values
            if modelName == "Triplet":
                predic = predicModel.predict([test[:,0],test[:,1],test[:,2]], batch_size=None, verbose=0, steps=None)
            elif modelName == "BPR" or modelName == "NeuMF": # !!! test[:,0],test[:,2]
                predic = predicModel.predict([test[:,0],test[:,1]], batch_size=None, verbose=0, steps=None)
            elif modelName == "modifyBPR":
                predic = predicModel.predict([test[:,0],test[:,2]], batch_size=None, verbose=0, steps=None)
            elif modelName == "attentionBPR":
                validAttri = self.Data.getValidAttri(test[:,0],test[:,2])
                predic = predicModel.predict([test[:,0], test[:,1], test[:,2], validAttri], batch_size=None, verbose=0, steps=None)

            # 看正確答案有沒有在前1 5 10
            p = predic[-1]
            n = predic[0:-1]

            # 有的話把數量加1
            avgRank += np.sum(n>p)
            hit1 += np.sum(n>p) == 0
            hit5 += np.sum(n>p) < 5
            hit10 += np.sum(n>p) < 10

        avgRank /= len(self.Data.testRank)
        hit1 /= len(self.Data.testRank)
        hit5 /= len(self.Data.testRank)
        hit10 /= len(self.Data.testRank)
        
        print("avgRank : " + str(avgRank))
        print("hit1 : " + str(hit1))
        print("hit5 : " + str(hit5))
        print("hit10 : " + str(hit10))
        
        return [avgRank, hit1, hit5, hit10]