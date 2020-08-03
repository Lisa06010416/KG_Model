from .modelUnit.dataGenerator import Triplet_DataGenerator
from .modelUnit.modelUnit import baseModel
from .modelUnit.dataUnit import dataUnit
import pandas as pd
import numpy as np
import math
import os
from keras import Model
import keras.backend as K
from keras.layers import Embedding, Reshape, Input, Lambda, Multiply, Concatenate,Flatten, Dense
from keras import regularizers, activations, initializers

class dataSet(dataUnit):
    def __init__(self, path, testfile, isValid=0, attriTestFile=""):
        # set parameters
        self.attriTestFile = attriTestFile
        self.path = path
        self.isValid = isValid
        self.initPar()
        # read data
        self.readData(testfile)
        # preprocess
        self.preprocess()
    
    def initPar(self):
        numCal = pd.read_csv(self.path + "numCal.txt", sep='\t', index_col=0)
        self.max_e = numCal.loc['entituNum'].values[0]    # number of entity
        self.max_e_u = numCal.loc['euNum'].values[0]      # number of user entity
        self.max_e_i = numCal.loc['emNum'].values[0]      # number of item entity
        self.max_e_a = numCal.loc['eaNum'].values[0]      # number of attribute entity
        self.max_r = numCal.loc['relationNum'].values[0]  # number of relation

    def readData(self, testfile):  
        # read tripley
        self.triplet_train = pd.read_csv(self.path + self.attriTestFile   + "train.txt", sep='\t')
        if self.isValid == 1 : # 有valid
            self.triplet_valid = pd.read_csv(self.path + "valid.txt", sep='\t')
           
        # read testRank tripley
        files = os.listdir(self.path + testfile)
        self.testRank = []
        for f in files:
            d = pd.read_csv(self.path + testfile +"/" + f, sep='\t')
            self.testRank.append(d)
                
    def preprocess(self):
        # get valid data
        if self.isValid == 1 : # 有valid
            valid = self.triplet_validReplace.values
            self.triplet_validationData = ([valid[:,0],valid[:,1],valid[:,2]],valid[:,3])     
            # posKGTriplet_Train   
            self.posKGTriplet_Train = self.getPosTriplet(self.triplet_train, 1, self.path + self.attriTestFile +"posKGTriplet_Train.txt")
        else:
            # posKGTriplet_Train   
            self.posKGTriplet_Train = self.getPosTriplet(self.triplet_train, 1, self.path + self.attriTestFile +"posKGTriplet_Train_all.txt")
            
class creatModel(baseModel):
    def __init__(self, pars, Data):
        baseModel.__init__(self, pars, Data)
        # build model
        self.buildModel()
        # data generate
        self.TransH_DataGenerator = Triplet_DataGenerator(dataset = self.Data.triplet_train, 
                                                         entityNum = self.Data.max_e, 
                                                         posKGTriplet_Train = self.Data.posKGTriplet_Train, 
                                                         batchSize = self.batchSize)
        # isload
        if self.isLoad == 1:
            print("Load model from " + str(self.loadPath))
            self.model.load_weights(self.loadPath)
            
    def createEmbedding(self):
        # embedding
        sqrt_size = 6.0 / math.sqrt(self.emSize)
        self.WR_E  = Embedding(self.Data.max_r, self.emSize, embeddings_regularizer=regularizers.l2(self.alpha), embeddings_initializer = initializers.RandomUniform(minval=-sqrt_size, maxval=sqrt_size), name="WR_E", input_length=1)
        self.R_E = Embedding(self.Data.max_r, self.emSize, embeddings_regularizer=regularizers.l2(self.alpha), embeddings_initializer = initializers.RandomUniform(minval=-sqrt_size, maxval=sqrt_size), name="R_E", input_length=1)
        self.E_E = Embedding(self.Data.max_e, self.emSize, embeddings_regularizer=regularizers.l2(self.alpha), embeddings_initializer = initializers.RandomUniform(minval=-sqrt_size, maxval=sqrt_size), name="E_E", input_length=1)
        
    def buildModel(self):
        # create Embedding layer
        self.createEmbedding()
        # build TransH model
        self.buildTransH()
    
    def cal_wr_trans(self, vects):
        e, wr = vects
        wr_n = K.l2_normalize(wr)
        return e - K.sum(e*wr_n, axis = -1, keepdims = True)*wr_n
    
    def inverseScore(self, vects):
        return -1*vects
    
    def getDissims(self, vects):
        h_v, t_v, r_v = vects
        return K.sqrt(K.sum(K.square(h_v+r_v-t_v), axis=-1))
    
    def transE_Score(self, vects):
        pos_dissims, neg_dissims = vects
        score = K.relu(self.gamma + pos_dissims - neg_dissims)
        return score
        
    def transE_calLoss(self, yture, ypredic):
        loss = K.mean(ypredic)
        return loss
    
    def buildTransH(self):
        # input
        intputshape = (1,)
        hp = Input(shape=intputshape, name="hp", dtype="int32")
        tp = Input(shape=intputshape, name="tp", dtype="int32")
        hn = Input(shape=intputshape, name="hn", dtype="int32")
        tn = Input(shape=intputshape, name="tn", dtype="int32")
        r = Input(shape=intputshape, name="r", dtype="int32")
        
        # get vects
        hp_v = self.E_E(hp)
        tp_v = self.E_E(tp)
        hn_v = self.E_E(hn)
        tn_v = self.E_E(tn)
        r_v = self.R_E(r)
        wr_v = self.WR_E(r)
        
        # translate to wr
        hp_v = Lambda(self.cal_wr_trans, name="cal_wr_trans_hp")([hp_v, wr_v])
        tp_v = Lambda(self.cal_wr_trans, name="cal_wr_trans_tp")([tp_v, wr_v])
        hn_v = Lambda(self.cal_wr_trans, name="cal_wr_trans_hn")([hn_v, wr_v])
        tn_v = Lambda(self.cal_wr_trans, name="cal_wr_trans_tn")([tn_v, wr_v])
        
        # score
        pos_dissims = Lambda(self.getDissims, name="getPosDissims")([hp_v, tp_v, r_v]) # score smaller is better
        neg_dissims = Lambda(self.getDissims, name="getNegDissims")([hn_v, tn_v, r_v])
        score = Lambda(self.transE_Score, name="TransH_Score")([pos_dissims, neg_dissims])
        
        # build TransH model
        self.model = Model([hp, tp, hn, tn, r], score, name='TransH')       
        # compile
        self.model.compile(optimizer='adam', loss=self.transE_calLoss)
        
        # predict model
        i_pos_dissims = Lambda(self.inverseScore, name="inverseScore")(pos_dissims) # score bigger is better
        self.predictModel = Model([hp, r ,tp], i_pos_dissims, name='TransH_predictModel')  
        
    def training(self):
        if self.Data.isValid == 0 :
            history = self.model.fit_generator( generator=self.TransH_DataGenerator, 
                                                epochs=1, 
                                                verbose=1,
                                                workers= self.numWorks,
                                                use_multiprocessing=self.mulitPreprocess,
                                                shuffle=True)

        print("test on TransH ...")    
        testTransH = self.test(self.predictModel, "Triplet", self.Data.testRank)
        # getHistory
        self.getHistory(test = {"testTransH":testTransH}, 
                                history = history, 
                                modelName = "TransH", 
                                recordList = ['loss', "testTransH"])
        return self.model
    
    def Test(self):
        print("test on TransH ...")
        testTransH = self.test(self.predictModel, "Triplet", self.Data.testRank)
        return {"TransH" : {"testTransH" : [testTransH]}}