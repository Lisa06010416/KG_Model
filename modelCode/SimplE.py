from .modelUnit.dataGenerator import SimplE_DataGenerator
from .modelUnit.modelUnit import baseModel, SimplE
from .modelUnit.dataUnit import dataUnit
import numpy as np
import time
import math
from keras import Model
from keras import backend as K
from keras import regularizers, activations, initializers
from keras.layers import Input, Embedding, Add, Lambda, Reshape
import os
import pandas as pd

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
        self.triplet_train = pd.read_csv(self.path + self.attriTestFile + "train.txt", sep='\t')
        if self.isValid == 1 : # 有valid
            self.SimplE_valid = pd.read_csv(self.path + "valid.txt", sep='\t')
           
        # read testRank tripley
        files = os.listdir(self.path + testfile)
        self.testRank = []
        for f in files:
            d = pd.read_csv(self.path + testfile +"/" + f, sep='\t')
            self.testRank.append(d)
                
    def preprocess(self):
        # get SimplE valid
        if self.isValid == 1 : # 有valid
            valid = self.SimplE_valid.values
            self.SimplE_validationData = ([valid[:,0],valid[:,1],valid[:,2]],valid[:,3])     
            # posKGTriplet_Train   
            self.posKGTriplet_Train = self.getPosTriplet(self.triplet_train, 1, self.path + self.attriTestFile + "posKGTriplet_Train.txt")
        else:
            # posKGTriplet_Train   
            self.posKGTriplet_Train = self.getPosTriplet(self.triplet_train, 1, self.path + self.attriTestFile + "posKGTriplet_Train_all.txt")
            
class creatModel(baseModel, SimplE):
    def __init__(self, pars, Data):
        baseModel.__init__(self, pars, Data)
        # build model
        self.buildModel()
        # data generate
        self.SeimpE_Train_DataGenerator = SimplE_DataGenerator(dataset = self.Data.triplet_train, 
                                                               entityNum = self.Data.max_e, 
                                                               posKGTriplet_Train = self.Data.posKGTriplet_Train, 
                                                               batchSize = self.batchSize, negRatio = self.negRatio)
        # isload
        if self.isLoad == 1:
            print("Load model from " + str(self.loadPath))
            self.SimplE.load_weights(self.loadPath)
            
    def createEmbedding(self):
        # embedding
        sqrt_size = 6.0 / math.sqrt(self.emSize)
        self.R1_E = Embedding(self.Data.max_r, 
                              self.emSize, 
                              embeddings_regularizer=regularizers.l2(self.alpha), 
                              embeddings_initializer = initializers.RandomUniform(minval=-sqrt_size, maxval=sqrt_size), 
                              name="R1_E", 
                              input_length=1)
        self.R2_E = Embedding(self.Data.max_r, 
                              self.emSize, 
                              embeddings_regularizer=regularizers.l2(self.alpha), 
                              embeddings_initializer = initializers.RandomUniform(minval=-sqrt_size, maxval=sqrt_size), 
                              name="R2_E", 
                              input_length=1)
        self.Eh_E = Embedding(self.Data.max_e, 
                              self.emSize, 
                              embeddings_regularizer=regularizers.l2(self.alpha), 
                              embeddings_initializer = initializers.RandomUniform(minval=-sqrt_size, maxval=sqrt_size), 
                              name="Eh_E", input_length=1)
        self.Et_E = Embedding(self.Data.max_e, 
                              self.emSize, 
                              embeddings_regularizer=regularizers.l2(self.alpha), 
                              embeddings_initializer = initializers.RandomUniform(minval=-sqrt_size, maxval=sqrt_size), 
                              name="Et_E", 
                              input_length=1)

    def buildModel(self):
        # create Embedding layer
        self.createEmbedding()
        # build SimplE model
        self.buildSimplE()
        
    def training(self):
        if self.Data.isValid == 0 :
            history = self.SimplE.fit_generator(generator=self.SeimpE_Train_DataGenerator, 
                                        epochs=1, 
                                        verbose=1, 
                                        workers= self.numWorks,
                                        use_multiprocessing=self.mulitPreprocess,
                                        shuffle=True)
        else:
            history = self.SimplE.fit_generator(generator=self.SeimpE_Train_DataGenerator, 
                                                validation_data = self.Data.SimplE_validationData, 
                                                epochs=1, 
                                                verbose=1, 
                                                workers= self.numWorks,
                                                use_multiprocessing=self.mulitPreprocess,
                                                shuffle=True)
            
        testSimplE = self.test(self.SimplE, "Triplet", self.Data.testRank)
        # getHistory
        self.getHistory(test = {"testSimplE":testSimplE}, 
                                history = history, 
                                modelName = "trainSimplE", 
                                recordList = ['loss', 'val_loss', 'SimplE_calAccuracy', 'val_SimplE_calAccuracy', "testSimplE"])
        return self.SimplE
    
    def Test(self):
        print("test on SimplE ...")
        testSimplE = self.test(self.SimplE, "Triplet", self.Data.testRank)
        return {"SimplE" : {"test" : [testSimplE]}}