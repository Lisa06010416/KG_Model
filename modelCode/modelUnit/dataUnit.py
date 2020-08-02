import json
import numpy as np
class dataUnit:
    def saveDic(self, path, data):
        with open(path, 'w') as hisfile:
            json.dump(data, hisfile)
            
    def getPosTriplet(self, dataSet, isLoad, path):
        if isLoad==1:
            try :
                with open(path) as handle:
                    posTriplet = json.loads(handle.read())
                return posTriplet
            except:
                print("no posKGTriplet_Train.txt ... preprocess ...")
        
        posTriplet = {'replaceHead':{}, 'replaceTail':{}}
        # init
        for key in posTriplet:
            for r in range(self.max_r):
                posTriplet[key][str(r)] = {}
        # 1. replace head
        relationTailData = dataSet.drop('head', axis=1).drop_duplicates().values.tolist()
        for r, t in relationTailData:
            maskR = dataSet['relation']== r
            maskT = dataSet['tail']== t
            headSet = dataSet[(maskR & maskT)]['head'].values.tolist()
            posTriplet['replaceHead'][str(r)][str(t)] = headSet
        # 2. replace tail
        headRelationData = dataSet.drop('tail', axis=1).drop_duplicates().values.tolist()
        for h, r in headRelationData:
            maskH = dataSet['head']== h
            maskR = dataSet['relation']== r    
            tailSet = dataSet[(maskH & maskR)]['tail'].values.tolist()
            posTriplet['replaceTail'][str(r)][str(h)] = tailSet
        self.saveDic(path, posTriplet)
        return posTriplet

    def getUserRating(self, dataSet):
        userRating = {}
        ratingGroup = dataSet.groupby('user')
        for user, group in ratingGroup:
            userRating[str(user)] = list(set(group['item'].values.tolist()))
        return userRating
    
    def getItemRating(self, dataSet):
        itemRating = {}
        ratingGroup = dataSet.groupby('item')
        for item, group in ratingGroup:
            itemRating[str(item)] = list(set(group['user'].values.tolist()))
        return itemRating