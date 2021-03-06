#Copyright 2019 Luke Griswold

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, add, Activation
from keras.optimizers import Adam
import tensorflow as tf
from utils import ScoreAnalyzer

#Hyperparameters for Adam Optimizer (if used)
DEF_LR = .01
DEF_LR_DEC = .01

#Model architecture and batch generation / loss methods
class ConvNetwork:

    #Constructor
    def __init__(self, batches=20, num_convs=20, alpha = DEF_LR, alpha_dec = DEF_LR_DEC):
        self.batchSize = batches
        self.nconvs = num_convs
        self.rolls = np.zeros((192,128))
        self.masks = np.zeros((20,4,32,128))
        self.alpha = alpha
        self.alpha_decay = alpha_dec
        self.num_erased = 0
        self.m = self.buildModel()

    #Trains a model from the list of filenames in the music21 Corpus (dataList)
    def TrainModel(self, dataList):
        dlen = len(dataList)*4*7
        histories = self.m.fit_generator(self.batchGenerator(dataList), dlen/self.batchSize, 8)
        return histories

    #Generates random batches of size self.batchSize in random keys
    #and random 4-bar increments.  The erased notes are shared accross the batch
    def batchGenerator(self, dataList):
        while True:
            batchData = random.sample(dataList, self.batchSize)
            self.masks=list()
            batch_x = list()
            batch_y = list()
            self.num_erased = 0
            masks = list()
            for j in range(4):
                mask = np.ones((32,128))
                
                #erases between 10 and 20 notes for each voice
                timeSteps = np.random.randint(10,20)
                self.num_erased += timeSteps
                timeSteps = np.random.choice(a=list(range(32)), size=timeSteps, replace=False)
                for time in timeSteps:
                    mask[time] = np.zeros(128)
                masks.append(mask)
            masks = np.array(masks)
            
            #creates piano rolls for the selected file names
            for fname in batchData:
                SA = ScoreAnalyzer(fname)
                roll = SA.transpose()
                measures = roll.shape[1]//8
                endMeasure=np.random.randint(4,(measures+1))
                yDat = roll[:,(8*endMeasure - 32):endMeasure*8,:]
                ipt = yDat.copy()
                ipt = ipt * masks
                batch_y.append(yDat)
                inst = np.concatenate((ipt, masks.copy()))
                self.masks.append(masks.copy())
                batch_x.append(inst)
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            self.masks = np.array(self.masks)
            print(batch_x.shape, self.masks.shape, np.count_nonzero(batch_y), np.count_nonzero(self.masks))
            yield(batch_x,batch_y)
            
    #Custom lossFunction (see writeup)
    def lossFunction(self, y_true, y_pred):
        masks = self.masks
        masks = 1. - masks
        num = np.count_nonzero(masks[:,:,:,0])
        masks = tf.convert_to_tensor(masks, dtype=tf.float32)
        logits = tf.math.log(y_pred)
        unred_err = tf.math.multiply(logits,y_true)
        masks = tf.math.multiply(masks,y_true)
        res = tf.math.multiply(masks, logits)
        print(self.num_erased)
        return -(tf.reduce_sum(res)/num)

    #Model architecture
    def buildModel(self):
        data = Input(shape=(8,32,128))
        preconv = Conv2D(64, 3, padding='same', data_format="channels_first")(data)
        lastInput = BatchNormalization(axis=1)(preconv)
        lastInput = Activation('relu')(lastInput)
        for i in range(self.nconvs//2):
            conv1 = Conv2D(64, 5, padding='same', data_format="channels_first")(lastInput)
            bn1 = BatchNormalization(axis=1)(conv1)
            act1 = Activation('relu')(bn1)
            conv2 = Conv2D(64, 5, padding='same', data_format="channels_first")(bn1)
            bn2 = BatchNormalization(axis=1)(conv2)
            act2 = Activation('relu')(bn2)
            lastInput = add([lastInput,act2])
            lastInput = BatchNormalization(axis=1)(lastInput)
            lastInput = Activation('relu')(lastInput)
        output = Conv2D(4, 3, padding='same', activation='softmax', data_format="channels_first")(lastInput)
        model = Model(inputs=data, outputs=output)
        model.compile(optimizer="Adam", loss=self.lossFunction)
        return model
