import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, add
from keras.optimizers import Adam

DEF_LR = .01
DEF_LR_DEC = .01

#gets all availble good data from Bach Chorales
def makeDataList():
    chorales = list(range(250,439))
    files = list()
    for num in chorales:
        file = "bach/bwv" + str(num) + ".xml"
        SA = ScoreAnalyzer(file)
        if SA.notFourFour():
            continue
        if SA.badData:
            continue
        files.append(file)
    return files


#Model architecture and batch generation / loss methods
class ConvNetwork:
    def __init__(self, batches=20, num_convs=20, alpha = DEF_LR, alpha_dec = DEF_LR_DEC):
        self.batchSize = batches
        self.nconvs = num_convs
        self.rolls = np.zeros((192,128))
        self.masks = np.zeros((20,4,64,128))
        self.alpha = alpha
        self.alpha_decay = alpha_dec
        self.num_erased = 0
        self.m = self.buildModel()

    def TrainModel(self, dataList):
        dlen = len(dataList)*4*8
        histories = self.m.fit_generator(self.batchGenerator(dataList), dlen/self.batchSize, 5)
        return histories

    #generates a batch while remembering the masks as part of the class
    def batchGenerator(self, dataList):
        while True:
            batchData = random.sample(dataList, self.batchSize)
            self.masks=list()
            batch_x = list()
            batch_y = list()
            self.num_erased = 0
            masks = list()
            for j in range(4):
                mask = np.ones((64,128))
                timeSteps = np.random.randint(20,30)
                self.num_erased += timeSteps
                timeSteps = np.random.choice(a=list(range(64)), size=timeSteps, replace=False)
                for time in timeSteps:
                    mask[time] = np.zeros(128)
                masks.append(mask)
            masks = np.array(masks)
            for fname in batchData:
                SA = ScoreAnalyzer(fname)
                roll = SA.transpose()
                measures = roll.shape[1]//16
                endMeasure=np.random.randint(4,(measures+1))
                yDat = roll[:,(16*endMeasure - 64):endMeasure*16,:]
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

    #custom loss function
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

    #model architecture
    def buildModel(self):
        data = Input(shape=(8,64,128))
        preconv = Conv2D(64, 3, padding='same', activation='relu', data_format="channels_first")(data)
        lastInput = BatchNormalization(axis=1)(preconv)
        for i in range(self.nconvs//2):
            conv1 = Conv2D(64, 5, padding='same', activation='relu', data_format="channels_first")(lastInput)
            bn1 = BatchNormalization(axis=1)(conv1)
            conv2 = Conv2D(64, 5, padding='same', activation='relu', data_format="channels_first")(bn1)
            bn2 = BatchNormalization(axis=1)(conv2)
            lastInput = add([lastInput,bn2])
            lastInput = BatchNormalization(axis=1)(lastInput)
        output = Conv2D(4, 3, padding='same', activation='softmax', data_format="channels_first")(lastInput)
        model = Model(inputs=data, outputs=output)
        model.compile(optimizer="Adam", loss=self.lossFunction)
        return model
