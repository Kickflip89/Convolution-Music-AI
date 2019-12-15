from utils import *
import ConvNetwork as cn
import numpy as np
import matplotlib.pyplot as plt
import random

def sortOnProb(val):
    return val[0]

#turns an output of the model into a piano roll
def sample(b_x, predictions):
    llhd = list()
    sampled = b_x[:,0:4,:,:].copy()
    print(sampled.shape)
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            for time in range(len(predictions[i][j])):
                if(b_x[i][j+4][time][0] == 0):
                    pitches = list()
                    for p in range(128):
                        pitch = [predictions[i][j][time][p], p]
                        pitches.append(pitch)
                    pitches.sort(key = sortOnProb, reverse = True)
                    prob = np.random.random()
                    totalProb = 0
                    index = 0
                    for p in range(128):
                        totalProb += pitches[p][0]
                        if prob < totalProb:
                            index = pitches[p][1]
                            llhd.append(-np.log(pitches[p][0]))
                            break
                    sampled[i][j][time][index] = 1
    dat = (np.average(llhd), len(llhd))
    print(dat)
    return (sampled,dat)

#Executes Gibbs Sampling on a batch of input data.
def Gibbs(batch_x, a_min, a_max, nu, N):
    output = network.m.predict(batch_x)
    (preds, nll) = sample(batch_x, output)
    nlls = [nll]
    orig_mask = batch_x[0][4:8].copy()
    print("Orig mask", orig_mask.shape)
    for n in range(N):
        newData = preds.copy()
        prob = max(a_min, (a_max - (n*(a_max-a_min)/(nu*N))))
        print(prob)
        masks = list()
        for i in range(4):
            mask = orig_mask[i].copy()
            for time in range(64):
                if (mask[time][0] == 0):
                    if(prob < np.random.random()):
                        mask[time] = np.ones((128))
            masks.append(mask)
        masks = np.array(masks)
        dats = list()
        for i in range(len(batch_x)):
            data = newData[i][0:4].copy()
            data = data * masks
            newinst = np.concatenate((data, masks.copy()))
            dats.append(newinst)
        newData = np.array(dats)
        output = network.m.predict(newData)
        (preds, nll) = sample(newData, output)
        nlls.append(nll)
        print(np.count_nonzero(masks[:,:,0]), np.count_nonzero(newData[:,0:4,:,:]), np.count_nonzero(preds))
    return (preds, nlls)

def getValidationSet(val):
    batchData = random.sample(flist, 20)
    allMasks=list()
    batch_x = list()
    batch_y = list()
    masks = list()
    erased = 0
    #masks.append(np.ones((64,128)))
    for j in range(4):
        mask = np.ones((64,128))
        timeSteps = np.random.randint(8,16)
        end = np.random.randint(40,52)
        erased += end - timeSteps
        for time in range(timeSteps, end):
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
        allMasks.append(masks.copy())
        batch_x.append(inst)
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    allMasks = np.array(allMasks)
    #print(batch_x.shape)
    #for i in range(20):
    #print(np.count_nonzero(batch_y[i]), np.count_nonzero(batch_x[i,0:4,:,:]), np.count_nonzero(batch_x[i,4:8,:,0]),
         #erased)
    #unmask = 1. - allMasks
    #print(np.count_nonzero(unmask[i,:,:,0]))

def plotNLL(nlls):
    plt.figure()
    plt.title("Framewise Negative Log-Liklihood 1/16th Notes")
    plt.xlabel("Step in Gibbs Sampling")
    plt.ylabel("Avg NLL for Replaced Notes")
    plt.plot(range(len(nlls)), nlls, label="N=256")
    plt.legend()
    plt.show()

def Main():
    flist = makeDataList()
    random.shuffle(flist)
    val = flist[0:4].copy()
    trn = flist[4:].copy()

    network = cn.ConvNetwork()
    histories = network.TrainModel(trn)

    batch_x = getValidationSet(val)
    (preds, data) = Gibbs(batch_x, .03,.9,.85,(64*4))

    #plot NLL through GIbbs process:
    #nums = list(zip(*data))[0]
    #plotNLL(nums)

    #sample from scores
    #r = preds[random.randint(0,20)]
    #s = buildScore(r)
    #s.write('midi', 'sample.midi')
