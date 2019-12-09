#This version of code is set up for 1/16th temporality

import numpy as np
from music21 import *
import random

#This class handles the manipulation of musicXML files
class ScoreAnalyzer:
    def __init__(self, fname=None):
        self.fname = fname
        self.roll = None
        self.badData = False
        self.s = corpus.parse(fname)
        self.getPianoRoll()

    #Turns a musicXML file into a piano roll
    def getPianoRoll(self):
        parts = list()
        badData = False
        for part in self.s.getElementsByClass(stream.Part):
            if(badData):
                return
            roll = np.zeros(((16*len(part.getElementsByClass(stream.Measure))),128))
            measureNum = 0
            for measure in part.getElementsByClass(stream.Measure):
                if(badData):
                    return
                for note in measure.notes:
                    onset = int(4*note.offset)
                    pitch = int(note.pitch.ps)
                    duration = int(note.duration.quarterLength*4)
                    for i in range(duration):
                        try:
                            roll[(16*measureNum)+i+onset][pitch] = 1
                        except:
                            print("Error with measure", measureNum, "in", self.fname)
                            self.badData = True
                            badData = True
                if(badData):
                    return
                measureNum += 1
            parts.append(roll)
            #print(roll.T)
            #print(roll.shape)
        self.roll = np.array(parts)

    def transpose(self, step=None):
        if(not step):
            step = np.random.randint(0,12)
        up = np.random.randint(0,2)
        newroll = self.roll.copy()
        for k in range(self.roll.shape[0]):
            for i in range(self.roll.shape[1]):
                for j in range(self.roll.shape[2]):
                    if(self.roll[k][i][j]==1):
                        newroll[k][i][j] = 0
                        if(up):
                            newj = j+step
                        else:
                            newj = j-step
                        newroll[k][i][newj] = 1
        return newroll[0:4,:,:]

    def notFourFour(self):
        time = int(self.s.getElementsByClass(stream.Part)[0].getElementsByClass(stream.Measure)[0].barDuration.quarterLength)
        return (time != 4)

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

def buildScore(r):
    s = stream.Score()
    for part in r:
        p1 = stream.Part()
        m = stream.Measure()
        lastPitch=None
        currNote = None
        currDuration = 0
        currOnset = 0
        for time in range(part.shape[0]):
            #print(time)
            pitcharr = np.nonzero(part[time])[0]
            if(len(pitcharr) > 0):
                currPitch = pitcharr[0]
            if time == 0:
                currNote = currPitch
                currDuration = 0
                currOnset = 0
            if currPitch != currNote:
                n = note.Note(currNote)
                n.quarterLength = currDuration / 4
                n.onset = currOnset
                print(n.pitch, n.quarterLength, n.onset)
                m.append(n)
                lastPitch = currNote
                currNote = currPitch
                currDuration = 1
                currOnset = (time%16) / 4
            else:
                currDuration += 1
            if time > 0 and (time) % 16 == 0:
                if(currDuration > 1):
                    n = note.Note(currNote)
                    n.quarterLength = (currDuration-1)/4
                    n.onset = currOnset
                    print(n.pitch, n.quarterLength, n.onset)
                    currOnset = 0
                    currDuration = 1
                    m.append(n)
                print("Measure over")
                p1.append(m)
                m = stream.Measure()
        n = note.Note(currNote)
        n.quarterLength = currDuration / 4
        n.onset = currOnset
        m.append(n)
        p1.append(m)
        s.append(p1)
    return s
