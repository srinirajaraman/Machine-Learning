'''
This python code is to test the naive-bayes-classifier-scratch-python
http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
'''

import csv
import os as os  
import math
import random

#Compute mean
def computeMean(numbers):
    return sum(numbers)/float(len(numbers))


#Compute standard dev	
def computeStdDev(numbers):
    avg = computeMean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
     
#Load the dataset given the file
def loadDataset(fileName):
    lines = csv.reader(open(fileName, 'rb'))
    dataset = list(lines)
    lengthData = len(dataset)
    for i in range(lengthData):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def seperateByClass(dataset):
    seperated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if(vector[-1] not in seperated):
            seperated[vector[-1]] = []
        seperated[vector[-1]].append(vector)
    return seperated

def summarize(dataset):
    summaries = [(computeMean(attribute), computeStdDev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    seperated = seperateByClass(dataset)
    summaries = {}
    for classval, instances in seperated.iteritems():
        summaries[classval] = summarize(instances)
    return summaries


def calcGaussianProb(x, mean, stdDev):
    exponent = math.exp(- (math.pow(x - mean, 2)/(2 * math.pow(stdDev, 2))))
    return 1/(math.sqrt(2 * math.pi) * stdDev) * exponent

#Compute class probabilities for each of the summaries
def calculateClassProb(summaries, inputVec):
    prob = {}
    for classVal, classSumm in summaries.iteritems(): 
        prob[classVal] = 1
        for i in range(len(classSumm)):
            mean, stddev = classSumm[i]
            x = inputVec[i]
            prob[classVal] *= calcGaussianProb(x, mean, stddev)
    return prob
	
def predict(summaries, inputVec):   
    probs = calculateClassProb(summaries, inputVec)
    bestLabel  = None
    bestprob = -1
    for classValue, probability in probs.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
    return bestLabel

# Predictive computation
def makePredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        res = predict(summaries, testSet[i])
        predictions.append(res)
    return predictions

#Accuracy computation
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

	
	# Split the dataset to get training 
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

if __name__ == '__main__':
    currDir = os.getcwd()
    newpath = currDir + '\Log' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    dataset_loc = '/Datasets/'
    dataset_name = 'Sampledata.csv'
    data_file = currDir + dataset_loc + dataset_name
    dataset = loadDataset(data_file)
    splitRatio = 0.67
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
    summaries = summarizeByClass(trainingSet)
	# test model
    predictions = makePredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%').format(accuracy)
 