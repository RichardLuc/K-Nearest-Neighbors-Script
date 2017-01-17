import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def KNN(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []

    for group in data:
        for features in data[group]:
            euclideanDistance = sqrt(sum(abs((np.array(features)-np.array(predict)))**2))
            distances.append([euclideanDistance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    voteResult = Counter(votes).most_common(1)[0][0]

    return voteResult

df = pd.read_csv('winequality-red.csv', sep=';')
fullData = df.astype(float).values.tolist()
random.shuffle(fullData)

trainSet = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
testSet = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}

trainData = fullData[:1000]
testData = fullData[len(fullData)-500:]

for i in trainData:
    trainSet[i[-1]].append(i[:-1])

for i in testData:
    testSet[i[-1]].append(i[:-1])

correct = 0
total = 0
for group in testSet:
    for data in testSet[group]:
        vote = KNN(trainSet, data, k=32)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/total)
