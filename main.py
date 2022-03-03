import numpy as np
import matplotlib.pyplot as plt
MINIMAL = 0.0001
# loading the datasets split into ~2300 entries each
trainingData = np.loadtxt("data/spambase-training.csv", delimiter=",")
testData = np.loadtxt("data/spambase-test.csv", delimiter=",")

totalRight = 0
# counting the spam in training data
numSpamTraining = 0
numTotalTraining = 0
for x in trainingData:
    if x[-1] == 1:
        numSpamTraining += 1
    numTotalTraining += 1

# counting the spam in testing data
numSpamTest = 0
numTotalTest = 0
for x in testData:
    if x[-1] == 1:
        numSpamTest += 1
    numTotalTest += 1

# Find the mean for each of the 57 features given each class:

spamData = []
nonSpamData = []

# create 2 arrays of spam data and nonspam data
for x in trainingData:
    if x[-1] == 1:
        spamData.append(x)
    else:
        nonSpamData.append(x)

# mean = sum of all data points / number of data points
meanSpamAttributes = np.sum(spamData, axis=0)/numSpamTraining
meanNonSpamAttributes = np.sum(nonSpamData, axis=0)/(numTotalTraining - numSpamTraining)

# calc square of distance to the mean
stdSpamData = spamData - meanSpamAttributes
stdNonSpamData = nonSpamData - meanNonSpamAttributes
stdSpamData = np.square(stdSpamData)
stdNonSpamData = np.square(stdNonSpamData)

# sum together squares of distance
stdSpam = np.sum(stdSpamData, axis=0)
stdNonSpam = np.sum(stdNonSpamData, axis=0)

# divide sum of squares by number of data points for each set
stdSpam = stdSpam / numSpamTraining
stdNonSpam = stdNonSpam / (numTotalTraining-numSpamTraining)

# square root of the resulting division from previous step is the standard deviations
stdSpam = np.sqrt(stdSpam)
stdNonSpam = np.sqrt(stdNonSpam)

# set std dev values to minimum if they are 0.0 to avoid div by 0:
index = 0
for x in stdSpam:
    if x == 0.0:
        stdSpam[index] = 0.0001
    index += 1
index = 0
for x in stdNonSpam:
    if x == 0.0:
        stdNonSpam[index] = 0.0001
    index += 1


def argmax(arr):
    falsePositive = 0
    falseNegative = 0
    truePositive = 0
    trueNegative = 0
    tot = 0
    arr1 = []
    arr2 = []
    pSpam = np.log(numSpamTraining/numTotalTraining)
    pNSpam = np.log(1-(numSpamTraining/numTotalTraining))

    for emails in arr:
        inde = 0
        # this is the main equation used for gaussian naive bayes
        # 'vals' is each of the attributes of the test data arrays
        # using logarithms to avoid underflow
        for vals in emails[:-1]:
            arr1.append(np.log((1/(np.sqrt(2*np.pi)*stdSpam[inde]))*np.exp(-(vals-meanSpamAttributes[inde]) / (2 * np.square(stdSpam[inde])))))
            arr2.append(np.log((1/(np.sqrt(2*np.pi)*stdNonSpam[inde]))*np.exp(-(vals-meanNonSpamAttributes[inde]) / (2 * np.square(stdNonSpam[inde])))))
            inde += 1
        # adding the probability of spam or not spam with the sum of the respective arrays
        arg1 = pSpam + sum(arr1)
        arg2 = pNSpam + sum(arr2)

        # if arg1 > arg2, that means that we've classified the data point as spam
        # check to see if its actually spam, and if it is, increment the counter for true positive
        # else increment false positive
        if arg1 > arg2:
            if emails[-1] == 1:
                truePositive += 1
            else:
                falsePositive += 1
        # if arg1 < arg2, that means that we've classified the data point as not spam
        # check to see if its actually not spam, and if it is, increment the counter for true negative
        # else increment false negative
        if arg1 < arg2:
            if emails[-1] == 0:
                trueNegative += 1
            else:
                falseNegative += 1
        tot += 1
        arr1.clear()
        arr2.clear()
    print("Total right: ", truePositive + trueNegative)
    print("True Positive: ", truePositive)
    print("True Negative: ", trueNegative)
    print("False Positive: ", falsePositive)
    print("False Negative: ", falseNegative)
    print("total: ", tot)


print("Total Spam in Training Set: ", numSpamTraining)
print("Total in Training Set: ", numTotalTraining)
print("Probability of spam in training: ", numSpamTraining/numTotalTraining)

print("Total Spam in Test Set: ", numSpamTest)
print("Total in Test Set: ", numTotalTest)
print("Probability of spam in test: ", numSpamTest/numTotalTest)

# print("Mean values for spam: ", meanSpamAttributes)
# print("Mean values for non-spam: ", meanNonSpamAttributes)

# print("STD DEV for spam: ", stdSpam)
# print("STD DEV for nonspam: ", stdNonSpam)

argmax(testData)

