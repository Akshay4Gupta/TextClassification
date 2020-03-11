# imports
import pandas as pd
import math
import time
import re
import nltk
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
print('------------------------------------------Importing Ended------------------------------------------')

# -----------------------------------------------------------------------preprocessing----------------------------------------------------------------------------------
# data/testdata.manual.2009.06.14.csv
# data/training.1600000.processed.noemoticon.csv
trainingData = pd.read_csv("../data/training.1600000.processed.noemoticon.csv",
header = None,
names = ['polarity', 'id', 'date', 'query',
'username', 'tweet'],
encoding = "Latin-1")

testingData = pd.read_csv("../data/testdata.manual.2009.06.14.csv",
header = None,
names = ['polarity', 'id', 'date', 'query',
'username', 'tweet'],
encoding = "ISO-8859-1")

stop_words = set(stopwords.words('english')).union({',', '.', '!', ' ', '\n', '\t'})
stemming = PorterStemmer()
# tknzr = TweetTokenizer(strip_handles=True)
twittere = re.compile("^@[a-zA-Z0-9_]*$")
vectorizer = TfidfVectorizer()

# -------------------------------------------------------------------------functions-------------------------------------------------------------------------------------
def processingDocument(yEqualsK, words, dictionary, yEqualsKwc):
    wordcount = 0
    yEqualsK += 1
    for word in words:
        wordcount += 1
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1
    yEqualsKwc += wordcount
    return yEqualsK, dictionary, yEqualsKwc

def wordsExtraction(data, case):
    dataBuilt = dict()
    for j in data.itertuples():
        if(j[1] != 0 and j[1] != 4):
            continue
        words = list()
        if case == 'd':
            word_tokens_noTwitter = tokenize(j[6])

            # -------------------------------------------------------------------------------twitter tokenize------------------------------------------------------------
            # noTwitter_tokens = tknzr.tokenize(j[6])
            # word_tokens_noTwitter = []
            # for w in noTwitter_tokens:
            #     word_tokens_noTwitter += w.replace(',', ' ').replace('.', ' ').split()          #first part Q1(a)
            # -------------------------------------------------------------------------------twitter tokenize------------------------------------------------------------

            words = [stemming.stem(w) for w in word_tokens_noTwitter if not w in stop_words]
        elif case == 'a':
            words = j[6].replace(',', ' ').replace('.', ' ').split()
        dataBuilt[j[2]] = (j[1], words)
    return dataBuilt

def total_keys(dict1, dict2):
    dicto = {}
    count = 0
    for i, j in dict1.items():
        dicto[i] = j
        count += 1
    for i, j in dict2.items():
        if i in dicto:
            dicto[i] += j
        else:
            dicto[i] = j
            count += 1
    return (dicto, count)

def tokenize(word):
    wordlist = word.replace(',', ' ').replace('.', ' ').split()
    word_no_twitter = [w for w in wordlist if (not twittere.match(w))]
    return word_no_twitter

def accuracy(Data, philyEquals, phi, keysInDict, wcyEquals, confusionMatrix, case):
    GivenxyEqual = list()
    correct = 0
    wrong = 0
    LISTOFPROB0 = []
    LISTOFPROB1 = []

    for (id, (polarity, words)) in Data.items():
        gxye0 = phi[0]
        gxye1 = phi[1]
        for k in words:
            if k in philyEquals[0]:
                gxye0 += math.log(philyEquals[0][k])
            else:
                gxye0 += math.log(1/(keysInDict + wcyEquals[0]))
            if k in philyEquals[1]:
                gxye1 += math.log(philyEquals[1][k])
            else:
                gxye1 += math.log(1/(keysInDict + wcyEquals[1]))
        GivenxyEqual.append([math.log(phi[0]) + gxye0, math.log(phi[1]) + gxye1])
        if(GivenxyEqual[-1][0] > GivenxyEqual[-1][1]):
            if(polarity == 0):
                correct += 1
                confusionMatrix[0][0] += 1
            else:
                wrong += 1
                confusionMatrix[0][1] += 1

        if(GivenxyEqual[-1][0] < GivenxyEqual[-1][1]):
            if(polarity == 0):
                wrong += 1
                confusionMatrix[1][0] += 1
            else:
                correct += 1
                confusionMatrix[1][1] += 1
        LISTOFPROB0.append(gxye0)
        LISTOFPROB1.append(gxye1)
    return (correct, wrong, confusionMatrix, LISTOFPROB0, LISTOFPROB1)

def roc(LISTOFPROB0, LISTOFPROB1):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    LISTOFPROB = [LISTOFPROB0, LISTOFPROB1]
    data = np.array(testingData)[:,0].reshape(-1,1).astype(int)
    data0 = data[data!=2]
    for i in range(2):
    	fpr[i], tpr[i], _ = roc_curve(data0,LISTOFPROB[i], pos_label = 4)
    	roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

# --------------------------------------------------------------------------main code------------------------------------------------------------------------------------
def main():
    yEquals = [0, 0]                                                            # sum(1{y = 0})
    wcyEquals = [0, 0]                                                          # total no. of words in all the documents in class k
    dictionary = [dict(), dict()]                                               # dictionary of count of all the words
    keysInDict = [0, 0]
    start = time.time()
    case = sys.argv[1]
    trainingDataPreprocessed = wordsExtraction(trainingData, case)
    testingDataPreprocessed = wordsExtraction(testingData, case)
    for (id, (polarity, words)) in trainingDataPreprocessed.items():
        wordcount = 0
        if(polarity == 0):
            (yEquals[0], dictionary[0], wcyEquals[0]) = processingDocument(
                                yEquals[0], words, dictionary[0], wcyEquals[0])
        elif(polarity == 4):
            # -----------------------------------------------------------------------------without function calling-----------------------------------------------------
            # yEquals[1] += 1
            # for word in words:
            #     wordcount += 1
            #     if word in dictionary[1]:
            #         dictionary[1][word] += 1
            #     else:
            #         dictionary[1][word] = 1
            # wcyEquals[1] += wordcount
            # ----------------------------------------------------------------------------------------------------------------------------------
            (yEquals[1], dictionary[1], wcyEquals[1]) = processingDocument(
                                yEquals[1], words, dictionary[1], wcyEquals[1])

    m = yEquals[0] + yEquals[1]

    phi = [(yEquals[0]+1)/(m+2), (yEquals[1]+1)/(m+2)]

    (dicto, keysInDict) = total_keys(dictionary[0], dictionary[1])

    philyEquals = [{j: (i+1)/(wcyEquals[k]+keysInDict) for j, i in
                                    dictionary[k].items()} for k in range(2)]   #phi x = l and y = k

    confusionMatrix = [[0,0], [0,0]]

    print("Time:\t", time.time() - start, "\n----------------------------------Q1(a, c) Training Part-------------------------------------")
    (correct, wrong, confusionMatrix, LISTOFPROB0, LISTOFPROB1) = accuracy(trainingDataPreprocessed, philyEquals, phi, keysInDict, wcyEquals, confusionMatrix, case)
    print("accuracy over the training: ", (correct*100)/(correct + wrong), "\n")
    print("confusionMatrix:\t", "negative", "\t", "positive")
    print("negative(predicted)\t", confusionMatrix[0][0], "\t", confusionMatrix[0][1])
    print("positive(predicted)\t", confusionMatrix[1][0], "\t", confusionMatrix[1][1])

    print("\nTime:\t", time.time() - start, "\n----------------------------------Q1(a,c) Testing Part-------------------------------------")
    confusionMatrix = [[0,0], [0,0]]
    (correct, wrong, confusionMatrix, LISTOFPROB0, LISTOFPROB1) = accuracy(testingDataPreprocessed, philyEquals, phi, keysInDict, wcyEquals, confusionMatrix, case)
    roc(LISTOFPROB0, LISTOFPROB1)
    print("accuracy over the test: ", (correct*100)/(correct + wrong), "\n")
    print("confusionMatrix:\t", "negative", "\t", "positive")
    print("negative(predicted)\t", confusionMatrix[0][0], "\t\t", confusionMatrix[0][1])
    print("positive(predicted)\t", confusionMatrix[1][0], "\t\t", confusionMatrix[1][1])
    print("\nTime:\t", time.time() - start, "\n-------------------------------------------------------------------------------------------")

if __name__ == '__main__':
    main()
