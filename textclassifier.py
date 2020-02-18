# imports
import pandas as pd
import math
import time
import nltk
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer

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
tknzr = TweetTokenizer(strip_handles=True)

# functions
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

def accuracy(Data, philyEquals, phi, keysInDict, wcyEquals, confusionMatrix, case):
    GivenxyEqual = list()
    correct = 0
    wrong = 0
    for j in Data.itertuples():
        if j[1] == 0 or j[1] == 4 :
            words = list()
            if case == 'd':
                noTwitter_tokens = tknzr.tokenize(j[6])
                word_tokens_noTwitter = []
                for w in noTwitter_tokens:
                    word_tokens_noTwitter += w.replace(',', ' ').replace('.', ' ').split()          #first part Q1(a)
                words = [stemming.stem(w) for w in word_tokens_noTwitter if not w in stop_words]
            elif case == 'a':
                words = j[6].replace(',', ' ').replace('.', ' ').split()

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
                if(j[1] == 0):
                    correct += 1
                    confusionMatrix[0][0] += 1
                else:
                    wrong += 1
                    confusionMatrix[0][1] += 1

            if(GivenxyEqual[-1][0] < GivenxyEqual[-1][1]):
                if(j[1] == 0):
                    wrong += 1
                    confusionMatrix[1][0] += 1
                else:
                    correct += 1
                    confusionMatrix[1][1] += 1
    return (correct, wrong, confusionMatrix)

# preprocessing

# main code
def main():
    yEquals = [0, 0]                                                            # sum(1{y = 0})
    wcyEquals = [0, 0]                                                          # total no. of words in all the documents in class k
    dictionary = [dict(), dict()]                                               # dictionary of count of all the words
    keysInDict = [0, 0]
    start = time.time()
    case = sys.argv[1]

    for j in trainingData.itertuples():
        words = list()
        if case == 'd':
            noTwitter_tokens = tknzr.tokenize(j[6])
            word_tokens_noTwitter = []
            for w in noTwitter_tokens:
                word_tokens_noTwitter += w.replace(',', ' ').replace('.', ' ').split()          #first part Q1(a)
            words = [stemming.stem(w) for w in word_tokens_noTwitter if not w in stop_words]
        elif case == 'a':
            words = j[6].replace(',', ' ').replace('.', ' ').split()

        wordcount = 0
        if(j[1] == 0):
            (yEquals[0], dictionary[0], wcyEquals[0]) = processingDocument(
                                yEquals[0], words, dictionary[0], wcyEquals[0])
        elif(j[1] == 4):
            # without function calling
            # yEquals[1] += 1
            # for word in words:
            #     wordcount += 1
            #     if word in dictionary[1]:
            #         dictionary[1][word] += 1
            #     else:
            #         dictionary[1][word] = 1
            # wcyEquals[1] += wordcount
            (yEquals[1], dictionary[1], wcyEquals[1]) = processingDocument(
                                yEquals[1], words, dictionary[1], wcyEquals[1])

    m = yEquals[0] + yEquals[1]

    phi = [(yEquals[0]+1)/(m+2), (yEquals[1]+1)/(m+2)]

    (dicto, keysInDict) = total_keys(dictionary[0], dictionary[1])

    philyEquals = [{j: (i+1)/(wcyEquals[k]+keysInDict) for j, i in
                                    dictionary[k].items()} for k in range(2)]   #phi x = l and y = k

    confusionMatrix = [[0,0], [0,0]]

    print("Time:\t", time.time() - start, "\n----------------------------------Q1(a)-------------------------------------")
    (correct, wrong, confusionMatrix) = accuracy(trainingData, philyEquals, phi, keysInDict, wcyEquals, confusionMatrix, case)
    print("accuracy over the training: ", (correct*100)/(correct + wrong), "\n")
    print("confusionMatrix:\t", "negative", "\t", "positive")
    print("negative(predicted)\t", confusionMatrix[0][0], "\t", confusionMatrix[0][1])
    print("positive(predicted)\t", confusionMatrix[1][0], "\t", confusionMatrix[1][1])

    print("\nTime:\t", time.time() - start, "\n----------------------------------Q2(a)-------------------------------------")
    confusionMatrix = [[0,0], [0,0]]
    (correct, wrong, confusionMatrix) = accuracy(testingData, philyEquals, phi, keysInDict, wcyEquals, confusionMatrix, case)
    print("accuracy over the test: ", (correct*100)/(correct + wrong), "\n")
    print("confusionMatrix:\t", "negative", "\t", "positive")
    print("negative(predicted)\t", confusionMatrix[0][0], "\t\t", confusionMatrix[0][1])
    print("positive(predicted)\t", confusionMatrix[1][0], "\t\t", confusionMatrix[1][1])
    print("\nTime:\t", time.time() - start, "\n----------------------------------")

if __name__ == '__main__':
    main()
