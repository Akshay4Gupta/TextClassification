# imports
import pandas as pd
import math
import time

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



# data/testdata.manual.2009.06.14.csv
# data/training.1600000.processed.noemoticon.csv
# preprocessing
trainingData = pd.read_csv("../data/training.1600000.processed.noemoticon.csv",
                                header = None,
                                names = ['polarity', 'id', 'date', 'query',
                                                        'username', 'tweet'],
                                encoding = "Latin-1")

testingData = pd.read_csv("../data/training.1600000.processed.noemoticon.csv",
                                header = None,
                                names = ['polarity', 'id', 'date', 'query',
                                                        'username', 'tweet'],
                                encoding = "ISO-8859-1")

# main code
def main():
    yEquals = [0, 0]                                                            # sum(1{y = 0})
    wcyEquals = [0, 0]                                                          # total no. of words in all the documents in class k
    dictionary = [dict(), dict()]                                               # dictionary of count of all the words
    keysInDict = [0, 0]
    start = time.time()
    for j in trainingData.itertuples():
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
    print(keysInDict)
    philyEquals = [{j: (i+1)/(wcyEquals[k]+keysInDict) for j, i in
                                    dictionary[k].items()} for k in range(2)]   #phi x = l and y = k

    GivenxyEqual = list()
    correct = 0
    wrong = 0
    for j in testingData.itertuples():
        if j[1] == 0 or j[1] == 4 :
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
                else:
                    wrong += 1

            if(GivenxyEqual[-1][0] < GivenxyEqual[-1][1]):
                if(j[1] == 0):
                    wrong += 1
                else:
                    correct += 1

    print(time.time() - start)
    print((correct*100)/(correct + wrong))

if __name__ == '__main__':
    main()
