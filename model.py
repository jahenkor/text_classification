import pandas as pd
import operator
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def main():
   # np.set_printoptions(threshold=np.sys.maxsize)
    #Data Preprocessing/ Load Dataset()
    column_names = ['sentiment','doc']

    #delimiter using #EOF
    train_dataset = pd.read_csv("1548889051_0353532_train.dat", sep='#EOF', engine="python", header=None, names=column_names)
    train_dataset['sentiment'],train_dataset['doc'] = train_dataset['sentiment'].str.split('\t',1).str



    print(train_dataset)

    #Split training dataset into train/test, shuffle every time
    train, test = train_test_split(train_dataset, train_size = 0.50, test_size=0.0005,shuffle=False)
    print("len of training data: %d" % len(train))
    print("len of test data %d" % len(test))
    print(train)
    print(test)

    #Create dictionary/bow from training set
    docs_train = []
    docs_test = []
    sent_test = []
    sent_train = []
    for index, row in train.iterrows():
        docs_train.append(row['doc'])
        sent_train.append(row['sentiment'])
    for index, row in test.iterrows():
        docs_test.append(row['doc'])
        sent_test.append(row['sentiment'])
    

    vectorizer_train = CountVectorizer(stop_words='english')
    bow_train = vectorizer_train.fit_transform(docs_train).toarray()
    dictionary = vectorizer_train.get_feature_names()
    print("Bow Train\n")
    print(bow_train)
    print("Dictionary length: %d\n" % len(dictionary))

    #Use bow from training set as a mapping for bow for test set
    vectorizer_test = CountVectorizer(vocabulary=dictionary, stop_words='english')
    bow_test = vectorizer_test.fit_transform(docs_test).toarray()

    print("Bow Test\n")
    print(bow_test)
    print("Columns: %d\n" % len(vectorizer_test.get_feature_names()))

#Euclidean distance between two vectors
    #euclideanDistance(len(vectorizer_test.get_feature_names()),bow_train[0],bow_test[0])
    print("Test instance: \n",bow_test[0])

    print("nearest neighbors \n")

    predictions_arr = []
    for i in range(len(bow_test)):
        neighbors = calculateNeighbors(10, bow_train, bow_test[i])
        for x in neighbors:
            #print(x)
            predictions = prediction(3,neighbors, sent_train)
        if(predictions):
            print("Document prediction: +1")
            predictions_arr.append('+1')
        else:
            print("Document prediction: -1")
            predictions_arr.append('-1')

    accuracy_measure = accuracy (predictions_arr, sent_test)
    print("Accuracy: %f" % accuracy_measure) 



#Euclidean Distance between two vectors
def euclideanDistance(length, train, test):
    #dist = 0
    #for i in range(length):
     #   cos = cosine_similarity(train[i],test)
     #   dist = dist + pow(cos,2)
    #print("Distance: %d" % dist)
    return euclidean_distances(train,test.reshape(-1,1))

#calculate neighbors based on train set and a given test instance
def calculateNeighbors(k, train, test_inst):
    distances = []
    neighbors = []
    testLength = len(test_inst)
    trainLength = len(train)
    for x in range(trainLength):
        distance = euclideanDistance(testLength, train[x], test_inst)
        
        #Save train_inst, distance and indexFound in distances array
        distances.append((train[x],distance,x))
        #print(distances[x])
    distances.sort(key=operator.itemgetter(1))
    for i in range(k):
        #index of values found
        #print(distances[i][2])
        neighbors.append(distances[i][2])
        print("Closest k distances: %d" % distances[i][2])
    return neighbors

def prediction(k, neighbors, train):
    sentiment_arr = []
    x = 0
    pos_sentiment = 0
    neg_sentiment = 0
    for i in range(k):
        #for sentiment in train:
            #print(row['sentiment'])
        if train[neighbors[i]] == '+1':
         #   x+=1
                #continue
        #if sentiment == '+1':
                #sentiment_arr.append('+1')
            pos_sentiment += 1
        else:
                #sentiment_arr.append('-1')
            neg_sentiment += 1
        #x+=1
    #x = 0

    #print(sentiment_arr)
    #ternary conditional
    return (0,1)[pos_sentiment > neg_sentiment]

def accuracy(prediction, actual):

    #Calculate percentage of correct predictions
    correct = 0
    for x in range(len(prediction)):
        if prediction[x] == actual[x]:
            correct += 1
    return correct/len(actual)

            

        


main()
#i = 0
#pos_docs = numpy


#Debug
#print(pos_docs)
#print(len(pos_docs))

#vectorizer = TfidfVectorizer(stop_words='english', min_df=100)
#X = vectorizer.fit_transform(pos_docs)
#idf = vectorizer.idf_
#feature_names = vectorizer.get_feature_names()
#corpus_index = [n for n in pos_docs]
#dic = dict(zip(vectorizer.get_feature_names(),idf))
#print(dic)



#Test Dataset
#test_dataset = pd.read_csv("1548889052_1314285_test.dat", error_bad_lines=False, sep = "#EOF", engine="python", header=None)

#print(test_dataset)






