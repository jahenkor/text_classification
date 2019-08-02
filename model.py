import pandas as pd
import operator
import numpy as np
from numpy import array
import random
from numpy import dot
from numpy.linalg import norm
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt

error_rate = []
recall_rate = []
precision_rate = []
#KNN - lazy classifer
def main():


    dist_metrics = ['manhattan', 'euclidean', 'jaccard','cosine']
    train_size = 0.07
    test_size = 0.01
    k_val_list = list(range(1,45))

    #Select odd k values for training
    k_val_listOdds = [i for i in k_val_list if i % 2]
    accuracy_measures = {}
    recall_measures = {}
    precision_measures = {}
    
    #Iterate and test distance metrics
    for i in dist_metrics:
        accuracy_measure_list = []
        apply_model = False

#Put this statement, outside for loop
        if(apply_model):
             k_val = 25
             compute(k_val, apply_model, train_size, test_size, "manhattan")
             exit()
        else:
            for k in k_val_listOdds:
                accuracy_measure_list.append(compute(k, apply_model, train_size, test_size, i))
                
            accuracy_measures[i] = accuracy_measure_list.copy()
            precision_measures[i] = precision_rate.copy()
            recall_measures[i] = recall_rate.copy()
            
            recall_rate.clear()
            precision_rate.clear()


    #Plots
    #Accuracy over K values
    plt.subplot(2,2,1)
    plt.plot(k_val_listOdds, accuracy_measures['manhattan'],'g')
    plt.plot(k_val_listOdds, accuracy_measures['euclidean'],'b')
    plt.plot(k_val_listOdds, accuracy_measures['jaccard'], 'r')
    plt.plot(k_val_listOdds, accuracy_measures['cosine'],'y')
    plt.xlabel('Number of neighbors k')
    plt.ylabel('Accuracy')

    #Error rate over K values
#    plt.subplot(2,2,2)
#    plt.plot(k_val_listOdds, error_rate)
#    plt.xlabel('Number of neighbors k')
#    plt.ylabel('Error Rate')
        
    #Recall
    plt.subplot(2,2,3)
    plt.plot(k_val_listOdds, recall_measures['manhattan'],'g')
    plt.plot(k_val_listOdds, recall_measures['euclidean'],'b')
    plt.plot(k_val_listOdds, recall_measures['jaccard'], 'r')
    plt.plot(k_val_listOdds, accuracy_measures['cosine'],'y')
    plt.xlabel('Number of neighbors K')
    plt.ylabel('Recall rate')
    
    #Precision
    plt.subplot(2,2,4)
    plt.plot(k_val_listOdds, precision_measures['manhattan'],'g')
    plt.plot(k_val_listOdds, precision_measures['euclidean'],'b')
    plt.plot(k_val_listOdds, precision_measures['jaccard'], 'r')
    plt.plot(k_val_listOdds, accuracy_measures['cosine'],'y')
    plt.xlabel('Number of neighbors K')
    plt.ylabel('Precision rate')

    plt.show()

       
        

def compute(k_val, applyModel, train_size, test_size, dist_metric):

    applyModel = applyModel
    k_val = k_val
    column_names = ['sentiment','doc']

    #Load Datasets
    train_dataset = pd.read_csv("1548889051_0353532_train.dat", sep='#EOF', engine="python", header=None, names=column_names)
    train_dataset['sentiment'],train_dataset['doc'] = train_dataset['sentiment'].str.split('\t',1).str
    test_dataset = 0



    #Splitting train/test set
    train, test = train_test_split(train_dataset, train_size = train_size, test_size=test_size,shuffle=False)

    #Split document/sentiment
    docs_train = []
    docs_test = []
    sent_test = []
    sent_train = []
    for index, row in train.iterrows():
        docs_train.append(row['doc'])
        sent_train.append(row['sentiment'])
    if(not applyModel):
        for index, row in test.iterrows():
            docs_test.append(row['doc'])
            sent_test.append(row['sentiment'])

    if(dist_metric == 'euclidean'):
        vectorizer_train = TfidfVectorizer(stop_words='english', max_df=0.65, ngram_range=(1,3), norm='l2', max_features = 25000)
    elif(dist_metric == 'jaccard'):
        vectorizer_train = CountVectorizer(stop_words='english', max_df=0.65, ngram_range=(1,3), binary=True, max_features = 25000)

    elif(dist_metric == 'manhattan'):
        vectorizer_train = TfidfVectorizer(stop_words='english', max_df=0.65, norm='l1')

    else:
        vectorizer_train = TfidfVectorizer(stop_words='english', norm='l2', max_df = 0.65)




        




#    lsa = TruncatedSVD(n_components = 3000)
    #docs_train = lsa.fit_transform(array(docs_train).reshape(1,-1))
    bow_train = vectorizer_train.fit_transform(docs_train).toarray() 
#    bow_train = lsa.fit_transform(bow_train)



   #Applying model
    bow_test = 0
    if(applyModel):
        bow_test = loadTest(vectorizer_train)
    else:
        bow_test = vectorizer_train.transform(docs_test).toarray()
#        bow_test = lsa.transform(bow_test)
    #bow_test = lsa.transform(bow_test)

    predictions_arr = []
    for i in range(len(bow_test)):
        neighbors = calculateNeighbors(k_val, bow_train, bow_test[i], dist_metric)
        predictions = prediction(k_val,neighbors, sent_train)
        if(predictions):
            predictions_arr.append('+1')
        else:
            predictions_arr.append('-1')

    #Print training accuracy
    if(not applyModel):
        accuracy_measure = accuracy (predictions_arr, sent_test)
        print("Accuracy: %f" % accuracy_measure) 


    #Print to file
    if(applyModel):
        return outTestSet(predictions_arr)
    
    return accuracy_measure

 
def outTestSet(predictions_arr):
    with open('test.dat','w') as the_file:
        for i in range(len(predictions_arr)):
            the_file.write("%s\n" % predictions_arr[i])
        the_file.flush()



def loadTest(vectorizer_train):

    test_dataset = pd.read_csv("1548889052_1314285_test.dat", sep="#EOF", engine="python", header=None)
    test_dataset.columns=["doc","garbage"]
    del test_dataset['garbage']
    bow_test = vectorizer_train.transform(test_dataset['doc'].values).toarray()


    return bow_test





#Euclidean Distance between two vectors
def distanceMetric(length, train, test, dist_metric):

    if(dist_metric == 'cosine'):
        return 1-(dot(train, test)/(norm(train)*norm(test)))


    distanceType = DistanceMetric.get_metric(dist_metric)
    distance = distanceType.pairwise(train.reshape(1,-1),test.reshape(1,-1))
    
    #Jaccard Distance
    if(dist_metric == 'jaccard'):
        return distance
    
    
    return distance

#calculate neighbors based on train set and a given test instance
def calculateNeighbors(k, train, test_inst, dist_metric):
    distances = []
    neighbors = []
    testLength = len(test_inst)
    trainLength = len(train)

    for x in range(trainLength):
        
        distance = distanceMetric(testLength, train[x], test_inst,dist_metric)

        #Save distance and indexFound in distances array
        distances.append((distance,x))

    
    distances.sort(key=operator.itemgetter(0))

    #Find closest k neighbors
    for i in range(k):
        neighbors.append(distances[i][1])
    return neighbors

def prediction(k, neighbors, train):
    sentiment_arr = []
    x = 0
    pos_sentiment = 0
    neg_sentiment = 0
    for i in range(k):
        if train[neighbors[i]] == '+1':
            pos_sentiment += 1
        else:
            neg_sentiment += 1

    return (0,1)[pos_sentiment > neg_sentiment]

def accuracy(prediction, actual):

    
    #Calculate percentage of correct predictions
    correct = 0
    false_neg = 0
    false_pos = 0
    true_pos = 0
    true_neg = 0
    total_incorrect = 0
    for x in range(len(prediction)):
        if prediction[x] == actual[x]:
            if(prediction[x] == '+1'):
                true_pos += 1
            if(prediction[x] == '-1'):
                true_neg += 1
            correct += 1
 
        else:
            if(prediction[x] == '+1'):
                false_pos += 1
            if(prediction[x] == '-1'):
                false_neg += 1
            total_incorrect += 1

    error_rate.append(total_incorrect/len(actual))
    if(true_pos == 0):
        recall_rate.append(0)
    else:
        recall_rate.append(true_pos / (true_pos + false_neg))
    if(true_pos == 0):
        precision_rate.append(0)
    else:
        precision_rate.append(true_pos / (true_pos + false_pos))
    return correct/len(actual)

            

        


main()








