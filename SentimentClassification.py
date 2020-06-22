# @authors: Anusha Voloju , Yukthi Papanna Suresh
# CS 583, Spring 2019
# Project2: Sentiment Classification of Tweets

import warnings
import os
import numpy
import pathlib
import pandas as pd
import sys
import DataPreprocessing
import Classifiers


# parse the training excel file using pandas and return dataframe of tweets and class labels
def parseTrainingExcelFile(excelfile, sheet):
    trainsheet = pd.ExcelFile(excelfile)
    df = trainsheet.parse(sheet, encoding='ISO-8859-1', converters={'Anootated tweet': str, 'Class': str}, usecols= {3,4})
    df = df.rename(index=str, columns = {'Anootated tweet':'Anootated_tweet', 'Class':'Class'})
    indexNames = df[(df['Class'] == '-1') | (df['Class'] == '0') | (df['Class'] == '1')].index
    tweets = df.reindex(indexNames)
    tweets = tweets.reset_index(drop=True)
    return tweets


# parse the test csv files using pandas and return dataframe of tweet ids and tweet texts
def parseTestCSVFile(csvfile):
    df = pd.read_csv(csvfile, sep=",", encoding='ISO-8859-1', converters={'Tweet_ID': str, 'Tweet_text': str}, usecols= {'Tweet_ID', 'Tweet_text'})
    df = df.dropna(how = "all")
    tweets = df.rename(index=str, columns={'Tweet_text': 'Anootated_tweet'})
    return tweets


# update the predicted class labels of each tweet in the test set to an output text file
def updatePredictedClassInTextFile(tweet_Ids, predicted, testfilename):
    fname = pathlib.Path.cwd().joinpath('Output', testfilename+".txt")
    f = open(fname, "w+")
    print("Creating "+testfilename+".txt file..")
    for i in range(tweet_Ids.size):
        id = int(str(tweet_Ids[i]).replace("'","").replace("b",""))
        p = str(predicted[i]).replace("'","").replace("b","")
        f.write("%d;;%d \n" %((id),int(p)))
        #print("%d;;%d" %((id),int(p)))
    f.close()
    print(testfilename + ".txt file created successfully..\n")


# main function
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # check if there are two command line arguments
    if len(sys.argv) == 3:
        classifierName = sys.argv[1]
        PerformanceFlag = sys.argv[2]
    # check if there is one command line arguments
    elif len(sys.argv) == 2:
        classifierName = sys.argv[1]
        PerformanceFlag = "False"
    # if there are no command line arguments
    else:
        classifierName = "SVM"
        PerformanceFlag = "False"

    # get the excel file from the Train folder
    excelfiles = pathlib.Path.cwd().joinpath('Train')
    excelfileslist = list()
    for currentFile in excelfiles.iterdir():
        excelfileslist.append(currentFile)
    trainexcelfile = excelfileslist[0]

    # pre process Obama tweets
    Obama_tweets = parseTrainingExcelFile(trainexcelfile, 'Obama')
    Obama_tweets = DataPreprocessing.preprocessTweetText(Obama_tweets)

    X_Obama = Obama_tweets.Anootated_tweet
    y_Obama = numpy.asarray(Obama_tweets.Class, dtype="|S6")

    # pre process Romney tweets
    Romney_tweets = parseTrainingExcelFile(trainexcelfile, 'Romney')
    Romney_tweets = DataPreprocessing.preprocessTweetText(Romney_tweets)

    X_Romney = Romney_tweets.Anootated_tweet
    y_Romney = numpy.asarray(Romney_tweets.Class, dtype="|S6")

    # initialize classifier variables
    Obamaclassifier = None
    Obamatweet_tfidfvec = None
    Romneyclassifier = None
    Romneytweet_tfidfvec = None

    # if the command line argument for classifier name is given as MNB
    if(classifierName == "MNB"):
        print("Running Multinomial Naive Bayes")
        Obamaclassifier, Obamatweet_tfidfvec = Classifiers.train_NaiveBayes_Classifier(X_Obama, y_Obama)
        Romneyclassifier, Romneytweet_tfidfvec = Classifiers.train_NaiveBayes_Classifier(X_Romney, y_Romney)


    # if the command line argument for classifier name is given as SVM
    elif (classifierName == "SVM"):
        print("Running SVM")
        Obamaclassifier, Obamatweet_tfidfvec = Classifiers.train_SVM_Classifier(X_Obama, y_Obama)
        Romneyclassifier, Romneytweet_tfidfvec = Classifiers.train_SVM_Classifier(X_Romney, y_Romney)


    # if the command line argument for classifier name is given as LR
    elif (classifierName == "LR"):
        print("Running Logistic Regression")
        Obamaclassifier, Obamatweet_tfidfvec = Classifiers.train_LogisticRegression_Classifier(X_Obama, y_Obama)
        Romneyclassifier, Romneytweet_tfidfvec = Classifiers.train_LogisticRegression_Classifier(X_Romney, y_Romney)


    # if the command line argument for classifier name is given as DT
    elif (classifierName == "DT"):
        print("Running Decision Tree")
        Obamaclassifier, Obamatweet_tfidfvec = Classifiers.train_DecisionTree_Classifier(X_Obama, y_Obama)
        Romneyclassifier, Romneytweet_tfidfvec = Classifiers.train_DecisionTree_Classifier(X_Romney, y_Romney)


    # if the command line argument for classifier name is given as KNN
    elif (classifierName == "KNN"):
        print("Running K-Nearest Neighbour")
        Obamaclassifier, Obamatweet_tfidfvec = Classifiers.train_KNN_Classifier(X_Obama, y_Obama)
        Romneyclassifier, Romneytweet_tfidfvec = Classifiers.train_KNN_Classifier(X_Romney, y_Romney)


    # if the command line argument for classifier name is given as RF
    elif (classifierName == "RF"):
        print("Running Random Forest")
        Obamaclassifier, Obamatweet_tfidfvec = Classifiers.train_RandomForest_Classifier(X_Obama, y_Obama)
        Romneyclassifier, Romneytweet_tfidfvec = Classifiers.train_RandomForest_Classifier(X_Romney, y_Romney)


    # if the command line argument for classifier name is given as AB
    elif (classifierName == "AB"):
        print("Running Ada Boost")
        Obamaclassifier, Obamatweet_tfidfvec = Classifiers.train_AdaBoost_Classifier(X_Obama, y_Obama)
        Romneyclassifier, Romneytweet_tfidfvec = Classifiers.train_AdaBoost_Classifier(X_Romney, y_Romney)


    # if the command line argument for classifier name is not given
    else:
        Obamaclassifier, Obamatweet_tfidfvec = Classifiers.train_SVM_Classifier(X_Obama, y_Obama)
        Romneyclassifier, Romneytweet_tfidfvec = Classifiers.train_SVM_Classifier(X_Romney, y_Romney)


    # if the command line argument for Performance Flag is given as True
    if (PerformanceFlag == "True"):
        Classifiers.performanceOf_Classifier(X_Obama, y_Obama, 'Obama', classifierName)
        Classifiers.performanceOf_Classifier(X_Romney, y_Romney, 'Romney', classifierName)


    # get the csv file from the Train folder
    csvfiles = pathlib.Path.cwd().joinpath('Test')
    csvfileslist = list()
    for currentFile in csvfiles.iterdir():
        csvfileslist.append(currentFile)

    # pre process the test tweets and use classifier to predict the class labels
    if len(csvfileslist) > 0:

        for file in csvfileslist:

            test_tweets = parseTestCSVFile(file)
            test_tweets = DataPreprocessing.preprocessTweetText(test_tweets)
            X_test = test_tweets.Anootated_tweet
            tweet_Ids = numpy.asarray(test_tweets.Tweet_ID, dtype="|S6")

            if ("Obama" in os.path.basename(file)):
                predicted = Classifiers.testClassifier(Obamaclassifier, Obamatweet_tfidfvec, X_test)
                updatePredictedClassInTextFile(tweet_Ids, predicted, os.path.basename(file).split('.')[0])

            if ("Romney" in os.path.basename(file)):
                predicted = Classifiers.testClassifier(Romneyclassifier, Romneytweet_tfidfvec, X_test)
                updatePredictedClassInTextFile(tweet_Ids, predicted, os.path.basename(file).split('.')[0])
    else:
        print("There are no test files to predict the class labels..")






