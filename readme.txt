
Sentiment Classification of Tweets

Authors:

Anusha Voloju
UIN: 677775723
NetId: avoloj2

Yukthi Papanna Suresh
UIN: 675253362
NetId: ypapan2


The code for the Sentiment Classification of Tweets is implemented in python3.
Please use the appropriate python interpreter to execute the program.


Required packages :

1. os
2. numpy
3. pathlib
4. pandas
5. sys
6. re
7. nltk
8. emoji
9. sklearn
10. warnings


Setup :

Extract the contents of the compressed file.
There will be three python files - SentimentClassification.py, Classifiers.py, DataPreprocessing.py,
and three folders - Train, Test and Output folder.

SentimentClassification.py - main file which contains the implementation of parsing excel and csv files.
preprocessing.py - implementation of preprocessing the training and test tweets.
classifiers.py - implementation of training of classifiers.

Train folder - contains trainingObamaRomneytweets.xlsx file which is used to train the classifier.
Test folder - contains the test csv files with the tweets for which the classes are to be predicted.
Output folder - output text files(with predicted classes), created from the test files.


Steps to execute :

1. Make sure that the python files, Train, Test and Output are in same directory.

2. Place the training file (in excel format with Obama and Romney sheets both with 'Anootated tweet','Class' columns) in Train folder.

3. Place the test csv files (with 'Tweet_ID', 'Tweet_Text' columns) in the Test folder, with the file names having Obama and Romney keywords, to identify and differentiate them.

4. Execute the SentimentClassification.py file. By default it uses SVM classifier and creates test files.It does not show any evaluation results.
> python SentimentClassification.py

5. If you want to train a different classifier, execute the SentimentClassification.py file by passing classifier name as command line argument.
> python SentimentClassification.py <classifier name (MNB, SVM, LR, DT, KNN, RF, AB)>
> python SentimentClassification.py SVM
Note: classifier name which you would pass as the command line argument, should be one among the list (NB, SVM, LR, DT, KNN, RF, AB) where
NB-NaiveBayes, SVM-SupportVectorMachines, LR-LogisticRegression, DT-DecisionTree, KNN-KNearestNeighbours, RF-RandomForest, AB-AdaBoost.

6. If you want to know the performance of the classifier on training set (by 10-fold cross validation), execute the
SentimentClassification.py file by passing classifier name as first command line argument and True as first command line argument.
The second argument is a Performance Flag to show the performance of the classifer.
> python SentimentClassification.py <classifier name (MNB, SVM, LR, DT, KNN, RF, AB)> <PerformanceFlag>
> python SentimentClassification.py SVM True
Note: classifier name which you would pass as the command line argument, should be one among the list (NB, SVM, LR, DT, KNN, RF, AB)
The second argument should be True or False.

7. After execution, two text files will be generated in Output folder with the content in below format.
"Tweet_ID;;Predicted Class Label\n"
