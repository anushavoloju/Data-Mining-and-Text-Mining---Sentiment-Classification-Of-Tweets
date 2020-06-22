# @authors: Anusha Voloju , Yukthi Papanna Suresh
# CS 583, Spring 2019
# Project2: Sentiment Classification of Tweets

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics


# train Naive Bayes Classifier
def train_NaiveBayes_Classifier(X, y):

    X_train = X
    y_train = y

    tweet_tfidfvec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))
    tweet_tfidfvec.fit(X_train.astype('U'))
    X_train_tfidfvec = tweet_tfidfvec.transform(X_train.astype('U'))

    classifier = MultinomialNB(alpha=1.0, fit_prior=False)
    classifier.fit(X_train_tfidfvec, y_train)

    return classifier, tweet_tfidfvec


# train SVM Classifier
def train_SVM_Classifier(X, y):

    X_train = X
    y_train = y

    tweet_tfidfvec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))
    tweet_tfidfvec.fit(X_train.astype('U'))
    X_train_tfidfvec = tweet_tfidfvec.transform(X_train.astype('U'))

    classifier = svm.SVC(C=1.0, kernel='linear')
    classifier.fit(X_train_tfidfvec, y_train)

    return classifier, tweet_tfidfvec


# train Logistic Regression Classifier
def train_LogisticRegression_Classifier(X, y):

    X_train = X
    y_train = y

    tweet_tfidfvec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3))
    tweet_tfidfvec.fit(X_train.astype('U'))
    X_train_tfidfvec = tweet_tfidfvec.transform(X_train.astype('U'))

    classifier = LogisticRegression()
    classifier.fit(X_train_tfidfvec, y_train)

    return classifier, tweet_tfidfvec


# train Decision Tree Classifier
def train_DecisionTree_Classifier(X, y):

    X_train = X
    y_train = y

    tweet_countvector = CountVectorizer(stop_words='english')
    X_train_twmatrix = tweet_countvector.fit_transform(X_train.astype('U'))

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X_train_twmatrix, y_train)

    return classifier, tweet_countvector


# train K-Nearest Neighbours Classifier
def train_KNN_Classifier(X, y):

    X_train = X
    y_train = y

    tweet_tfidfvec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3))
    tweet_tfidfvec.fit(X_train)
    X_train_tfidfvec = tweet_tfidfvec.transform(X_train.astype('U'))

    classifier = KNeighborsClassifier(n_neighbors=10)
    classifier.fit(X_train_tfidfvec, y_train)

    return classifier, tweet_tfidfvec


# train Random Forest Classifier
def train_RandomForest_Classifier(X, y):

    X_train = X
    y_train = y

    tweet_tfidfvec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3))
    tweet_tfidfvec.fit(X_train)
    X_train_tfidfvec = tweet_tfidfvec.transform(X_train.astype('U'))

    classifier = RandomForestClassifier()
    classifier.fit(X_train_tfidfvec, y_train)

    return classifier, tweet_tfidfvec


# train Ada Boost Classifier
def train_AdaBoost_Classifier(X, y):

    X_train = X
    y_train = y

    tweet_tfidfvec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3))
    tweet_tfidfvec.fit(X_train)
    X_train_tfidfvec = tweet_tfidfvec.transform(X_train.astype('U'))

    classifier = AdaBoostClassifier(base_estimator=None, n_estimators=10)
    classifier.fit(X_train_tfidfvec, y_train)

    return classifier, tweet_tfidfvec


# get performance of the Classifier by 10-fold cross validation
def performanceOf_Classifier(X, y, sheet, classifierName):
    tenfold = KFold(n_splits=10)
    sum_p_score_neg = 0
    sum_p_score_pos = 0
    sum_p_score_neu = 0
    sum_r_score_neg = 0
    sum_r_score_pos = 0
    sum_r_score_neu = 0
    sum_f_score_neg = 0
    sum_f_score_pos = 0
    sum_f_score_neu = 0
    Sum_accuracy = 0

    print()
    print(sheet +" tweets")

    for train_index, test_index in tenfold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier = None
        tweet_vec = None

        if (classifierName == "MNB"):
            classifier, tweet_vec = train_NaiveBayes_Classifier(X_train, y_train)

        elif (classifierName == "SVM"):
            classifier, tweet_vec = train_SVM_Classifier(X_train, y_train)

            # if the command line argument for classifier name is given as LR
        elif (classifierName == "LR"):
            classifier, tweet_vec = train_LogisticRegression_Classifier(X_train, y_train)

            # if the command line argument for classifier name is given as DT
        elif (classifierName == "DT"):
            classifier, tweet_vec = train_DecisionTree_Classifier(X_train, y_train)

            # if the command line argument for classifier name is given as KNN
        elif (classifierName == "KNN"):
            classifier, tweet_vec = train_KNN_Classifier(X_train, y_train)

            # if the command line argument for classifier name is given as RF
        elif (classifierName == "RF"):
            classifier, tweet_vec = train_RandomForest_Classifier(X_train, y_train)

            # if the command line argument for classifier name is given as AB
        elif (classifierName == "AB"):
            classifier, tweet_vec = train_AdaBoost_Classifier(X_train, y_train)


        X_test_vec = tweet_vec.transform(X_test.astype('U'))

        y_predict_class = classifier.predict(X_test_vec)

        p_score = metrics.precision_score(y_test, y_predict_class, average=None)
        sum_p_score_neg = sum_p_score_neg + p_score[0]
        sum_p_score_neu = sum_p_score_neu + p_score[1]
        sum_p_score_pos = sum_p_score_pos + p_score[2]

        r_score = metrics.recall_score(y_test, y_predict_class, average=None)
        sum_r_score_neg = sum_r_score_neg + r_score[0]
        sum_r_score_neu = sum_r_score_neu + r_score[1]
        sum_r_score_pos = sum_r_score_pos + r_score[2]

        f_score = metrics.f1_score(y_test, y_predict_class, average=None)
        sum_f_score_neg = sum_f_score_neg + f_score[0]
        sum_f_score_neu = sum_f_score_neu + f_score[1]
        sum_f_score_pos = sum_f_score_pos + f_score[2]

        accuracy = metrics.accuracy_score(y_test, y_predict_class)
        # print("Accuracy", accuracy * 100)

        Sum_accuracy = Sum_accuracy + (accuracy * 100)

    print()
    print("Average Precision Positive: ", sum_p_score_pos/10)
    print("Average Precision Negative: ", sum_p_score_neg/10)
    print("Average Precision Neutral: ", sum_p_score_neu/10)
    print()
    print("Average Recall Positive: ", sum_r_score_pos/10)
    print("Average Recall Negative: ", sum_r_score_neg/10)
    print("Average Recall Neutral: ", sum_r_score_neu/10)
    print()
    print("Average F-Score Positive: ", sum_f_score_pos/10)
    print("Average F-Score Negative: ", sum_f_score_neg/10)
    print("Average F-Score Neutral: ", sum_f_score_neu/10)
    print()
    print("Average Accuracy: ", Sum_accuracy/10)
    print()


# predict class labels of test set using trainedClassifier
def testClassifier(classifier, tweet_tfidfvec, X_test):

    X_test_tfidfvec = tweet_tfidfvec.transform(X_test.astype('U'))
    y_predict_class = classifier.predict(X_test_tfidfvec)

    return y_predict_class