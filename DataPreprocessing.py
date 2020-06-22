# @authors: Anusha Voloju , Yukthi Papanna Suresh
# CS 583, Spring 2019
# Project2: Sentiment Classification of Tweets

import re
import nltk
import emoji
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

def preprocessTweetText(tweets):
    # # remove stopwords from tweets
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                  'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                  'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                  'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                  'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                  'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                  'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 's', 't', 'can',
                  'will', 'just', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ma']
    ps = PorterStemmer()

    tweets_preprocess = dict()
    i = 0

    tweet_tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)

    for tweet in tweets.Anootated_tweet:
        # remove tags from tweets
        removetags = re.compile('<.*?>')
        cleanedtags = removetags.sub('', str(tweet))
        tweet = cleanedtags

        # remove tags from tweets
        cleanedurls = re.sub(r"http\S+", "", tweet)
        tweet = cleanedurls

        # preprocess the tweet text by removing stop words and perform stemming
        tweet_preprocess = ""
        for w in tweet.split():
            w = w.lower()
            for c in w:
                if c in '#' or c in '@':
                    w = w.replace(c, " ")
                if c in emoji.UNICODE_EMOJI:
                    st = ""
                    low = emoji.UNICODE_EMOJI[c].split('_')
                    for item in low:
                        st = st + " " + item
                    w = w.replace(c, st)

            w = ''.join([i if ord(i) < 128 else '' for i in w])
            w = re.sub(r'[^\x00-\x7F]+', ' ', w)
            if w not in stop_words and not w.isdigit():
                word = wordnet_lemmatizer.lemmatize(w)
                #word = ps.stem(w)
                tweet_preprocess = tweet_preprocess + " " + word

        tweet = tweet_preprocess

        tweets_preprocess[i] = tweet
        i = i + 1

    # update Obamatweets.Anootated_tweet with the pre processed tweets
    for i, tweet in tweets_preprocess.items():
        tweets.Anootated_tweet[i] = tweet

    return tweets