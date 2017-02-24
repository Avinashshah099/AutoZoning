"""
    Scientific articles come in the form of sections or zones.
    The code below tries to segment/decompose the text into zones.
    Some portions of the code are taken refers sentence classification by 'Nenad Todorovic' and 'Dragan Vidakovic'
    [https://github.com/vdragan1993/sentence-classification]

    Author: Yogesh H Kulkarni
    Github: https://github.com/yogeshhk
"""

import os
import os.path
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import gensim
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score


stopwords = set(stopwords.words('english'))

def read_directory(directory):
    """
    Lists all file paths from given directory
    """

    ret_val = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            ret_val.append(str(directory) + "/" + str(file))
    return ret_val

def read_file(path):
    """
    Reads all lines from file on given path
    """

    f = open(path, "r")
    read = f.readlines()
    ret_val = []
    for line in read:
        if line.startswith("#"):
            pass
        else:
            ret_val.append(line)
    return ret_val

def read_line(line):
    """
    Returns sentence category and sentence in given line
    """

    splits = []
    s_category = ""
    sentence = ""
    if "\t" in line:
        splits = line.split("\t")
        s_category = splits[0]
        sentence = splits[1].lower()
    else:
        splits = line.split(" ")
        s_category = splits[0]
        sentence = line[len(s_category)+1:].lower()

    sentence = " ".join([word for word in word_tokenize(sentence) if word not in stopwords])
    # for sw in stopwords:
    #     sentence = sentence.replace(sw, "")
    pattern = re.compile("[^\w']")
    sentence = pattern.sub(' ', sentence) # Any non-characters (here ^ is for negation and not for the start) replace with white space
    sentence = re.sub(' +', ' ', sentence) # If more than one spaces, make them just one space
    return s_category, sentence

def read_testdata(input_folder):
    """
    Maps each sentence to it's category
    """

    test_folder = read_directory(input_folder)
    t_sentences = []
    t_categories = []
    for file in test_folder:
        lines = read_file(file)
        for line in lines:
            c, s = read_line(line)
            if s.endswith('\n'):
                s = s[:-1]
            t_sentences.append(s)
            t_categories.append(c)
    return t_categories, t_sentences


def comput_sentence2vec(model,sentence):
    return np.array([np.mean([model[w] for w in sentence.split() if w in model], axis=0)])

def fill_dataframe(sentences_vecs, categories_training):
    vec_len = len(sentences_vecs[0][0])
    col_names = [ "Feature_" + str(i) for i in range(1, vec_len+1)] + ["Target"]
    df = pd.DataFrame(columns=col_names)
    for i in range(len(sentences_vecs)):
        vec = sentences_vecs[i]
        # if i > 500:
        #     break
        try:
            ndarray = vec[0]
            len_array = len(ndarray)
            print(".{}.".format(i))
            df.loc[i,"Target"] = categories_training[i]#).replace(map_to_int)
            for j in range(len_array):
                df.loc[i,col_names[j]] = ndarray[j]
        except:
            print("Error at {} sentence...".format(ndarray))
            continue

    # Target label encoding
    targets = df["Target"].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df["Target"] = df["Target"].replace(map_to_int)
    return df

def extract_zones_deep_learning():
    """
        Gensim method using Word2Vec
    """

    # prepare training and test data
    categories_training, sentences_training = read_testdata("training_set")
    categories_testing, sentences_test = read_testdata("test_set")
    all_sentences_list = [word_tokenize(sentence) for sentence in sentences_training]

    # trains here itself. if you get more sentences, use "train" method
    modelword = gensim.models.Word2Vec(all_sentences_list,sg=1)
    modelword.init_sims()

    sentences_training_vecs = [comput_sentence2vec(modelword,sentence) for sentence in sentences_training]
    sentences_testing_vecs = [comput_sentence2vec(modelword,sentence) for sentence in sentences_test]

    save_training_df_file = "training_word2vec.csv"
    if os.path.exists(save_training_df_file):
        df_training = pd.read_pickle(save_training_df_file)
    else:
        df_training = fill_dataframe(sentences_training_vecs, categories_training)
        df_training.to_pickle(save_training_df_file)

    save_testing_df_file = "testing_word2vec.csv"
    if os.path.exists(save_testing_df_file):
        df_testing = pd.read_pickle(save_testing_df_file)
    else:
        df_testing = fill_dataframe(sentences_testing_vecs, categories_testing)
        df_testing.to_pickle(save_testing_df_file)


    features = ["Feature_" + str(i) for i in range(1, 100 + 1)] # Default word2vec size is 100
    train_Y = df_training["Target"]
    train_X = df_training[features]
    test_X = df_testing[features]
    test_Y = df_testing["Target"]

    classifiers = []
    classifiers.append(("LogisticRegressionCV", LogisticRegressionCV()))
    classifiers.append(("NaiveBayesClassifier", GaussianNB())) # THIS DOES WORSE as it does not like -ve values in word2vec
    classifiers.append(("SVM", svm.SVC()))
    classifiers.append(("SGDClassifier", SGDClassifier(loss='log', penalty='l1')))

    for name, clf in classifiers:
        print("----------------------------------------")
        print("Testing with " + name)
        clf.fit(train_X, train_Y)

        ypred_train = clf.predict(train_X)
        ypred_test = clf.predict(test_X)

        train_acc = accuracy_score(train_Y, ypred_train)
        print('Training Accuracy score: {}'.format(train_acc))

        test_acc = accuracy_score(test_Y, ypred_test)
        print('Testing Accuracy score: {}'.format(test_acc))

        diff = ypred_test - test_Y
        rsqr = np.mean(diff * diff)
        print("Mean squared error: {}".format(rsqr))


if __name__ == "__main__":
    # extract_zones_without_features()
    extract_zones_deep_learning()