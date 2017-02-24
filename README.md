Yogesh H. Kulkarni http://yati.io

# Segmentation of Scientific articles into zones

Automatically assigning input sentences to a set of categories. Sentence assigning process is based on it's semantic. Categories are defined with the indicator words.

# Data

"Sentence Classification Data Set" downloaded from Machine Learning Repository of Center for Machine Learning and Intelligent Systems - University of California, Irvine (http://cml.ics.uci.edu/).
Data Set contains data for  classification of sentences in scientific articles into categories:
  1. AIMX - The specific research goal of the paper
  2. OWNX - The author's own work (methods, results, conclusions...)
  3. CONT - Contrast, comparasion or critique of past work
  4. BASE - Past work that provides the basis for the work in the article
  5. MISC - Any other sentences.

Data Set contains sentences from the abstract and introduction of 90 scientific article from three different domains:
  1. PLoS Computational Biology (PLOS)
  2. The Machine Learning repository on arXiv (ARXIV)
  3. The rsychology journal Judgment and Decision Making (JDM).

# Directories

1. "training_set"
  - 24 PLOS articles with 893 sentences
  - 24 ARXIV articles with 793 sentences
  - 24 JDM articles with 814 sentences

2. "test_set"
  - 6 PLOS articles with 220 sentences
  - 6 ARXIV articles with 197 sentences
  - 6 JDM articles with 200 sentences

3. "word_lists"

  This directory contains one plaintext file for each of the 4 categories AIMX, OWNX, CONT and BASE. Each plaintext file lists the indicator words for the corresponding category. This directory also contains a stopwords file. The stopwords file contains stopwords that are not likely to be important for the taske of sentence classification (how, show, our...). File contains a set of stopwords that are not likely to be strong features for this task and thus can be safely removed.
  
# Implementation

There are quite a few ways of doing the sentence classification.

- Bag of Words approach: Train the model with all the senetences-words along with labels. Such implementation using Naive Bayes Classifier, Decision Tree Classifier and Support Vector Machines is shown in https://github.com/vdragan1993/sentence-classification

- Feature-based approach: Research papers like http://www.cs.pomona.edu/~achambers/papers/thesis_final.pdf and https://academic.oup.com/bioinformatics/article/28/7/991/210210/Automatic-recognition-of-conceptualization-zones discuss features that can be extracted from text which can be used for training-testing the machine learning model.

- Word2vec based approach: Instead of supplying whole senetnces or extracted hand-made features, vector representation of words-sentences is used as features for training the machine learning classifiers. This apporach is presnted here.
