import sys
import codecs
from os.path import dirname, realpath
import random
import json

import nltk
from nltk.metrics.scores import precision, recall, f_measure
from nltk.metrics import ConfusionMatrix
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

import client as cl

import time

start = time.time()

def load_text(file_name):
    # Define a function > open() method, to minimize reading/splitting errors.

    bar = '\\' if sys.platform == 'win32' or 'win64' else '/'
    curr_dir = dirname(realpath(__file__)) + bar
    file_text_route = curr_dir + file_name
    with codecs.open(file_text_route, mode='r', encoding='utf-8') as file_text:
        text_file = file_text.read()
    return text_file

def create_initial_corpus(file_pos, file_neg):

    reviews_pos = load_text(file_pos).split('\n')
    reviews_neg = load_text(file_neg).split('\n')
    
    review_list = []
    for n in reviews_pos:
        review_list.append((n, 'pos'))
    for n in reviews_neg:
        review_list.append((n, 'neg'))
    random.shuffle(review_list)
    return review_list

var_initial_corpus = create_initial_corpus("pos_revs.txt", "neg_revs.txt")

list_pos = json.load(open('positives.json'))
list_neg = json.load(open('negatives.json'))

def feature_extractor(review):
    
    feature_vector = {}  
    words = pos_tag(word_tokenize(review))
    #  (word, pos_tag)

    lemmatizer = WordNetLemmatizer()
    for (word, tag) in words:
        if tag == 'JJ' or 'JJR' or 'JJS':
            word = lemmatizer.lemmatize(word, 'a')
        else:
            word = lemmatizer.lemmatize(word, 'n')
        if word in list_pos:
            feature_vector[word] = True
        elif word in list_neg:
            feature_vector[word] = True

    return feature_vector

def create_feature_corpus(initial_corpus):

    feature_corpus = []
    for (review, tag) in initial_corpus:
        vector = feature_extractor(review)
        feature_corpus.append((vector, tag))

    return feature_corpus

var_feature_corpus = create_feature_corpus(var_initial_corpus)

def divide_corpus(corpus):

    lim_1 = int(len(corpus) * 0.80)
    lim_2 = int(len(corpus) * 0.90)
    training_corpus = corpus[0:lim_1]
    test_corpus = corpus[lim_1:lim_2]
    evaluation_corpus = corpus[lim_2:]

    return [training_corpus, test_corpus, evaluation_corpus]


def error_check(method, test_icorpus):

    error_list = []
    for (review, test_tag) in test_icorpus:

        given_tag = method.classify(feature_extractor(review))
       
        if given_tag != test_tag:
            error_list.append(
                f'The review \'{review}\' is {test_tag}, '
                f'but the classifier identified as {given_tag}.')
    return error_list


def evaluation(method, reference_corpus, test_corpus):
    
    ref_set_pos = set()
    test_set_pos = set()
    ref_set_neg = set()
    test_set_neg = set()
    for i, (n, t) in enumerate(reference_corpus):
        if t == 'pos':
            ref_set_pos.add(i)
        else:
            ref_set_neg.add(i)
    for i, (v, t) in enumerate(test_corpus):
        if method.classify(v) == 'pos':
            test_set_pos.add(i)
        else:
            test_set_neg.add(i)

    precis_pos = precision(ref_set_pos, test_set_pos)
    rec_pos = recall(ref_set_pos, test_set_pos)
    f1_pos = f_measure(ref_set_pos, test_set_pos)
    precis_neg = precision(ref_set_neg, test_set_neg)
    rec_neg = recall(ref_set_neg, test_set_neg)
    f1_neg = f_measure(ref_set_neg, test_set_neg)

    return precis_pos, rec_pos, f1_pos, precis_neg, rec_neg, f1_neg

var_training_icorpus = divide_corpus(var_initial_corpus)[0]
var_test_icorpus = divide_corpus(var_initial_corpus)[1]
var_evaluation_icorpus = divide_corpus(var_initial_corpus)[2]

var_training_fcorpus = divide_corpus(var_feature_corpus)[0]
var_test_fcorpus = divide_corpus(var_feature_corpus)[1]
var_evaluation_fcorpus = divide_corpus(var_feature_corpus)[2]

print(
    f'\nOur positive corpus has {len(load_text("pos_revs.txt").splitlines())} elements.'
    f'\nOur negative corpus has {len(load_text("neg_revs.txt").splitlines())} elements.'
    )
print(
    f'Our total corpus has {len(var_initial_corpus)} elements in total.')



naive_bayes_model = nltk.NaiveBayesClassifier.train(var_training_fcorpus)

""" naive_bayes_model.show_most_informative_features(10) """

error_list_nb = error_check(naive_bayes_model, var_test_icorpus)

""" print(f'\nNaive Bayes s Classifier:')
print(f'\nNaives Bayes s classifier has {len(error_list_nb)} errors.'
      f'\nFirst 20 are:')
for line in error_list_nb[:20]:
    print(line)"""

accuracy_nb = nltk.classify.util.accuracy(naive_bayes_model, var_evaluation_fcorpus)
metrics_nb = evaluation(naive_bayes_model, var_evaluation_icorpus, var_evaluation_fcorpus)

""" print(
    f'\nNaive Bayes accuracy is {accuracy_nb}.')
print(
    f'Positive reviews preccission: {metrics_nb[0]}, '
    f'la cobertura es {metrics_nb[1]}, F1 = {metrics_nb[2]}.')
print(
    f'Negative reviews preccission: {metrics_nb[3]}, '
    f'la cobertura es {metrics_nb[4]}, F1 = {metrics_nb[5]}.') """

reviews_db_dic = cl.changeRevFormat()
reviews_db_class_dic = {}

def classify_nb_model(review):
    class_tag = naive_bayes_model.classify(feature_extractor(review))
    return class_tag

for c, (k,v) in enumerate(reviews_db_dic .items()):
     
    for i,j in v.items():
        if i == 'review':
            tag = classify_nb_model(j)
            rev_tag = {(i,j): tag}
            reviews_db_class_dic[c] = rev_tag
            print(f'\nIteration nº{c}: {reviews_db_class_dic[c]}')

print(f'\nFinal iteration: {reviews_db_class_dic}')

end = time.time()
print(f'The program took {(end-start)} seconds to finish')