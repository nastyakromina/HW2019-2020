#pip install pymorphy2

import os
import pymorphy2
import pandas as pd
morph = pymorphy2.MorphAnalyzer()

import re
import nltk
nltk.download('stopwords')
stemmer = nltk.stem.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('russian'))

from collections import Counter
import operator
import math

def tokenize(doc):
    allWordsInOneText = []
    text = doc.split()
    for word in text:
        if word not in stopwords:
            word = re.sub('[.,-;:?!@#$%^&()_+=—\ufeff–"…«»>wwwtvsubtitlesnet]', '', word).lower()
            if word != "":
                p = morph.parse(word)[0]
                allWordsInOneText.append(p.normal_form)
    return allWordsInOneText

def build_terms(corpus):
    terms = {}
    current_index = 0
    for doc in corpus:
        for word in tokenize(doc):
            if word not in terms:
                terms[word] = current_index
                current_index += 1
    return terms

def tf(document, terms):
    words = tokenize(document)
    total_words = len(words)
    doc_counter = Counter(words)
    for word in doc_counter:
        doc_counter[word] /= total_words
    tfs = [0 for _ in range(len(terms))]
    for term, index in terms.items():
        tfs[index] = doc_counter[term]
    return tfs

def _count_docs_with_word(word, docs):
    counter = 1
    for doc in docs:
        if word in doc:
            counter += 1
    return counter

# documents - это корпус
def idf(documents, terms):
    idfs = [0 for _ in range(len(terms))]
    total_docs = len(documents)
    for word, index in terms.items():
        docs_with_word = _count_docs_with_word(word, documents)
        idf = 1 + math.log10(total_docs / docs_with_word)
        idfs[index] = idf
    return idfs

def _merge_td_idf(tf, idf, terms):
    return [tf[i] * idf[i] for i in range(len(terms))]


def build_tfidf(corpus, document, terms):
    doc_tf = tf(document, terms)
    doc_idf = idf(corpus, terms)
    return _merge_td_idf(doc_tf, doc_idf, terms)


def cosine_similarity(vec1, vec2):
    def dot_product2(v1, v2):
        return sum(map(operator.mul, v1, v2))

    def vector_cos5(v1, v2):
        prod = dot_product2(v1, v2)
        len1 = math.sqrt(dot_product2(v1, v1))
        len2 = math.sqrt(dot_product2(v2, v2))
        return prod / (len1 * len2)

    return vector_cos5(vec1, vec2)

def tf_idf(query):
    table = pd.read_csv('quora_question_pairs_rus.csv')
    
    texts = []
    for row in table['question2']:
        texts.append(str(row))
    
    tf_idf_total = []
    corpus = texts[0:1000]
    terms = build_terms(corpus)
    
    for document in corpus:
        tf_idf_total.append(build_tfidf(corpus, document, terms))
    
    results = {}
    query_tfidf = build_tfidf(corpus, query, terms)
    
    for index, document in enumerate(tf_idf_total):
        results[index] = cosine_similarity(query_tfidf, document)
        
    ld = list(results.items())
    ld.sort(key=lambda i: i[1], reverse= True)
    
    return texts[ld[0][0]]
