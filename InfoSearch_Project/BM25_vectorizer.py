import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = CountVectorizer()
import operator

import nltk
nltk.download('stopwords')
stemmer = nltk.stem.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('russian'))

import numpy as np
import re

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

def tokenize(doc):
    allWordsInOneText = []
    text = doc.split()
    for word in text:
        if word not in stopwords:
            word = re.sub('[.,-;:?!@#$%^&()_+=—\ufeff–"…«»>]', '', word).lower()
            if word != "":
                p = morph.parse(word)[0]
                allWordsInOneText.append(p.normal_form)
    return allWordsInOneText

def tokenize_text(allcorpus):
    table = pd.read_csv('quora_question_pairs_rus.csv')
    
    texts = []
    for row in allcorpus:
        texts.append(str(row))
    
    allWordsAllTexts = []
    for text in texts[0:100]:
        sent = tokenize(text)
        allWordsAllTexts.append(sent)
    return allWordsAllTexts

def get_array_string(array):
    arrayString = []
    for text in tokenize_text(array):
        s = ' '.join(text)
        arrayString.append(s)
    return arrayString

def get_matrix_bm25(arrayString, allWordsAllTexts):
    X = vectorizer.fit_transform(arrayString)
    Matrix = X.toarray()
    
    lens = []
    for i in arrayString:
        lens.append(len(i))
        
    tf_matrix = Matrix/np.array(lens).reshape((-1,1))
    
    vectorizer1 = TfidfVectorizer()
    X1 = vectorizer1.fit_transform(arrayString)
    
    Matrix1 = X1.toarray()
    
    k = 2
    b = 0.75
    
    avgdl = sum([len(doc) for doc in tokenize_text(allWordsAllTexts)])/len(allWordsAllTexts)
    
    Matrix_bm25 = np.zeros((100, 436))
    
    for i in range(100):
        for j in range(436):
            if tf_matrix[(i,j)] != 0.0:
                b = ((Matrix1[(i,j)]*(2+1))/tf_matrix[(i,j)]+k*(1-0.75+0.75*(lens[i]/avgdl)))
                Matrix_bm25[i,j] = b
    
    return Matrix_bm25

def bm25(query):
    N = 1000 #количество документов коллекции
    table = pd.read_csv('quora_question_pairs_rus.csv')
    
    texts = []
    for row in table['question2']:
        texts.append(str(row))
        
    X = vectorizer.fit_transform(get_array_string(tokenize_text(table['question2'])))
    words = vectorizer.get_feature_names()
        
    q = np.zeros(shape=(1,436))
    
    t = tokenize(query)
    
    for ind, j in enumerate(t):
        for index, i in enumerate(words):
            if j == i:
                q[0][index] += 1
    res = {}
    for index, m in enumerate(get_matrix_bm25(get_array_string(tokenize_text(table['question2'])),
                                              table['question2'])):
        res[index] = np.dot(m, q[0])
    ld = list(res.items())
    ld.sort(key=lambda i: i[1], reverse= True)
    
    return texts[ld[0][0]]
