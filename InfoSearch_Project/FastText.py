import numpy as np
import pandas as pd

#%load_ext autoreload

from gensim.models import Word2Vec, KeyedVectors

def get_word_vectors():
    model_file = 'model.model'
    word_vectors = KeyedVectors.load(model_file)
    return word_vectors

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

import re
import nltk
nltk.download('stopwords')
stemmer = nltk.stem.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('russian'))

def texts():
    table = pd.read_csv('quora_question_pairs_rus.csv')
    
    texts = []
    for row in table['question2']:
        texts.append(str(row))
    return texts

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

def tokenize_texts():
    tex = texts()
    
    allWordsAllTexts = []
    for text in tex[0:100]:
        sent = tokenize(text)
        allWordsAllTexts.append(sent)
    return allWordsAllTexts

def get_text_vectors(allWordsAllTexts, word_vectors):
    vectors = []
    #allWordsAllTexts = tokenize_texts()
    #word_vectors = get_word_vectors()
    
    for s in allWordsAllTexts:
        # создаем маски для векторов 
        lemmas_vectors = np.zeros((len(s), word_vectors.vector_size))
        vec = np.zeros((word_vectors.vector_size,))
        # если слово есть в модели, берем его вектор
        for idx, lemma in enumerate(s):
            if lemma in word_vectors.vocab:
                lemmas_vectors[idx] = word_vectors[lemma]
                if lemmas_vectors.shape[0] is not 0:
                    vec1 = np.mean(lemmas_vectors, axis=0)
        vectors.append(vec1)
    return vectors

import sklearn
from sklearn.metrics.pairwise import cosine_similarity

def fastText(query, word_vectors, vectors):
    lemmas_vectors_inputs = np.zeros((len(query), word_vectors.vector_size))
    
    # если слово есть в модели, берем его вектор
    for idx, lemma in enumerate(query):
        if lemma in word_vectors.vocab:
            lemmas_vectors_inputs[idx] = word_vectors[lemma]
            if lemmas_vectors_inputs.shape[0] is not 0:
                vec = np.mean(lemmas_vectors_inputs, axis=0)
    
    bests = {}
    m = []
    mean_input = vec.reshape(1, -1)
    for idx1, j in enumerate(vectors):
        mean_text = j.reshape(1, -1)
        sk = cosine_similarity(mean_input, mean_text, dense_output=True)
        m.append(sk[0][0])
    best_result = m.index(max(m))
    bests[idx] = best_result
    
    return bests
