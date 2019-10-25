from TF_IDF import tf_idf
from BM25_vectorizer import bm25
from FastText import fastText, get_word_vectors, get_text_vectors, tokenize_texts  

#%run BM25_vectorizer.ipynb

from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/results')
def results():
    query = request.args['q']
    if request.args['model'] == 'Tf-IDF':
        r = tf_idf(query)
    elif request.args['model'] == 'BM25':
        r = bm25(query)
    elif request.args['model'] == 'FastText':
        r = fastText(query, get_word_vectors(), get_text_vectors(tokenize_texts(), get_word_vectors()))
                     
    return render_template('results.html', rezult=r)
                     
if __name__ == '__main__':
    app.run(host='127.0.0.1', port = 80)
