from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

from flask import Flask, render_template,Response, request
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__,  static_url_path='', static_folder='static')

# Get your API keys from openai, you will need to create an account.
import os
os.environ["OPENAI_API_KEY"] = "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"
chain = None
docsearch = None


def init():
    global chain, docsearch
    # using embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.load_local("D:/compi project files/faiss_index", embeddings)

    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI

    chain = load_qa_chain(OpenAI(), chain_type="stuff")


@app.route('/q/<query>',  methods=['GET'])
def query_handler(query):
    global chain, docsearch

    #query = request.args.get('query')
    # queries
    #query = "how direct access sort is different compared to bubble sort?"
    docs = docsearch.similarity_search(query)
    return chain.run(input_documents=docs, question="explain "+query+" what is document name")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('CONTACT.html')





if __name__=="__main__":
    init()
    app.run(debug=True)
