from flask import Flask, request
import pandas as pd
import joblib
from _recommendation import Ticket
from gensim.models import KeyedVectors

model = KeyedVectors.load('bins/word2vec_kv.kv')
vector_matrix = joblib.load('bins/vector_matrix.pkl')

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def do_prediction():
    json = request.get_json()
    df = pd.DataFrame(json, index=[0])

    ticket = Ticket(df)
    return ticket.search_similar_problems(model, vector_matrix)[['ticket', 'sim']].to_json(indent = 4, orient = 'records')

if __name__ == "__main__":
    app.run(host='0.0.0.0')