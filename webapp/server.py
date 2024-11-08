from flask import Flask, request, jsonify
from waitress import serve
from catboost import CatBoostClassifier
import pandas as pd

import os
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)

model = CatBoostClassifier()
model.load_model("../data/model.cbm")
stemmer = SnowballStemmer(language="russian")

STOPWORDS_RU = set(stopwords.words("russian"))
EXECUTOR_SIZE = pd.read_parquet("../data/executor_size.parquet")
REGION_SIZE = pd.read_parquet("../data/region_size.parquet")


@app.route("/")
def root():
    """
    Статичный контент index.html
    """
    with open("index.html", encoding='utf-8') as file:
        return file.read()


@app.route("/process", methods=["GET", "POST"])
def process():
    if request.method == "POST":
        region_size = REGION_SIZE.loc[
            REGION_SIZE.index == int(request.form["inputRegion"])
        ]
        if region_size.empty:
            region_size = "small"
        else:
            region_size = region_size.at[
                int(request.form["inputRegion"]), "region_size"
            ]

        executor_size = EXECUTOR_SIZE.loc[
            EXECUTOR_SIZE.index == int(request.form["inputINN"])
        ]
        if executor_size.empty:
            executor_size = "small"
        else:
            executor_size = executor_size.at[
                int(request.form["inputINN"]), "executor_size"
            ]

        data = [
            request.form["inputContractStatus"],
            request.form["inputBudgetLevel"],
            float(request.form["inputContractDuration"]),
            float(request.form["inputPrice"]),
            region_size,
            executor_size,
            prepare_text(request.form["inputData"]),
        ]
        print(data)

        try:
            result = get_prediction(model, data)
        except:
            result = "Ошибка"

        return jsonify({"result": result})


def get_prediction(model, data):
    result = model.predict(data)
    print(result)
    return result[0]


def prepare_text(text, stemmer=stemmer, stopwords=STOPWORDS_RU):
    text = re.sub(r"[^а-яё]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not t in stopwords]
    return " ".join([stemmer.stem(t) for t in tokens])


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
