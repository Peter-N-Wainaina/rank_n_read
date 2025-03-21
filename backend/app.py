import json
import os
import pandas as pd
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS

from .processing import Processing
from .dataset import Dataset

app = Flask(__name__)
CORS(app)

processor = Processing()

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/getbooks", methods=["POST"])
def books_search():
    user_input = request.get_json()
    books = processor.get_recommended_books(user_input)
    return jsonify(books)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)