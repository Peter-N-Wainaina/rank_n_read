import json
import os
import pandas as pd
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS

from .processing import Processor
from .dataset import Dataset

app = Flask(__name__)
CORS(app)

processor = Processor()

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/getbooks", methods=["POST"])
def books_search():
    user_input = request.get_json()
    books = processor.get_recommended_books(user_input)
    result_json = jsonify(books)
    return result_json

# @app.route("/titles")
# def get_title_suggestions():
#     query = request.args.get("q", "").lower()

#     suggestions = []
#     for title in processor.books.keys():
#         if query in title.lower():
#             suggestions.append(title)

#     return jsonify(suggestions[:10])

# @app.route("/authors")
# def get_author_suggestions():
#     query = request.args.get("q", "").lower()
#     seen = set()
#     suggestions = []

#     for book_list in processor.books.values():
#         for book in book_list:
#             for author in book.get("authors", []):
#                 author_lower = author.lower()
#                 if query in author_lower and author_lower not in seen:
#                     suggestions.append(author)
#                     seen.add(author_lower)

#     return jsonify(suggestions[:10])

@app.route("/categories")
def get_category_suggestions():
    query = request.args.get("q", "").lower()
    seen = set()
    suggestions = []

    for book_list in processor.books.values():
        for book in book_list:
            for category in book.get("categories", []):
                category_lower = category.lower()
                if query in category_lower and category_lower not in seen:
                    suggestions.append(category)
                    seen.add(category_lower)

    return jsonify(suggestions[:5])

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)