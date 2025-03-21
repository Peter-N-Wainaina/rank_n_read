import json
import os
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from collections import defaultdict
from processing import Processing

processing = Processing()

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/getbooks")
def books_search():
    """Return a JSON response of randomly selected books"""
    books = processing.get_books()
    return jsonify(json.loads(books))

@app.route("/get_from_categories", methods=["GET"])
def get_recs_from_categories():
    """
    Returns book recommendations from a list of genre queries.
    """
    query = request.args.get("query")
    recommendations = processing.get_recs_from_categories(query)
    return recommendations

@app.route("/get_from_authors", methods=["GET"])
def get_recs_from_authors():
    """
    Returns book recommendations from an author input.
    """
    author = request.args.get("query")
    recommendations = processing.get_recs_from_author(author)
    return recommendations


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)