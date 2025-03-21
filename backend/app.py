import json
import os
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from collections import defaultdict
from processing import Processing

# ROOT_PATH for linking with all your files. 
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'data_exploration', 'books.json')

processing = Processing(json_file_path)

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
    pass

@app.route("/get_from_authors", methods=["GET"])
def get_recs_from_authors():
    """
    Returns book recommendations from an author input.
    """
    author = request.args.get("query")
    pass


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)