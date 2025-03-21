import json
import os
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from processing import processing
from dataset import Dataset

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'data_exploration', 'books.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)


# Extract all the book details into a list
books_data = []
for book_title in data:
    book_details = data[book_title]
    for book in book_details:
        books_data.append(book)

#Assuming the "books" key holds the books dat in the JSON 
books_df = pd.DataFrame(books_data)

app = Flask(__name__)
CORS(app)

def get_books():
    #randomly select 10 books from our dataset
    sample_books = books_df.head(10)

    #converting the DataFrame JSON format, selecting the all the columns
    books_json = sample_books.to_json(orient='records')
    return books_json


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

#@app.route("/getbooks", methods=["POST"])
@app.route("/getbooks")
def books_search():
    # data = request.get_json()
    # titles = data.get("titles", [])
    # authors = data.get("authors", [])
    # categories = data.get("categories", [])
    # call function that takes in titles, authors, categories as input and returns reccomended books
    books = get_books()
    return jsonify(json.loads(books)) 


@app.route('/getbooks', methods=['POST'])
def get_rec_books():
    """
    Returns book recommendations from user inputs
    """
    data = request.get_jason()

    author = data.get('author', None)
    category = data.get('category', None)
    title = data.get('title', None)

    #ensuring we have atleast one of the three fields
    if not (author or category or title):
        return jsonify({"error": "At least one of 'author', 'category', or 'title' must be provided"}), 400
    
    recommended_books = []
    #get recommendations for the title
    if title:
        books_by_title = processing.get_recs_from_title(title, Dataset.books)
        recommended_books.extend(books_by_title)

    #get recommendations by author
    if author:
        books_by_author = processing. get_recs_from_author(author)
        recommended_books.extend(books_by_author)

    #get recomendations by categories
    if category:
        books_by_category = processing.get_recs_from_categories(category)
        recommended_books.extend(books_by_category)

    #removing duplicates by converting to set
    unique_books = {book['title']: book for bok in recommended_books}.values()

    result_books = list(unique_books)[:50]

    return jsonify(result_books)



if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)