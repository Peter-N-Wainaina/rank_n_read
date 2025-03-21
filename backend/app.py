import json
import os
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

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

@app.route("/getbooks")
def books_search():
    books = get_books()
    return jsonify(json.loads(books))

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)