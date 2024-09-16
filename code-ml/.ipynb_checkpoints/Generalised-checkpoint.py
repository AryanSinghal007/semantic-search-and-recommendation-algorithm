# Import necessary libraries

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For object serialization
import pickle

# For progress bars
import tqdm

# For numerical operations
import numpy as np

# For data manipulation and analysis
import pandas as pd

# For plotting (not used in this script)
# import matplotlib.pyplot as plt

# For word embeddings
from gensim.models import Word2Vec

# For tokenizing text
from nltk.tokenize import word_tokenize

# For parallel processing
import multiprocessing

# For approximate nearest neighbor search
from annoy import AnnoyIndex

# For regular expressions
import re

# For stopwords removal
from nltk.corpus import stopwords

# For word stemming
from nltk.stem.snowball import SnowballStemmer

# Load and preprocess data
df = pd.read_csv('data.csv')
df.dropna(inplace = True)  # Remove rows with missing values

# Group data by relevant columns and aggregate
group_columns = ['id_column', 'text_column']  # Replace with your actual column names
agg_dict = {
    'description_column': 'first',
    'preprocessed_description_column': 'first',
    'count_column': 'sum'
}
df = df.groupby(group_columns).agg(agg_dict).reset_index()

# Sort data by count and keep top 60% for each ID
df = df.sort_values(by='count_column', ascending=False).groupby('id_column').apply(lambda x: x.head(int(0.6*len(x)))).reset_index(drop=True)

# Save processed data
df.to_csv('processed_data.csv', index=False)

# Create Word2Vec model
sentences = [word_tokenize(sent) for sent in df['text_column'].tolist()]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=multiprocessing.cpu_count())
model.save("word2vec.model")

# Create Annoy index for fast similarity search
dim = 100  # Dimension of Word2Vec vectors
t = AnnoyIndex(dim, 'euclidean')  # Create Annoy index using Euclidean distance

# Function to get vector representation of a sentence
def get_vector(s):
    tokenized_phrase = word_tokenize(s)
    filtered_tokenized_phrase = [word for word in tokenized_phrase if word in model.wv.key_to_index]
    average_vector = sum([model.wv.get_vector(word) for word in filtered_tokenized_phrase]) / len(filtered_tokenized_phrase)
    return average_vector

# Add items to Annoy index
for j, i in tqdm.tqdm(enumerate(df['text_column'].tolist()), total=len(df)):
    vector = get_vector(i)
    t.add_item(j, vector)

# Build and save Annoy index
t.build(10)  # 10 trees for better accuracy
t.save('AnnIndex.ann')

# Create mapping dictionaries for fast lookup
index = list(range(len(df)))
desc_map = dict(zip(index, df['description_column'].tolist()))
text_map = dict(zip(index, df['text_column'].tolist()))
id_map = dict(zip(index, df['id_column'].tolist()))
count_map = dict(zip(index, df['count_column'].tolist()))

# Load standard data (if applicable)
df_std = pd.read_csv('standard_data.csv')
df_std.columns = ['standard_id', 'standard_description']

# Text cleaning function
def clean_query(query):
    stopwords_list = set(stopwords.words('english'))
    s_stemmer = SnowballStemmer('english')
    x = re.sub(r'[^\w\s\d]', ' ', query)  # Remove non-alphanumeric characters
    x = re.sub(r'(?<=\d)(?=\D)|(?<=\D)(?=\d)', ' ', x)  # Add space between numbers and letters
    x = re.sub(r'\b\w{1}\b', '', x)  # Remove single-character words
    x = re.sub(r'(\_+)', '', x)  # Remove underscores
    x = re.sub(r'\s+', ' ', x)  # Remove extra spaces
    x = x.strip().lower()  # Convert to lowercase and remove leading/trailing spaces
    x = ' '.join([i for i in x.split() if i not in stopwords_list])  # Remove stopwords
    x = ' '.join([s_stemmer.stem(i) for i in x.split()])  # Apply stemming
    return x

# Search function
def search_similar_items(query, top_k=10):
    similar_indices = t.get_nns_by_vector(get_vector(query), top_k)  # Get similar items from Annoy index
    results = []
    for i in similar_indices:
        results.append({
            'description': desc_map[i],
            'text': text_map[i],
            'id': id_map[i],
            'count': count_map[i]
        })
    results = sorted(results, key=lambda x: x['count'], reverse=True)  # Sort by count
    
    # Add standard items if applicable
    standard_ids = ['@' + str(i['id'])[:4] for i in results][:4]
    for _, row in df_std[df_std['standard_id'].isin(standard_ids)].iterrows():
        results.append({
            'standard_id': row['standard_id'],
            'standard_description': row['standard_description'],
        })
    
    # Format IDs if necessary
    for item in results:
        if 'id' in item:
            item['id'] = str(item['id']).zfill(4)  # Ensure ID is 4 digits
        if 'standard_id' in item:
            item['standard_id'] = item['standard_id'][1:]  # Remove '@' from standard ID
    
    return results

# Main search loop
while True:
    query = input("Enter your query (or 'exit' to quit): ")
    if query.lower() == "exit":
        break
    cleaned_query = clean_query(query)
    results = search_similar_items(cleaned_query)
    print("Search Results:")
    for result in results:
        if 'description' in result:
            print(f"Description: {result['description']}")
            print(f"ID: {result['id']}")
            print(f"Count: {result['count']}")
        elif 'standard_id' in result:
            print(f"Standard ID: {result['standard_id']}")
            print(f"Description: {result['standard_description']}")
        print("-" * 30)

# Explanation of the main components:

# 1. Data Loading and Preprocessing:
#    - Ld data from CSV file
#    - Remove rows with no values or duplicates
#    - Group data by relevant columns and aggregate
#    - Sort data by count and keep top 60% for each ID
#    - Save processed data

# 2. Word2Vec Model:
#    - Create a Word2Vec model from the text data
#    - This model creates vector representations of words

# 3. Annoy Index:
#    - Create an Annoy index for fast similarity search
#    - Add vector representations of sentences to the index
#    - Build and save the index for future use

# 4. Mapping Dictionaries:
#    - Create dictionaries for fast lookup of descriptions, texts, IDs, and counts

# 5. Text Cleaning:
#    - Define a function to clean and preprocess query text
#    - to only keep alphanumeric characters, remove stopwords, and apply stemming

# 6. Search Function:
#    - Use Annoy index to find similar items
#    - Retrieve details of similar items from mapping dictionaries
#    - Sort results by count
#    - Add standard items if applicable
#    - Format IDs as needed

# 7. Main Search Loop:
#    - Continuously prompt user for queries
#    - Clean the query
#    - Perform search and display results
#    - Exit when user types 'exit'

# This script provides a flexible framework for searching similar items based on text descriptions.
# It uses Word2Vec for creating meaningful vector representations of text and Annoy for fast
# approximate nearest neighbor search. The preprocessing steps and the use of mapping dictionaries
# help in optimizing the search process and presenting relevant results quickly.