from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)

# app = Flask(__name__)
CORS(app)

# Load data
def load_data(movies_path, ratings_path):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

movies_path = 'https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/movies.csv'
ratings_path = 'https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/ratings.csv'
movies, ratings = load_data(movies_path, ratings_path)

# Preprocess data
def preprocess_movie_name(movie_name):
    return re.sub(r'\W+', ' ', movie_name).strip()

def preprocess_movie_database(movies_df):
    movies_df['title'] = movies_df['title'].apply(lambda x: re.sub(r'\W+', ' ', x).strip())
    return movies_df

# Vectorize movie genres using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

# Train Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(tfidf_matrix, movies['title'])

# Movie recommendation function
def get_movie_recommendation_nb(movie_name, model, vectorizer, movies_df, k=10):
    movie_name = preprocess_movie_name(movie_name)
    movies_df = preprocess_movie_database(movies_df)
    movie_idx = movies_df[movies_df['title'].str.contains(movie_name, case=False)].index
    if len(movie_idx) > 0:
        movie_features = vectorizer.transform(movies_df.iloc[movie_idx]['genres'])
        similarity_scores = cosine_similarity(movie_features, tfidf_matrix)
        similar_movies_idx = np.argsort(similarity_scores[0])[::-1][1:k+1] 
        recommendations = movies_df.iloc[similar_movies_idx][['title', 'genres']]
        return recommendations.to_dict('records')
    else:
        return {"error": "No movies found. Please check your input."}

# Define route for recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    movie_name = request.args.get('movie')
    if movie_name:
        results = get_movie_recommendation_nb(movie_name, nb_model, tfidf_vectorizer, movies)
        return jsonify(results)
    else:
        return jsonify({"error": "Please provide a movie name."})

if __name__ == '__main__':
    app.run(debug=True)
