# from flask import Flask, request, jsonify
# import numpy as np
# import pandas as pd
# import re
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from flask_cors import CORS

# app = Flask(__name__)

# # app = Flask(__name__)
# CORS(app)

# # Load data
# def load_data(movies_path, ratings_path):
#     movies = pd.read_csv(movies_path)
#     ratings = pd.read_csv(ratings_path)
#     return movies, ratings

# movies_path = 'https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/movies.csv'
# ratings_path = 'https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/ratings.csv'
# movies, ratings = load_data(movies_path, ratings_path)

# # Preprocess data
# def preprocess_movie_name(movie_name):
#     return re.sub(r'\W+', ' ', movie_name).strip()

# def preprocess_movie_database(movies_df):
#     movies_df['title'] = movies_df['title'].apply(lambda x: re.sub(r'\W+', ' ', x).strip())
#     return movies_df

# # Vectorize movie genres using TF-IDF
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

# # Train Naive Bayes classifier
# nb_model = MultinomialNB()
# nb_model.fit(tfidf_matrix, movies['title'])

# # Movie recommendation function
# def get_movie_recommendation_nb(movie_name, model, vectorizer, movies_df, k=10):
#     movie_name = preprocess_movie_name(movie_name)
#     movies_df = preprocess_movie_database(movies_df)
#     movie_idx = movies_df[movies_df['title'].str.contains(movie_name, case=False)].index
#     if len(movie_idx) > 0:
#         movie_features = vectorizer.transform(movies_df.iloc[movie_idx]['genres'])
#         similarity_scores = cosine_similarity(movie_features, tfidf_matrix)
#         similar_movies_idx = np.argsort(similarity_scores[0])[::-1][1:k+1] 
#         recommendations = movies_df.iloc[similar_movies_idx][['title', 'genres']]
#         return recommendations.to_dict('records')
#     else:
#         return {"error": "No movies found. Please check your input."}

# # Define route for recommendations
# @app.route('/recommend', methods=['GET'])
# def recommend():
#     movie_name = request.args.get('movie')
#     if movie_name:
#         results = get_movie_recommendation_nb(movie_name, nb_model, tfidf_vectorizer, movies)
#         return jsonify(results)
#     else:
#         return jsonify({"error": "Please provide a movie name."})

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.svm import SVC
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans
# from scipy.sparse import csr_matrix
# from flask_cors import CORS
# app = Flask(__name__)
# CORS(app)
# # Load Data
# def load_data(movies_path, ratings_path):
#     movies = pd.read_csv(movies_path)
#     ratings = pd.read_csv(ratings_path)
#     return movies, ratings

# # Preprocessing Data
# def preprocess_data(ratings, min_user_votes = 15, min_movie_votes = 31):
#     user_counts = ratings['userId'].value_counts()
#     movie_counts = ratings['movieId'].value_counts()
    
#     ratings = ratings[ratings['userId'].isin(user_counts[user_counts >= min_user_votes].index)]
#     ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_movie_votes].index)]
    
#     final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
#     return final_dataset

# # Train Linear Kernel SVM Model
# def train_svm_model(movies, tfidf_vectorizer):
#     tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])
#     svm_model = SVC(kernel='linear')
#     svm_model.fit(tfidf_matrix, movies['title'])
#     return svm_model, tfidf_matrix

# # Movie Recommendation using Linear Kernel SVM
# def get_movie_recommendation_svm(movie_name, model, vectorizer, movies_df, tfidf_matrix, k = 10):
#     movie_idx = movies_df[movies_df['title'].str.contains(movie_name, case=False)].index
#     if len(movie_idx) > 0:
#         movie_features = vectorizer.transform(movies_df.iloc[movie_idx]['genres'])
#         similarity_scores = cosine_similarity(movie_features, tfidf_matrix)
#         similar_movies_idx = np.argsort(similarity_scores[0])[::-1][1:k+1] 
#         recommendations = movies_df.iloc[similar_movies_idx][['title', 'genres']]
#         return recommendations.to_dict('records')
#     else:
#         return [{"Title": "No movies found. Please check your input."}]

# # Train KNN Model
# def train_knn_model(user_movie_ratings):
#     knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
#     knn_model.fit(user_movie_ratings)
#     return knn_model

# # Movie Recommendation using KNN
# def get_movie_recommendation_knn(movie_id, model, user_movie_ratings, k = 10):
#     distances, indices = model.kneighbors(user_movie_ratings.loc[movie_id].values.reshape(1, -1), n_neighbors = k+1)
#     movie_recommendations = []
#     for i in range(1, len(distances.flatten())):
#         movie_recommendations.append({'Title': user_movie_ratings.index[indices.flatten()[i]], 'Distance': distances.flatten()[i]})
#     return movie_recommendations

# # Train K-Means Model
# def train_kmeans_model(user_movie_ratings, num_clusters):
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(user_movie_ratings)
#     return kmeans, user_movie_ratings

# # Recommendation for the user using K-Means
# def recommend_movies(target_user_id, user_movie_ratings, cluster_labels):
#     target_user_cluster = user_movie_ratings.loc[target_user_id, 'cluster']
#     similar_users = user_movie_ratings[user_movie_ratings['cluster'] == target_user_cluster].index
#     similar_users = similar_users[similar_users != target_user_id]
#     target_user_ratings = user_movie_ratings.loc[target_user_id].values.reshape(1, -1)
#     similar_users_ratings = user_movie_ratings.loc[similar_users]
#     similarities = cosine_similarity(target_user_ratings, similar_users_ratings)[0]
#     top_similar_users = similar_users[np.argsort(similarities)[::-1]][:5]
#     top_movies = user_movie_ratings.loc[top_similar_users].mean().sort_values(ascending=False)
#     recommendations = top_movies.head(10)
#     return recommendations.to_dict()

# # Main Route
# @app.route('/recommend', methods=['GET'])
# def recommend():
#     movie_name = request.args.get('movie')
#     model = request.args.get('model')

#     if movie_name is None or model is None:
#         return jsonify({"error": "Please provide both movie and model parameters."}), 400

#     if model == 'svm':
#         recommendations = get_movie_recommendation_svm(movie_name, svm_model, tfidf_vectorizer, movies, tfidf_matrix)
#     elif model == 'knn':
#         movie_id = int(movie_name)
#         recommendations = get_movie_recommendation_knn(movie_id, knn_model, final_dataset)
#     elif model == 'kmeans':
#         target_user_id = int(movie_name)
#         recommendations = recommend_movies(target_user_id, clustered_ratings, kmeans_model.labels_)
#     else:
#         return jsonify({"error": "Invalid model parameter. Please choose from 'svm', 'knn', or 'kmeans'."}), 400

#     return jsonify(recommendations)

# if __name__ == '__main__':
#     # Load Data
#     movies_path = 'https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/movies.csv'
#     ratings_path = 'https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/ratings.csv'
#     movies, ratings = load_data(movies_path, ratings_path)

#     # Preprocess Data
#     final_dataset = preprocess_data(ratings)

#     # Train Linear Kernel SVM Model
#     tfidf_vectorizer = TfidfVectorizer()
#     svm_model, tfidf_matrix = train_svm_model(movies, tfidf_vectorizer)

#     # Train KNN Model
#     knn_model = train_knn_model(final_dataset)

#     # Train K-Means Model
#     num_clusters = 9  # Based on the previous analysis
#     kmeans_model, clustered_ratings = train_kmeans_model(final_dataset, num_clusters)

#     app.run(debug=True)
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from flask_cors import CORS

app = Flask(__name__)
CORS(app )


# Define global variables for models and data
movies = None
ratings = None
final_dataset = None
tfidf_vectorizer = None
svm_model = None
tfidf_matrix = None
knn_model = None
kmeans_model = None
clustered_ratings = None

# Load Data
def load_data(movies_path, ratings_path):
    global movies, ratings
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

# Preprocessing Data
def preprocess_data(ratings, min_user_votes=15, min_movie_votes=31):
    user_counts = ratings['userId'].value_counts()
    movie_counts = ratings['movieId'].value_counts()
    
    ratings = ratings[ratings['userId'].isin(user_counts[user_counts >= min_user_votes].index)]
    ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_movie_votes].index)]
    
    final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    return final_dataset

# Train Linear Kernel SVM Model
def train_svm_model(movies, tfidf_vectorizer):
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])
    svm_model = SVC(kernel='linear')
    svm_model.fit(tfidf_matrix, movies['title'])
    return svm_model, tfidf_matrix

# Train KNN Model
def train_knn_model(user_movie_ratings):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_movie_ratings)
    return knn_model

# Train K-Means Model
def train_kmeans_model(user_movie_ratings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(user_movie_ratings)
    return kmeans, user_movie_ratings

# Recommendation for the user using K-Means
def recommend_movies(target_user_id, user_movie_ratings, cluster_labels):
    target_user_cluster = user_movie_ratings.loc[target_user_id, 'cluster']
    similar_users = user_movie_ratings[user_movie_ratings['cluster'] == target_user_cluster].index
    similar_users = similar_users[similar_users != target_user_id]
    target_user_ratings = user_movie_ratings.loc[target_user_id].values.reshape(1, -1)
    similar_users_ratings = user_movie_ratings.loc[similar_users]
    similarities = cosine_similarity(target_user_ratings, similar_users_ratings)[0]
    top_similar_users = similar_users[np.argsort(similarities)[::-1]][:5]
    top_movies = user_movie_ratings.loc[top_similar_users].mean().sort_values(ascending=False)
    recommendations = top_movies.head(10)
    return recommendations.to_dict()

# Movie Recommendation using Linear Kernel SVM
def get_movie_recommendation_svm(movie_name, model, vectorizer, movies_df, tfidf_matrix, k = 10):
    movie_idx = movies_df[movies_df['title'].str.contains(movie_name, case=False)].index
    if len(movie_idx) > 0:
        movie_features = vectorizer.transform(movies_df.iloc[movie_idx]['genres'])
        similarity_scores = cosine_similarity(movie_features, tfidf_matrix)
        similar_movies_idx = np.argsort(similarity_scores[0])[::-1][1:k+1] 
        recommendations = movies_df.iloc[similar_movies_idx][['title', 'genres']]
        return recommendations.to_dict('records')
    else:
        return [{"Title": "No movies found. Please check your input."}]

# Movie Recommendation using KNN
def get_movie_recommendation_knn(movie_id, model, user_movie_ratings, k = 10):
    distances, indices = model.kneighbors(user_movie_ratings.loc[movie_id].values.reshape(1, -1), n_neighbors = k+1)
    movie_recommendations = []
    for i in range(1, len(distances.flatten())):
        movie_recommendations.append({'Title': user_movie_ratings.index[indices.flatten()[i]], 'Distance': distances.flatten()[i]})
    return movie_recommendations

# Main Route
@app.route('/recommend', methods=['GET'])
def recommend():
    movie_name = request.args.get('movie')
    model = request.args.get('model')

    if movie_name is None or model is None:
        return jsonify({"error": "Please provide both movie and model parameters."}), 400

    global movies, ratings, final_dataset, tfidf_vectorizer, svm_model, tfidf_matrix, knn_model, kmeans_model, clustered_ratings

    if model == 'svm_linear':
        recommendations = get_movie_recommendation_svm(movie_name, svm_model, tfidf_vectorizer, movies, tfidf_matrix)
    elif model == 'knn':
        movie_id = int(movie_name)
        recommendations = get_movie_recommendation_knn(movie_id, knn_model, final_dataset)
    elif model == 'kmeans':
        target_user_id = int(movie_name)
        recommendations = recommend_movies(target_user_id, clustered_ratings, kmeans_model.labels_)
    else:
        return jsonify({"error": model}), 400

    return jsonify(recommendations)

if __name__ == '__main__':
    # Load Data
    movies_path = 'https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/movies.csv'
    ratings_path = 'https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/ratings.csv'
    load_data(movies_path, ratings_path)

    # Preprocess Data
    final_dataset = preprocess_data(ratings)

    # Train Linear Kernel SVM Model
    tfidf_vectorizer = TfidfVectorizer()
    svm_model, tfidf_matrix = train_svm_model(movies, tfidf_vectorizer)

    # Train KNN Model
    knn_model = train_knn_model(final_dataset)

    # Train K-Means Model
    num_clusters = 9  # Based on the previous analysis
    kmeans_model, clustered_ratings = train_kmeans_model(final_dataset, num_clusters)

    app.run(debug=True)
