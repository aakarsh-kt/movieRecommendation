from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
import numpy as np
app = Flask(__name__)
CORS(app)

# Load Data
movies = pd.read_csv('https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/movies.csv')
ratings = pd.read_csv('https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/ratings.csv')

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

# Movie Recommendation using Linear Kernel SVM
def get_movie_recommendation_svm(movie_name, model, vectorizer, movies_df, tfidf_matrix, k=10):
    movie_idx = movies_df[movies_df['title'].str.contains(movie_name, case=False)].index
    if len(movie_idx) > 0:
        movie_features = vectorizer.transform(movies_df.iloc[movie_idx]['genres'])
        similarity_scores = cosine_similarity(movie_features, tfidf_matrix)
        similar_movies_idx = np.argsort(similarity_scores[0])[::-1][1:k+1]
        recommendations = movies_df.iloc[similar_movies_idx][['title', 'genres']]
        return recommendations.to_dict('records')
    else:
        return [{"Title": "No movies found. Please check your input."}]

# Train KNN Model
def train_knn_model(user_movie_ratings):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_movie_ratings)
    return knn_model

# Movie Recommendation using KNN
def get_movie_recommendation_knn(movie_id, model, user_movie_ratings, k=10):
    distances, indices = model.kneighbors(user_movie_ratings.loc[movie_id].values.reshape(1, -1), n_neighbors=k+1)
    movie_recommendations = []
    for i in range(1, len(distances.flatten())):
        movie_recommendations.append({'Title': user_movie_ratings.index[indices.flatten()[i]], 'Distance': distances.flatten()[i]})
    return movie_recommendations

# Train K-Means Model
def train_kmeans_model(user_movie_ratings, num_clusters=9):
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

# Train Random Forest Model
def train_random_forest_model(user_movie_ratings, movies):
    le = LabelEncoder()
    movies_subset = movies[movies['movieId'].isin(user_movie_ratings.index)]
    movies_subset['title_encoded'] = le.fit_transform(movies_subset['title'])
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(user_movie_ratings.values, movies_subset['title_encoded'])
    return rf_model

# Movie Recommendation using Random Forest
def get_movie_recommendation_rf(movie_name, model, user_movie_ratings, movies_df, k=10):
    movie_idx = movies_df[movies_df['title'].str.contains(movie_name, case=False)].index
    if len(movie_idx) > 0:
        movie_pred = model.predict(user_movie_ratings.loc[movie_idx].values.reshape(1, -1))
        similar_movies_idx = np.where(model.predict(user_movie_ratings.values) == movie_pred)[0]
        recommendations = movies_df.iloc[similar_movies_idx][:k]
        return recommendations.to_dict('records')
    else:
        return [{"Title": "No movies found. Please check your input."}]

# Train Logistic Regression Model
def train_logistic_regression_model(user_movie_ratings, movies):
    le = LabelEncoder()
    movies_subset = movies[movies['movieId'].isin(user_movie_ratings.index)]
    movies_subset['title_encoded'] = le.fit_transform(movies_subset['title'])
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(user_movie_ratings.values, movies_subset['title_encoded'])
    return lr_model

# Movie Recommendation using Logistic Regression
def get_movie_recommendation_lr(movie_name, model, user_movie_ratings, movies_df, k=10):
    movie_idx = movies_df[movies_df['title'].str.contains(movie_name, case=False)].index
    if len(movie_idx) > 0:
        movie_pred = model.predict(user_movie_ratings.loc[movie_idx].values.reshape(1, -1))
        similar_movies_idx = np.where(model.predict(user_movie_ratings.values) == movie_pred)[0]
        recommendations = movies_df.iloc[similar_movies_idx][:k]
        return recommendations.to_dict('records')
    else:
        return [{"Title": "No movies found. Please check your input."}]

# Train SVD Model
def train_svd_model(user_movie_ratings, num_components=100):
    svd_model = TruncatedSVD(n_components=num_components, random_state=42)
    svd_model.fit(user_movie_ratings)
    return svd_model

# Movie Recommendation using SVD
# Movie Recommendation using SVD
def get_movie_recommendation_svd(movie_name, model, user_movie_ratings, movies_df, k=10):
    movie_idx = movies_df[movies_df['title'].str.contains(movie_name, case=False)].index
    if len(movie_idx) > 0:
        movie_latent = model.transform(user_movie_ratings.loc[movie_idx])
        user_movie_latent = model.transform(user_movie_ratings)
        similarity_scores = cosine_similarity(movie_latent, user_movie_latent)
        similar_users_idx = np.argsort(similarity_scores[0])[::-1]
        recommendations = movies_df.iloc[similar_users_idx][:k]
        return recommendations.to_dict('records')
    else:
        return [{"Title": "No movies found. Please check your input."}]

# Train Naive Bayes Model
def train_naive_bayes_model(user_movie_ratings, ratings):
    nb_model = GaussianNB()

    # Filter ratings to include only movie IDs present in user_movie_ratings
    ratings_subset = ratings[ratings['movieId'].isin(user_movie_ratings.index)]

    # Group ratings_subset by movieId and aggregate user IDs
    user_ids = ratings_subset.groupby('movieId')['userId'].apply(list)

    # Convert user_ids to a list
    user_ids_list = user_ids.tolist()

    # Fill missing user IDs with an empty list
    max_users_per_movie = max(len(ids) for ids in user_ids_list)
    user_ids_array = [ids + [np.nan] * (max_users_per_movie - len(ids)) for ids in user_ids_list]

    # Convert user_ids_array to numpy array
    user_ids_array = np.array(user_ids_array)

    # Flatten user_ids_array to a 1D array
    user_ids_array_flat = user_ids_array.flatten()

    # Fit the Naive Bayes model
    nb_model.fit(user_movie_ratings.values, user_ids_array_flat)

    return nb_model

# Movie Recommendation using Naive Bayes
def get_movie_recommendation_nb(movie_id, model, user_movie_ratings, movies_df, k=10):
    movie_idx = user_movie_ratings.index.get_loc(movie_id)
    user_id = model.predict(user_movie_ratings.iloc[movie_idx].values.reshape(1, -1))
    similar_user_movies_idx = user_movie_ratings.index[user_movie_ratings.iloc[:, :].eq(user_id[0]).any(1)]
    similar_user_movies_idx = similar_user_movies_idx.drop(movie_id)
    recommendations = movies_df.loc[similar_user_movies_idx][:k]
    return recommendations.to_dict('records')

# Main Route
@app.route('/recommend', methods=['GET'])
def recommend():
    movie_name = request.args.get('movie')
    model = request.args.get('model')

    if movie_name is None or model is None:
        return jsonify({"error": "Please provide both movie and model parameters."}), 400

    if model == 'svm_linear':
        recommendations = get_movie_recommendation_svm(movie_name, svm_model, tfidf_vectorizer, movies, tfidf_matrix)
    elif model == 'knn':
        movie_id = movies[movies['title'] == movie_name]['movieId'].values[0]
        recommendations = get_movie_recommendation_knn(movie_id, knn_model, final_dataset)
    elif model == 'kmeans':
        target_user_id = movies[movies['title'] == movie_name]['movieId'].values[0]
        recommendations = recommend_movies(target_user_id, clustered_ratings, kmeans_model.labels_)
    elif model == 'random_forest':
        recommendations = get_movie_recommendation_rf(movie_name, rf_model, final_dataset, movies)
    elif model == 'logistic_regression':
        recommendations = get_movie_recommendation_lr(movie_name, lr_model, final_dataset, movies)
    elif model == 'svd':
        recommendations = get_movie_recommendation_svd(movie_name, svd_model, final_dataset, movies)
    elif model == 'naive_bayes':
        movie_id = movies[movies['title'] == movie_name]['movieId'].values[0]
        recommendations = get_movie_recommendation_nb(movie_id, nb_model, final_dataset, movies)
    else:
        return jsonify({"error": "Invalid model parameter. Please choose from 'svm', 'knn', 'kmeans', 'random_forest', 'logistic_regression', 'svd', or 'naive_bayes'."}), 400

    return jsonify(recommendations)

if __name__ == '__main__':
    # Preprocess Data
    final_dataset = preprocess_data(ratings)

    # Train Linear Kernel SVM Model
    tfidf_vectorizer = TfidfVectorizer()
    svm_model, tfidf_matrix = train_svm_model(movies, tfidf_vectorizer)

    # Train KNN Model
    knn_model = train_knn_model(final_dataset)

    # Train K-Means Model
    kmeans_model, clustered_ratings = train_kmeans_model(final_dataset)

    # Train Random Forest Model
    rf_model = train_random_forest_model(final_dataset, movies)

    # Train Logistic Regression Model
    lr_model = train_logistic_regression_model(final_dataset, movies)

    # Train SVD Model
    svd_model = train_svd_model(final_dataset)

    # Train Naive Bayes Model
    # nb_model = train_naive_bayes_model(final_dataset, ratings)

    app.run(debug=True)