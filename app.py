from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS

app = Flask(__name__)
CORS(app,origins=["http://localhost:3000"])
# Load data
data = pd.read_csv("./random_user_product_mapping.csv")
user_ids = data.iloc[:, 0].values
product_vectors = data.iloc[:, 1:].values
product_names = data.columns[1:]

# Train model
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(product_vectors)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if user_id not in user_ids:
        return jsonify({"error": "User ID not found"}), 404
    
    target_idx = list(user_ids).index(user_id)
    distances, indices = knn.kneighbors([product_vectors[target_idx]])
    neighbor_indices = indices[0][1:]
    neighbor_vector = product_vectors[neighbor_indices].sum(axis=0)
    target_vector = product_vectors[target_idx]
    mask = (target_vector == 0) & (neighbor_vector > 0)
    recommendations = product_names[mask].tolist()[:5]
    print(recommendations)

    return jsonify({
        "user": user_id,
        "recommended_products": recommendations
    })

if __name__ == '__main__':
    print(user_ids)
    app.run(host='0.0.0.0', port=5000)

