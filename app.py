from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import open_clip
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import os
from PIL import Image
import pickle

df = pd.read_pickle("image_embeddings.pickle")

model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')

with open("embeddings.pkl", "rb") as f:
    pca_embeddings = pickle.load(f)

with open("img_names.pkl", "rb") as f:
    transform_image_names = pickle.load(f)
 
with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)   
    
def find_most_similar_image(df, new_embedding):

    embeddings = np.vstack(df["embedding"].values)

    new_embedding = new_embedding.detach().numpy()

    similarities = cosine_similarity(new_embedding, embeddings)

    sorted_indicies = np.argsort(similarities.flatten())[::-1]

    top_indices = sorted_indicies[:5]

    top_image_paths = df.iloc[top_indices]["file_name"].values

    top_similarities = similarities.flatten()[sorted_indicies]

    return zip(top_image_paths, top_similarities)
    # most_similar_idx = np.argmax(similarities)

    # return df.iloc[most_similar_idx]["file_name"]

def image_query(file):

    image = preprocess(file).unsqueeze(0)

    query_embedding = F.normalize(model.encode_image(image))

    return find_most_similar_image(df, query_embedding)

def text_query(query):

    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    text = tokenizer([query])
    query_embedding = F.normalize(model.encode_text(text))

    return find_most_similar_image(df, query_embedding)

def hybrid_query(file, query, weight):
    image = preprocess(file).unsqueeze(0)
    image_query = F.normalize(model.encode_image(image))
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text = tokenizer([query])
    text_query = F.normalize(model.encode_text(text))

    lam  = weight

    query_embedding = F.normalize(lam * text_query + (1.0 - lam) * image_query)

    return find_most_similar_image(df, query_embedding)

def nearest_neighbors(query_embedding, embeddings):
    app.logger.info("Query embedding:")
    app.logger.info(query_embedding)
    app.logger.info("Embeddings:")
    app.logger.info(embeddings)
    distances = euclidean_distances(query_embedding, embeddings).flatten()
    nearest_indices = np.argsort(distances)[:5]
    return nearest_indices, distances[nearest_indices]

def pca_query(file):

    query_image = file.convert("L")
    query_image = query_image.resize((224, 224))

    query_embedding = pca.transform([(np.asarray(query_image, dtype=np.float32) / 255.0).flatten()])

    indicies, distances = nearest_neighbors(query_embedding, pca_embeddings)

    return zip(np.array(transform_image_names)[indicies], distances)






app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", imgs=[])

@app.route("/search", methods=["POST"])
def search():

    query_type = request.form["type"]

    if query_type == "Image" or query_type == "Hybrid" or query_type == "PCA":
        
        file = request.files["file"]
        file.save(f"uploads/{file.filename}")

    if query_type == "Text" or query_type == "Hybrid":

        query = request.form["query"]

    if query_type == "Image":
    
        uploaded_image = Image.open(f"uploads/{file.filename}")

        imgs = image_query(uploaded_image)

    elif query_type == "Text":
        imgs = text_query(query)

    elif query_type == "Hybrid":

        weight = float(request.form["weight"])
        uploaded_image = Image.open(f"uploads/{file.filename}")

        imgs = hybrid_query(uploaded_image, query, weight)
    else:

        uploaded_image = Image.open(f"uploads/{file.filename}")

        imgs = pca_query(uploaded_image)

    return render_template("index.html", imgs=imgs)

if __name__ == "__main__":
    app.run(port=3000, host="0.0.0.0", debug=True)