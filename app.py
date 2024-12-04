from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import open_clip
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F

df = pd.read_pickle("image_embeddings.pickle")

model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')

def find_most_similar_image(df, new_embedding):

    embeddings = np.vstack(df["embedding"].values)

    new_embedding = new_embedding.detach().numpy()

    similarities = cosine_similarity(new_embedding, embeddings)

    most_similar_idx = np.argmax(similarities)

    return df.iloc[most_similar_idx]["file_name"]

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
    text = tokenizer([query])
    text_query = F.normalize(model.encode_text(text))

    lam  = weight

    query_embedding = F.normalize(lam * text_query + (1.0 - lam) * image_query)

    return find_most_similar_image(df, query_embedding)



app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():

    file = request.files["file"]

    query = request.form["query"]

    weight = float(request.form["weight"])

    query_type = request.form["type"]

    app.logger.info(f"Type: {query_type}")

    if query_type == "Image":
        app.logger.info("In image query")
        img = image_query(file)
    elif query_type == "Text":
        app.logger.info("In text query")
        img = text_query(query)
    else:
        app.logger.info("In hybrid query")
        img = hybrid_query(file, query, weight)

    return render_template("index.html", img=img)

    # TODO: Make it so that app can route images in dataset correctly
    # move to a static folder?

if __name__ == "__main__":
    app.run(port=3000, host="0.0.0.0", debug=True)