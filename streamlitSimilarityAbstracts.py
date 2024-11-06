import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

gteModel = SentenceTransformer("thenlper/gte-small")

@st.cache_data
def generateEmbedding(text):
    if type(text) != str:
        return gteModel.encode(" ")  # If abstract is missing, use placeholder
    return gteModel.encode(text)

@st.cache_data
def load_embeddings():
    if os.path.exists("embeddingsMatrix.csv"):
        return pd.read_csv("embeddingsMatrix.csv").values
    else:
        df = pd.read_csv("combinedData.csv")
        df['embeddings'] = df['Abstract'].apply(generateEmbedding)
        embeddingsMatrix = np.vstack(df['embeddings'].values)
        embeddingMatrixCSV = pd.DataFrame(embeddingsMatrix)
        embeddingMatrixCSV.to_csv("embeddingsMatrix.csv", index=False)
        return embeddingsMatrix

@st.cache_data
def load_combined_data():
    return pd.read_csv("combinedData.csv")

# def getTopAbstracts(query, top_n=100):
#     embeddingsMatrix = load_embeddings()
#     query_embedding = generateEmbedding(query).reshape(1, -1)
#     cosine_similarities = cosine_similarity(query_embedding, embeddingsMatrix).flatten()
#     top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
#     df = load_combined_data()
#     top_papers = [{"Author": df.iloc[idx]["Author"], "Abstract": df.iloc[idx]["Abstract"], "Score": cosine_similarities[idx]} for idx in top_indices]
#     return pd.DataFrame(top_papers)

def getTopUniqueProfessors(query, top_n=10):
    embeddingsMatrix = load_embeddings()
    df = load_combined_data()
    query_embedding = generateEmbedding(query).reshape(1, -1)
    cosine_similarities = cosine_similarity(query_embedding, embeddingsMatrix).flatten()
    df['Score'] = cosine_similarities
    df_sorted = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    top_papers = []
    unique_professors = set()
    for _, row in df_sorted.iterrows():
        professor = row["Author"]
        if professor not in unique_professors:
            top_papers.append({"Author": professor, "Abstract": row["Abstract"], "Score": row["Score"]})
            unique_professors.add(professor)
        if len(top_papers) >= top_n:
            break
    return pd.DataFrame(top_papers)

st.title("Academic Paper Similarity Finder")

query = st.text_input("Enter your research query:", value="machine learning visualizations, human computer interaction, graph mining")

if st.button("Find Top Unique Professors"):
    st.write("Top Unique Professors:")
    topProfessors = getTopUniqueProfessors(query)
    st.write(topProfessors)
