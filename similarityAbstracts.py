from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity

gteModel = SentenceTransformer("thenlper/gte-small")

def generateEmbedding(text):
    
    if type(text) != str:
        return gteModel.encode(" ") # if abstract is missing, need to remove from csv file
    
    embeddings = gteModel.encode(text)
    return embeddings


# if embeddingsMatrix.csv exists, load it else create it
if os.path.exists("embeddingsMatrix.csv"):
    embeddingsMatrix = pd.read_csv("embeddingsMatrix.csv").values

else:
    df = pd.read_csv("combinedData.csv")
    df['embeddings'] = df['Abstract'].apply(generateEmbedding)

    embeddingsMatrix = np.vstack(df['embeddings'].values)
    print(embeddingsMatrix.shape)
    embeddingMatrixCSV = pd.DataFrame(embeddingsMatrix)
    embeddingMatrixCSV.to_csv("embeddingsMatrix.csv", index=False)


def getTopAbstracts(query, top_n=100):
    query_embedding = generateEmbedding(query).reshape(1, -1)

    cosine_similarities = cosine_similarity(query_embedding, embeddingsMatrix).flatten()

    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]

    df = pd.read_csv("combinedData.csv")

    top_papers = []
    for idx in top_indices:
        author = df.iloc[idx]["Author"]
        abstract = df.iloc[idx]["Abstract"]
        score = cosine_similarities[idx]
        top_papers.append({"Author": author, "Abstract": abstract, "Score": score})

    top_papers_df = pd.DataFrame(top_papers)
    print(top_papers_df)

    return top_papers_df

def getTopUniqueProfessors(query, top_n=10):
    # Generate the query embedding
    query_embedding = generateEmbedding(query).reshape(1, -1)

    # Calculate cosine similarity between the query and all paper embeddings
    cosine_similarities = cosine_similarity(query_embedding, embeddingsMatrix).flatten()

    # Load the original data to fetch details
    df = pd.read_csv("combinedData.csv")

    # Add cosine similarity scores to the DataFrame for easy sorting
    df['Score'] = cosine_similarities

    # Sort by similarity score in descending order
    df_sorted = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    # Collect top unique professors based on highest similarity score per professor
    top_papers = []
    unique_professors = set()

    for idx, row in df_sorted.iterrows():
        professor = row["Author"]
        if professor not in unique_professors:
            top_papers.append({
                "Author": professor,
                "Abstract": row["Abstract"],
                "Score": row["Score"]
            })
            unique_professors.add(professor)
        
        # Stop if we have collected 'top_n' unique professors
        if len(top_papers) >= top_n:
            break

    # Convert top papers to DataFrame for easy viewing and sorting
    top_papers_df = pd.DataFrame(top_papers)

    return top_papers_df

query = "machine learning visualizations, human computer interaction, graph mining"
# topPapers = getTopAbstracts(query)
# print(topPapers)

topProfessors = getTopUniqueProfessors(query)
print(topProfessors)