import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import re
from sentence_transformers import SentenceTransformer

class HybridSearch:
    def __init__(self, documents: List[str], embeddings: np.ndarray, 
                 embedding_weight: float = 0.5):

        self.documents = documents
        self.embeddings = embeddings
        self.embedding_weight = embedding_weight
        
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf.fit_transform(documents)
        
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
        
    def get_tfidf_similarity(self, query: str) -> np.ndarray:
        query_vector = self.tfidf.transform([query])
        # print("Query vector shape ", query_vector.shape)
        # print("TFIDF matrix shape ", self.tfidf_matrix.shape)
        # print(cosine_similarity(query_vector, self.tfidf_matrix))
        return cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
    def get_embedding_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        query_embedding = query_embedding.reshape(1, -1)
        query_embedding = self.normalize_embeddings(query_embedding)
        
        doc_embeddings = self.normalize_embeddings(self.embeddings)
        print(cosine_similarity(query_embedding, doc_embeddings))
        return cosine_similarity(query_embedding, doc_embeddings)[0]
    
    def search(self, query: str, query_embedding: np.ndarray, 
               top_k: int = 5) -> List[Tuple[int, float, str]]:
        
        tfidf_scores = self.get_tfidf_similarity(query)
        embedding_scores = self.get_embedding_similarity(query_embedding)
        
        combined_scores = (
            self.embedding_weight * embedding_scores + 
            (1 - self.embedding_weight) * tfidf_scores
        )
        
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(combined_scores[idx]),
                self.documents[idx]
            ))
            
        return results

if __name__ == "__main__":
    documents = [
        "The weather is nice today",
        "It might rain tomorrow",
        "The rapidly evolving field of artificial intelligence has made tremendous strides in deep developing algorithms that can learn from vast amounts of data, enabling systems to recognize patterns, make predictions, and even make decisions autonomously by mimicking the way the human brain processes information, ultimately transforming industries such as healthcare, finance, and robotics, while continuously advancing through techniques like neural networks that improve over time as they are exposed to more complex datasets.",
        "By leveraging sophisticated computational models that can analyze and interpret large volumes of data, modern systems are now able to autonomously enhance their performance and accuracy over time, utilizing architectures inspired by biological neural networks, allowing for groundbreaking applications in fields such as natural language processing, computer vision, and autonomous systems, where the ability to make intelligent decisions based on learned experiences is becoming increasingly integral to technological progress."
    ]
    
    embeddingModel = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.array(embeddingModel.encode(documents))
    print("Embeddings shape ", embeddings.shape)
    
    searcher = HybridSearch(documents, embeddings, 0.7)
    
    query = "machine learning, deep learning" 
    query_embedding = embeddingModel.encode(query)
    
    results = searcher.search(query, query_embedding, top_k=4)
    
    for idx, score, doc in results:
        print(f"Score: {score:.3f} | Document: {doc}")