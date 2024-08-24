import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample datasets
dataset_A = ["How do I bake a cake?", "What's the weather like today?", "How tall is Mount Everest?"]
dataset_B = ["What's the recipe for baking a cake?", "Is it going to rain today?", "How high is the tallest mountain?", "What's the capital of France?"]

# Function to compute embeddings
def compute_embeddings(sentences):
    return model.encode(sentences, convert_to_tensor=True)

# Compute embeddings for both datasets
embeddings_A = compute_embeddings(dataset_A)
embeddings_B = compute_embeddings(dataset_B)

# Function to find similar questions
def find_similar_questions(embeddings_A, embeddings_B, threshold=0.7):
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(embeddings_A.cpu().numpy(), embeddings_B.cpu().numpy())
    
    similar_questions = []
    for i, row in enumerate(similarity_matrix):
        similar = [(j, score) for j, score in enumerate(row) if score > threshold]
        similar_questions.append((i, similar))
    
    return similar_questions

# Find similar questions
similar_questions = find_similar_questions(embeddings_A, embeddings_B)

# Print results
for i, similar in similar_questions:
    print(f"Question from A: '{dataset_A[i]}'")
    if similar:
        print("Similar questions from B:")
        for j, score in similar:
            print(f"  - '{dataset_B[j]}' (similarity: {score:.2f})")
    else:
        print("No similar questions found in B")
    print()