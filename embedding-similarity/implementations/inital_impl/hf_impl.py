import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Example datasets
dataset_A = ["What is the capital of France?", "How tall is Mount Everest?"]
dataset_B = ["What's the capital city of France?", "What's the height of Everest?", "Who wrote Hamlet?"]

# Generate embeddings
embeddings_A = get_embeddings(dataset_A)
embeddings_B = get_embeddings(dataset_B)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(embeddings_A, embeddings_B)

# Set similarity threshold
threshold = 0.8

# Find similar questions
for i, question_A in enumerate(dataset_A):
    similar_questions = []
    for j, question_B in enumerate(dataset_B):
        if similarity_matrix[i][j] > threshold:
            similar_questions.append((question_B, similarity_matrix[i][j]))
    
    print(f"Similar questions for '{question_A}':")
    for q, score in similar_questions:
        print(f"- '{q}' (similarity: {score:.2f})")
    print()