import os
import glob
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from parse import parse_clean_battle_simple, process_all_mmlu_data_files
import numpy as np

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
clean_battle_prompts = parse_clean_battle_simple('embedding-similarity/data/clean_battle_data/mini_conv_battle.json')
mmlu_prompts = process_all_mmlu_data_files('embedding-similarity/data/mmlu_data/val')

# Sample datasets
# clean_battle_prompts = ["How do I bake a cake?", "What's the weather like today?", "How tall is Mount Everest?"]
# mmlu_prompts = ["What's the recipe for baking a cake?", "Is it going to rain today?", "How high is the tallest mountain?", "What's the capital of France?"]

# Function to compute embeddings
def compute_embeddings(sentences):
    return model.encode(sentences, convert_to_tensor=True)

# Compute embeddings for both datasets
clean_battle_embeddings = compute_embeddings(clean_battle_prompts)
mmlu_embeddings = compute_embeddings(mmlu_prompts)

# Function to find similar questions
def find_similar_questions(embeddings_A, embeddings_B, threshold=0.3):
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(embeddings_A.cpu().numpy(), embeddings_B.cpu().numpy())
    
    similar_questions = []
    for i, row in enumerate(similarity_matrix):
        similar = [(j, score) for j, score in enumerate(row) if score > threshold]
        similar_questions.append((i, similar))
    
    return similar_questions

# Find similar questions
similar_questions = find_similar_questions(clean_battle_embeddings, mmlu_embeddings)

# Print results
for i, similar in similar_questions:
    print(f"Question from Clean Battle: '{clean_battle_prompts[i]}'")
    if similar:
        print("Similar questions from MMLU:")
        for j, score in similar:
            print(f"  - '{mmlu_prompts[j]}' (similarity: {score:.2f})")
    else:
        print("No similar questions found in MMLU")
    print()