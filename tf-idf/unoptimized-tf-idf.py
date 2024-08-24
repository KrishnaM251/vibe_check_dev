import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load and parse the JSON data
with open('mini_conv_battle.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract text content from conversations
documents = []
for item in data:
    document = ""
    for conversation in item['conversation_a'] + item['conversation_b']:
        document += conversation['content'] + " "
    documents.append(document)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(2, 5), 
    max_features=1000  # Limit to top 1000 features to manage output size
)

# Learn vocabulary and idf, return document-term matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Calculate document similarities
similarity_matrix = cosine_similarity(tfidf_matrix)

# Print useful metrics
print("TF-IDF Analysis Results:")
print("------------------------")
print(f"Number of documents: {len(documents)}")
print(f"Number of unique terms: {len(feature_names)}")
print(f"Shape of TF-IDF matrix: {tfidf_matrix.shape}")

print("\nTop 10 terms by TF-IDF score:")
for i, doc in enumerate(documents):
    print(f"\nDocument {i + 1}:")
    feature_index = tfidf_matrix[i,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
    sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    for idx, score in sorted_scores[:10]:
        print(f"{feature_names[idx]}: {score:.4f}")

print("\nDocument Similarity Matrix:")
print(np.round(similarity_matrix, 2))

print("\nMost similar document pairs:")
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        similarity = similarity_matrix[i][j]
        if similarity > 0.5:  # Adjust this threshold as needed
            print(f"Documents {i+1} and {j+1}: {similarity:.2f}")

print("\nAverage document similarity:", np.mean(similarity_matrix))

# Print n-gram distribution
ngram_lengths = [len(ngram.split()) for ngram in feature_names]
bigrams = ngram_lengths.count(2)
trigrams = ngram_lengths.count(3)
quadragrams = ngram_lengths.count(4)
pentagrams = ngram_lengths.count(5)

print("\nN-gram distribution:")
print(f"Bigrams: {bigrams}")
print(f"Trigrams: {trigrams}")
print(f"Quadragrams: {quadragrams}")
print(f"Pentagrams: {pentagrams}")
'''
Command to run:
python3 unoptimized-tf-idf.py
'''