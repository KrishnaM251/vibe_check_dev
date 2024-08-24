import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import islice

def load_documents(file_path, chunk_size=10000):
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = list(islice(file, chunk_size))
            if not chunk:
                break
            for line in chunk:
                try:
                    item = json.loads(line)
                    document = ""
                    for conversation in item['conversation_a'] + item['conversation_b']:
                        document += conversation['content'] + " "
                    yield document
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON: {line}")

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(2, 5),
    max_features=1000
)

# Process documents in chunks
chunk_size = 10000
document_count = 0
tfidf_sum = None

for i, chunk in enumerate(iter(lambda: list(islice(load_documents('mini_conv_battle.json'), chunk_size)), [])):
    print(f"Processing chunk {i+1}")
    tfidf_matrix = vectorizer.fit_transform(chunk)
    document_count += tfidf_matrix.shape[0]
    
    if tfidf_sum is None:
        tfidf_sum = tfidf_matrix.sum(axis=0)
    else:
        tfidf_sum += tfidf_matrix.sum(axis=0)

    # Free up memory
    del tfidf_matrix

# Get feature names (n-grams)
feature_names = vectorizer.get_feature_names_out()

# Print useful metrics
print("\nTF-IDF Analysis Results (with n-grams):")
print("---------------------------------------")
print(f"Number of documents processed: {document_count}")
print(f"Number of unique n-grams: {len(feature_names)}")

print("\nTop 10 n-grams by overall TF-IDF score:")
feature_index = np.argsort(tfidf_sum.A1)[::-1]
for idx in feature_index[:10]:
    print(f"{feature_names[idx]}: {tfidf_sum[0, idx]:.4f}")

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
python3 optimized-tf-idf.py
'''