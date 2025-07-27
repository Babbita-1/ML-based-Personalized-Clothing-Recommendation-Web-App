from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np

# Load CSV (use the data you pasted)
data = pd.read_csv('ecommerce_data.csv')

# Prepare text for the embeddings
texts = (
    data['productDisplayName'].fillna('') + ' ' +
    data['masterCategory'].fillna('') + ' ' +
    data['subCategory'].fillna('') + ' ' +
    data['articleType'].fillna('') + ' ' +
    data['baseColour'].fillna('') + ' ' +
    data['season'].fillna('') + ' ' +
    data['usage'].fillna('')
).tolist()

# Load Hugging Face model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Mean Pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element is token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Generate embeddings
embeddings = []
for text in texts:
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings.append(embedding[0].numpy())

embeddings = np.vstack(embeddings)

# Save embeddings
np.save('ecommerce_embeddings.npy', embeddings)

print("✅ Embeddings saved as 'ecommerce_embeddings.npy'")
