import json
import numpy as np
import pandas as pd
import re

save_path = "./output/topic"
import openai
import os
# replace "..." with your OpenAI key.
# os.environ["OPENAI_API_KEY"] = "..."
# openai.api_key = os.getenv("OPENAI_API_KEY")

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.backend import OpenAIBackend



input_file_path = "data/test_prompt_all.json"

df = pd.read_json(input_file_path)
breakpoint()

english_df = df[df['language'] == 'English'].copy()
english_df['Prompt'] = english_df.apply(lambda x: ' '.join([i['content'] for i in x['conversation_a'] if i['role'] == 'user']), axis=1)
english_df = english_df.drop_duplicates(subset='Prompt')
english_df = english_df[english_df['Prompt'].str.len() < 8000]
doc = english_df['Prompt']

client = openai.OpenAI()
embedding_model = OpenAIBackend(client, "text-embedding-3-large", batch_size=1000)
embeddings = embedding_model.embed(doc, verbose=True)

# save embeddings
np.save(f"{save_path}/embeddings.npy", embeddings)

# load saved embeddings
from huggingface_hub import hf_hub_download
file_path = hf_hub_download(
    repo_id="lmarena-ai/arena-explorer-preference-100k",
    filename="data/embeddings.npy",
    repo_type="dataset"
)

embeddings = np.load(file_path)


client = openai.OpenAI()
embedding_model = OpenAIBackend(client, "text-embedding-3-large", batch_size=1000)
umap_model = UMAP(n_neighbors=20, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 3))

topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,

        top_n_words=10,
        verbose=True,
        calculate_probabilities=True
)

topics, probs = topic_model.fit_transform(doc, embeddings=embeddings)

# number of clusters
print(len(topic_model.get_topic_info()))
print(topic_model.get_topic_info().head())