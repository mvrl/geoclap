# This script is adapted from https://github.com/khdlr/SoundingEarth/blob/master/lib/evaluation.py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn.functional import normalize


def get_retrevial_metrics(modality1_emb, modality2_emb, normalized=False,k=100):
    if not normalized:
        # Normalize embeddings using L2 normalization
        modality1_emb = normalize(modality1_emb, p=2, dim=1)
        modality2_emb = normalize(modality2_emb, p=2, dim=1)

    # Compute cosine similarity between embeddings
    cos_sim = torch.matmul(modality1_emb, modality2_emb.t()).detach().cpu().numpy() 
    distance_matrix = cos_sim
    K = cos_sim.shape[0]
    # Evaluate Img2Sound
    results = []
    for i in list(range(K)):
        tmpdf = pd.DataFrame(dict(
            k_snd = i,
            dist = distance_matrix[:, i]
        )).set_index('k_snd')

        tmpdf['rank'] = tmpdf.dist.rank(ascending=False)
        res = dict(
            rank=tmpdf.iloc[i]['rank']
        )
        results.append(res)
    df = pd.DataFrame(results)
    topk_str =str(1*k) 
    i2s_metrics = {
        'R@'+topk_str: (df['rank'] < k).mean(),
        'Median Rank': df['rank'].median(),
    }

    return i2s_metrics


def get_retrevial(modality1_emb, modality2_emb, keys,normalized=False,k=100,save_top=5):
    if not normalized:
        # Normalize embeddings using L2 normalization
        modality1_emb = normalize(modality1_emb, p=2, dim=1)
        modality2_emb = normalize(modality2_emb, p=2, dim=1)

    # Compute cosine similarity between embeddings
    cos_sim = torch.matmul(modality1_emb, modality2_emb.t()).detach().cpu().numpy() 
    distance_matrix = cos_sim
    K = cos_sim.shape[0]
    # Evaluate Img2Sound
    results = []
    df_final = pd.DataFrame(columns=['long_key','top_keys'])
    df_final['long_key'] = keys
    results_keys = []
    for i in list(range(K)):
        top_keys = []
        tmpdf = pd.DataFrame(dict(
            k_snd = i,
            dist = distance_matrix[:, i]
        )).set_index('k_snd')
        row_similarity = list(distance_matrix[i, :])
        top_indices = np.array(row_similarity).argsort()[-save_top:][::-1]
        top_keys = [keys[indice] for indice in top_indices]
        results_keys.append(top_keys)

        tmpdf['rank'] = tmpdf.dist.rank(ascending=False)
        res = dict(
            rank=tmpdf.iloc[i]['rank']
        )
        results.append(res)
    df = pd.DataFrame(results)
    topk_str =str(1*k) 
    i2s_metrics = {
        'R@'+topk_str: (df['rank'] < k).mean(),
        'Median Rank': df['rank'].median(),
    }
    df_final['top_keys'] = results_keys
    return i2s_metrics, df_final


if __name__ == '__main__':
    # Suppose we have unnormalized embeddings from two modalities with shape (batch_size, embedding_dim)
    modality1_emb = torch.randn(1000, 512)
    modality2_emb = torch.randn(1000, 512)
    keys = list(range(1000))
    print(get_retrevial(modality1_emb, modality2_emb, keys, normalized=False,k=100))