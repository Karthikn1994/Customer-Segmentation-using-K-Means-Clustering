#!/usr/bin/env python3
"""kmeans_clustering.py
Standalone script to run K-Means clustering for customer segmentation.
"""

import argparse, os, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(path, id_col=None):
    df = pd.read_csv(path)
    if id_col and id_col in df.columns:
        ids = df[id_col]
        df = df.drop(columns=[id_col])
    else:
        ids = None
    return df, ids

def preprocess(df):
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.fillna(numeric.median())
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric)
    return X, numeric.columns.tolist(), scaler, numeric

def elbow_method(X, max_k=10, output_path=None):
    wcss = []
    for k in range(1, max_k+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        wcss.append(km.inertia_)
    if output_path:
        plt.figure(figsize=(6,4))
        plt.plot(range(1, max_k+1), wcss, marker='o')
        plt.xlabel('Number of clusters k')
        plt.ylabel('WCSS (inertia)')
        plt.title('Elbow Method for optimal k')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    return wcss

def choose_k_by_elbow(wcss):
    drops = np.diff(wcss)
    if len(drops)==0:
        return 1
    rel = drops / np.array(wcss[:-1])
    k = int(np.argmax(-rel)) + 2
    if k < 2:
        k = 2
    return k

def run_kmeans(X, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    return km, labels, centers

def pca_scatter(X, labels, centers, output_path, title='Clusters (PCA)'):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    if centers is not None:
        centers2 = pca.transform(centers)
    else:
        centers2 = None
    plt.figure(figsize=(7,5))
    for lab in np.unique(labels):
        sel = X2[labels==lab]
        plt.scatter(sel[:,0], sel[:,1], label=f'Cluster {lab}', alpha=0.6, s=20)
    if centers2 is not None:
        plt.scatter(centers2[:,0], centers2[:,1], marker='X', s=100, edgecolor='k')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def cluster_profile(df_numeric, labels, output_path_json):
    dfc = df_numeric.copy()
    dfc['cluster'] = labels
    summary = dfc.groupby('cluster').agg(['count','mean','median','std']).to_dict()
    sizes = dfc['cluster'].value_counts().sort_index().to_dict()
    out = {'sizes': sizes, 'summary': summary}
    with open(output_path_json, 'w') as f:
        json.dump(out, f, indent=2, default=int)
    return dfc.groupby('cluster').mean(), sizes

def main():
    parser = argparse.ArgumentParser(description='K-Means clustering script.')
    parser.add_argument('--input','-i', required=True)
    parser.add_argument('--id_col', default=None)
    parser.add_argument('--output_dir','-o', default='outputs')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--max_k', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df, ids = load_data(args.input, id_col=args.id_col)
    X, feature_names, scaler, numeric_df = preprocess(df)

    wcss = elbow_method(X, max_k=args.max_k, output_path=os.path.join(args.output_dir,'elbow_wcss.png'))
    chosen_k = args.k if args.k else choose_k_by_elbow(wcss)
    print(f'Chosen k = {chosen_k}')

    km, labels, centers = run_kmeans(X, chosen_k)
    pca_scatter(X, labels, centers, os.path.join(args.output_dir,'clusters_pca.png'))

    out_df = numeric_df.copy()
    out_df['cluster'] = labels
    if ids is not None:
        out_df.insert(0, 'CustomerID', ids.reset_index(drop=True))

    out_csv = os.path.join(args.output_dir, 'clustered_data.csv')
    out_df.to_csv(out_csv, index=False)

    np.save(os.path.join(args.output_dir,'kmeans_centers.npy'), centers)

    mean_profiles, sizes = cluster_profile(numeric_df, labels, os.path.join(args.output_dir,'cluster_profiles.json'))

    report = {
        'chosen_k': int(chosen_k),
        'sizes': sizes,
        'features': feature_names
    }
    with open(os.path.join(args.output_dir,'summary.json'),'w') as f:
        json.dump(report, f, indent=2)

    print('Outputs saved to', args.output_dir)
    print('Files: ', os.listdir(args.output_dir))

if __name__=='__main__':
    main()
