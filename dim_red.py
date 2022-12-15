import os
import yaml
import argparse

import numpy as np
from sklearn.decomposition import PCA

from utils import load_dataset, CATEGORY, SUBCATEGORY

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml')

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

news_train, behaviors_train = load_dataset(os.path.join(os.environ['MINDPATH'], 'MINDsmall_train'))

available_categories = list(set(news_train[news_id][CATEGORY] for news_id in news_train))
available_subcategories = list(set(news_train[news_id][SUBCATEGORY] for news_id in news_train))

available_categories_mapper = {category: i for i, category in enumerate(available_categories)}
available_subcategories_mapper = {subcategory: i for i, subcategory in enumerate(available_subcategories)}

x_train_title = np.load('embeds/train_embeds.npy')
categories = [news[CATEGORY] for _, news in news_train.items()]

subsample = np.random.choice(len(x_train_title), 10000, replace=False)
x_train_title = x_train_title[subsample]
categories = np.array(categories)[subsample]

# Plot PCA plot for titles
pca = PCA(n_components=2)
pca.fit(x_train_title)
x_train_title_pca = pca.transform(x_train_title)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for category in available_categories:
    x = x_train_title_pca[np.array(categories) == category]
    plt.scatter(x[:, 0], x[:, 1], label=category)

plt.xlabel('PC1 ({:.2f}%)'.format(pca.explained_variance_ratio_[0] * 100))
plt.ylabel('PC2 ({:.2f}%)'.format(pca.explained_variance_ratio_[1] * 100))
plt.savefig('pca.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot TSNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
x_train_title_tsne = tsne.fit_transform(x_train_title)
categories = np.array(categories)

plt.figure(figsize=(10, 10))
for category in available_categories:
    x = x_train_title_tsne[np.array(categories) == category]
    plt.scatter(x[:, 0], x[:, 1], label=category)

plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.savefig('tsne.png', dpi=300, bbox_inches='tight')
plt.close()
