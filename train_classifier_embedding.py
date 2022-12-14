import os
import argparse
import yaml
import logging
import pickle
from utils import load_dataset, IMPRESSIONS, TITLE, CATEGORY, SUBCATEGORY, HISTORY, one_hot_encode, encode_distribution

from tqdm.autonotebook import tqdm

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from text_utils import tokenize_dataset, tokenize_news
from metrics_utils import compute_groupwise_metrics, ndcg_10, ndcg_5


class Model:
    def __init__(self, objective, n_trees, tree_depth):
        self.objective = objective

        if self.objective == 'classification':
            self.model = xgb.XGBClassifier(n_estimators=n_trees, max_depth=tree_depth, n_jobs=-1, gpu_id=0, tree_method='gpu_hist')
        elif self.objective == 'ranking':
            self.model = xgb.XGBRanker(n_estimators=n_trees, max_depth=tree_depth, n_jobs=-1, gpu_id=0, tree_method='gpu_hist')

    def fit(self, X, y, group_ids):
        if self.objective == 'classification':
            self.model.fit(X, y)
        elif self.objective == 'ranking':
            self.model.fit(X, y, qid=group_ids)

    def predict_proba(self, X):
        if self.objective == 'classification':
            return self.model.predict_proba(X)[:, 1]
        elif self.objective == 'ranking':
            return self.model.predict(X)

    def get_feature_importances(self):
        return self.model.feature_importances_

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def embed_news(news_train, news_test, config):
    feature_names = []
    if config['embedding'] == 'tfidf':
        news_train = tokenize_news(news_train)
        news_test = tokenize_news(news_test)

        news_train_titles = [article[TITLE] for article in news_train.values()]
        news_test_titles = [article[TITLE] for article in news_test.values()]

        vectorizer = TfidfVectorizer(max_features=config['max_tfidf_features'])

        news_train_titles = vectorizer.fit_transform(news_train_titles).todense()
        news_test_titles = vectorizer.transform(news_test_titles).todense()

        feature_names = list(vectorizer.get_feature_names())

        for k, title in zip(news_train.keys(), news_train_titles):
            news_train[k][TITLE] = title

        for k, title in zip(news_test.keys(), news_test_titles):
            news_test[k][TITLE] = title
    elif config['embedding'] == 'use':
        embedding = np.load("embeds/train_embeds.npy")
        for k, embed in zip(news_train.keys(), embedding):
            news_train[k][TITLE] = embed
        embedding = np.load("embeds/dev_embeds.npy")   
        for k, embed in zip(news_test.keys(), embedding):
            news_test[k][TITLE] = embed
        feature_names = [str(i) for i in range(embedding.shape[1])]

    # Save vectorizer
    with open('./vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    return news_test, news_train, feature_names

def prepare_data(behaviors, news, is_train = False):
    x, y, group_ids = [], [], []

    logging.info('Preparing data...')

    # Remove order biasing
    weights = [0.10824706, 0.10777562, 0.07549454, 0.06788137, 0.05707005, 0.05215175, 0.04936769, 0.04294588,
            0.04027012, 0.03807855]

    for group_id, behavior in enumerate(tqdm(behaviors, leave=False)):
        if config['limit_top']:
            if len(x) >= config['limit_top_count']:
                break

        behavior = list(behavior)
        if behavior[HISTORY][0] == '':
            behavior[HISTORY] = []

        if 'category_history' in config['features']:
            category_history = [news[news_id][CATEGORY] for news_id in behavior[HISTORY]]
            category_history_distribution = encode_distribution(category_history, available_categories_mapper)

        for impression_idx, impression in enumerate(behavior[IMPRESSIONS]):
            news_id, action = impression
            features = {}
            features['title'] = news[news_id][TITLE]
            if 'categories' in config['features']:
                features['category'] = one_hot_encode(news[news_id][CATEGORY], available_categories_mapper)
            if 'subcategories' in config['features']:
                features['subcategory'] = one_hot_encode(news[news_id][SUBCATEGORY], available_subcategories_mapper)

            if 'category_history' in config['features']:
                features['category_history'] = category_history_distribution.copy()

            if 'click_history' in config['features']:
                news_embed = np.zeros_like(features['title'])
                for click_news_id in behavior[HISTORY]:
                    if len(click_news_id) != 0:
                        news_embed += news[click_news_id][TITLE]
                news_embed = (news_embed / len(behavior[HISTORY]) if len(behavior[HISTORY]) != 0 else news_embed)
                features['click_history'] = news_embed

            x.append(features)
            if impression_idx < 10:
                y.append(int(action) / (weights[impression_idx] if is_train else 1))
            else:
                y.append(int(action) / (weights[9] if is_train else 1))
            group_ids.append(group_id)

    if config['limit_top']:
        x = x[:config['limit_top_count']]
        y = y[:config['limit_top_count']]
        group_ids = group_ids[:config['limit_top_count']]

    print('Average target:', np.mean(y))

    # logging.info('Tokenizing...')
    # x = tokenize_dataset(x)

    return x, y, group_ids

def format_data(x):
    all_features = [np.vstack([features['title'] for features in x])]

    if 'categories' in config['features']:
        x_categories = np.vstack([features['category'] for features in x])

        all_features.append(x_categories)

    if 'subcategories' in config['features']:
        x_subcategories = np.vstack([features['subcategory'] for features in x])

        all_features.append(x_subcategories)

    if 'category_history' in config['features']:
        x_category_history = np.vstack([features['category_history'] for features in x])

        all_features.append(x_category_history)

    if 'click_history' in config['features']:
        all_features.append(np.vstack([features['click_history'] for features in x]))

    return np.hstack(all_features)

def get_feature_names(feature_names):
    if 'categories' in config['features']:
        feature_names += ['category_' + category for category in available_categories]

    if 'subcategories' in config['features']:
        feature_names += ['subcategory_' + subcategory for subcategory in available_subcategories]

    if 'category_history' in config['features']:
        feature_names += ['category_history_' + category for category in available_categories]

    if 'click_history' in config['features']:
        feature_names += feature_names[:config['max_tfidf_features']]

    return feature_names

config = get_config()

logging.basicConfig(level=logging.INFO)

logging.info('Loading dataset...')

news_train, behaviors_train = load_dataset(os.path.join(os.environ['MINDPATH'], 'MINDsmall_train'))
news_test, behaviors_test = load_dataset(os.path.join(os.environ['MINDPATH'], 'MINDsmall_dev'))

print('Number of news in train:', len(news_train))
print('Number of behaviours in train:', len(behaviors_train))

print('Number of news in test:', len(news_test))
print('Number of behaviours in test:', len(behaviors_test))

available_categories = list(set(news_train[news_id][CATEGORY] for news_id in news_train))
available_subcategories = list(set(news_train[news_id][SUBCATEGORY] for news_id in news_train))

available_categories_mapper = {category: i for i, category in enumerate(available_categories)}
available_subcategories_mapper = {subcategory: i for i, subcategory in enumerate(available_subcategories)}

# Convert news titles into embeddings
logging.info("Tokenizing...")
news_train = tokenize_news(news_train)
news_test = tokenize_news(news_test)

# Convert titles into embeddings
logging.info("Calculating embedding...")
news_test, news_train, feature_names = embed_news(news_train, news_test, config)

x_train, y_train, groups_train = prepare_data(behaviors_train, news_train)
x_test, y_test, groups_test = prepare_data(behaviors_test, news_test)

# Delete reference to prevent excess memory use
del behaviors_train, news_train, behaviors_test, news_test, available_categories_mapper, available_subcategories_mapper

# Format and store test data (memory)
x_test = format_data(x_test)
np.savez("test_data.npz", x_test = x_test, y_test = y_test)
del x_test, y_test

# Get training data and train
x_train = format_data(x_train)
feature_names = get_feature_names(feature_names)
del available_categories, available_subcategories

print('Train shape:', x_train.shape)

logging.info('Training...')
model = Model(config['objective'], config['n_trees'], config['tree_depth'])
model.fit(x_train, y_train, groups_train)

logging.info('Evaluating...')
print('Train AUC:', roc_auc_score(y_train, model.predict_proba(x_train)))

# Get test data and eval
del x_train, y_train

data = np.load("test_data.npz")
x_test = data['x_test']
y_test = data['y_test']
del data

print('Test shape:', x_test.shape)

print('Test AUC:', roc_auc_score(y_test, model.predict_proba(x_test)))

# print('Most important features:')

importance = model.get_feature_importances()
indices = np.argsort(importance)[::-1]

# for i in range(config['print_top_features']):
#     print(feature_names[indices[i]], '\t', importance[indices[i]])
print(compute_groupwise_metrics(np.array(y_test), model.predict_proba(x_test), groups_test, [ndcg_5, ndcg_10]))
