import os
import argparse
import yaml
import logging
from utils import load_dataset, IMPRESSIONS, TITLE, CATEGORY, SUBCATEGORY, HISTORY, one_hot_encode, encode_distribution

from tqdm.autonotebook import tqdm

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from text_utils import tokenize_dataset
from metrics_utils import compute_groupwise_metrics, ndcg_10, ndcg_5


class Model:
    def __init__(self, objective, n_trees, tree_depth):
        self.objective = objective

        if self.objective == 'classification':
            self.model = xgb.XGBClassifier(n_estimators=n_trees, max_depth=tree_depth, n_jobs=-1)
        elif self.objective == 'ranking':
            self.model = xgb.XGBRanker(n_estimators=n_trees, max_depth=tree_depth, n_jobs=-1)

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


logging.basicConfig(level=logging.INFO)

logging.info('Loading dataset...')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml')

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

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


def prepare_data(behaviors, news):
    x, y, group_ids = [], [], []

    logging.info('Preparing data...')

    for group_id, behavior in enumerate(tqdm(behaviors, leave=False)):
        if config['limit_top']:
            if len(x) >= config['limit_top_count']:
                break

        if 'category_history' in config['features']:
            category_history = [news[news_id][CATEGORY] for news_id in behavior[HISTORY]]
            category_history_distribution = encode_distribution(category_history, available_categories_mapper)

        for news_id, action in behavior[IMPRESSIONS]:
            features = {}
            features['title'] = news[news_id][TITLE]
            if 'categories' in config['features']:
                features['category'] = one_hot_encode(news[news_id][CATEGORY], available_categories_mapper)
            if 'subcategories' in config['features']:
                features['subcategory'] = one_hot_encode(news[news_id][SUBCATEGORY], available_subcategories_mapper)

            if 'category_history' in config['features']:
                features['category_history'] = category_history_distribution.copy()

            x.append(features)
            y.append(int(action))
            group_ids.append(group_id)

    if config['limit_top']:
        x = x[:config['limit_top_count']]
        y = y[:config['limit_top_count']]
        group_ids = group_ids[:config['limit_top_count']]

    print('Average target:', np.mean(y))

    logging.info('Tokenizing...')
    x = tokenize_dataset(x)

    return x, y, group_ids


x_train, y_train, groups_train = prepare_data(behaviors_train, news_train)
x_test, y_test, groups_test = prepare_data(behaviors_test, news_test)

logging.info('Vectorizing...')
vectorizer = TfidfVectorizer(max_features=config['max_tfidf_features'])

x_train_title = vectorizer.fit_transform([features['title'] for features in x_train]).todense()
x_test_title = vectorizer.transform([features['title'] for features in x_test]).todense()

feature_names = list(vectorizer.get_feature_names())

all_features_train = [x_train_title]
all_features_test = [x_test_title]

if 'categories' in config['features']:
    x_train_categories = np.vstack([features['category'] for features in x_train])
    x_test_categories = np.vstack([features['category'] for features in x_test])
    feature_names += ['category_' + category for category in available_categories]

    all_features_train.append(x_train_categories)
    all_features_test.append(x_test_categories)

if 'subcategories' in config['features']:
    x_train_subcategories = np.vstack([features['subcategory'] for features in x_train])
    x_test_subcategories = np.vstack([features['subcategory'] for features in x_test])
    feature_names += ['subcategory_' + subcategory for subcategory in available_subcategories]

    all_features_train.append(x_train_subcategories)
    all_features_test.append(x_test_subcategories)

if 'category_history' in config['features']:
    x_train_category_history = np.vstack([features['category_history'] for features in x_train])
    x_test_category_history = np.vstack([features['category_history'] for features in x_test])
    feature_names += ['category_history_' + category for category in available_categories]

    all_features_train.append(x_train_category_history)
    all_features_test.append(x_test_category_history)

x_train = np.hstack(all_features_train)
x_test = np.hstack(all_features_test)

print('Train shape:', x_train.shape)
print('Test shape:', x_test.shape)

logging.info('Training...')
model = Model(config['objective'], config['n_trees'], config['tree_depth'])
model.fit(x_train, y_train, groups_train)

logging.info('Evaluating...')
print('Train AUC:', roc_auc_score(y_train, model.predict_proba(x_train)))
print('Test AUC:', roc_auc_score(y_test, model.predict_proba(x_test)))

print('Most important features:')

importance = model.get_feature_importances()
indices = np.argsort(importance)[::-1]

for i in range(config['print_top_features']):
    print(feature_names[indices[i]], '\t', importance[indices[i]])

print(compute_groupwise_metrics(np.array(y_test), model.predict_proba(x_test), groups_test, [ndcg_5, ndcg_10]))
