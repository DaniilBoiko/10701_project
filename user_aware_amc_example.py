import os
import yaml
import logging
import pickle
from datetime import datetime

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm

from additional_metrics_computation import Diversity, Novelty
from utils import (load_dataset, IMPRESSIONS, TITLE, CATEGORY, SUBCATEGORY,
                   HISTORY, one_hot_encode, encode_distribution, TIME)
from text_utils import tokenize_news


def parse_datetime(time):
    return datetime.strptime(time, '%m/%d/%Y %I:%M:%S %p')


def embed_news(news_train, news_test, config):
    feature_names = []
    if config['embedding'] == 'tfidf':
        news_train = tokenize_news(news_train)
        news_test = tokenize_news(news_test)

        news_train_titles = [article[TITLE] for article in news_train.values()]
        news_test_titles = [article[TITLE] for article in news_test.values()]

        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

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

    return news_test, news_train, feature_names


def prepare_data(behaviors, news, is_train=False):
    x, y, group_ids, news_ids, news_categories, current_dates = [], [], [], [], [], []

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

            news_ids.append(news_id)
            news_categories.append(news[news_id][CATEGORY])
            current_dates.append(parse_datetime(behavior[TIME]))

    if config['limit_top']:
        x = x[:config['limit_top_count']]
        y = y[:config['limit_top_count']]
        group_ids = group_ids[:config['limit_top_count']]
        news_ids = news_ids[:config['limit_top_count']]
        news_categories = news_categories[:config['limit_top_count']]
        current_dates = current_dates[:config['limit_top_count']]

    print('Average target:', np.mean(y))

    # logging.info('Tokenizing...')
    # x = tokenize_dataset(x)

    return x, y, group_ids, news_ids, news_categories, current_dates


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


with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

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

x_train, y_train, groups_train, news_ids_train, news_categories_train, current_dates_train = prepare_data(
    behaviors_train, news_train)
x_test, y_test, groups_test, news_ids_test, news_categories_test, current_dates_test = prepare_data(behaviors_test,
                                                                                                    news_test)

x_test = format_data(x_test)

# Load xgboost model
model = xgb.Booster()
model.load_model(os.path.join('model.json'))

y_pred = model.predict(xgb.DMatrix(np.array(x_test)))

# Iterate over groups in test and rerank documents within them:
last_group = None
group_news_ids = []
group_news_categories = []
group_news_scores = []
group_news_dates = []

div = Diversity(available_categories_mapper)

with open('news2dates.pkl', 'rb') as f:
    news2dates = pickle.load(f)

nov = Novelty(news2dates)
diversities = []
novelties = []

for group, news_id, news_category, score, date in zip(tqdm(groups_test), news_ids_test, news_categories_test, y_pred,
                                                      current_dates_test):
    if last_group is None:
        last_group = group

    if group != last_group:
        # Rerank documents within group
        group_news_ids, group_news_categories, group_news_scores = zip(
            *sorted(zip(group_news_ids, group_news_categories, group_news_scores), key=lambda x: x[2], reverse=True)
        )

        # Calculate diversity @ 5
        diversity = div.compute(group_news_categories[:5], rec=False)
        diversities.append(diversity)

        # Calculate novelty @ 5
        novelty = nov.compute(group_news_ids[:5], group_news_dates[0])
        novelties.append(novelty)

        group_news_ids = []
        group_news_categories = []
        group_news_scores = []
        group_news_dates = []

    group_news_ids.append(news_id)
    group_news_categories.append(news_category)
    group_news_scores.append(score)
    group_news_dates.append(date)

    last_group = group

plt.hist(novelties, bins=20)
plt.xlabel('Novelty')
plt.ylabel('Count')
plt.savefig('aware_novelties.png', dpi=300, bbox_inches='tight')
plt.close()

plt.hist(diversities, bins=20)
plt.xlabel('Diversity')
plt.ylabel('Count')
plt.savefig('aware_diversities.png', dpi=300, bbox_inches='tight')
plt.close()