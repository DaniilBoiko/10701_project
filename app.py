import os
import random
import logging
import yaml
import pickle

from tqdm import tqdm
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

from flask import Flask
from flask import render_template
app = Flask(__name__)

from utils import load_dataset, IMPRESSIONS, TITLE, CATEGORY, SUBCATEGORY, HISTORY, one_hot_encode, encode_distribution
from text_utils import tokenize_dataset

news_train, behaviors_train = load_dataset(os.path.join(os.environ['MINDPATH'], 'MINDsmall_train'))
news_test, behaviors_test = load_dataset(os.path.join(os.environ['MINDPATH'], 'MINDsmall_dev'))

available_categories = list(set(news_train[news_id][CATEGORY] for news_id in news_train))
available_subcategories = list(set(news_train[news_id][SUBCATEGORY] for news_id in news_train))

available_categories_mapper = {category: i for i, category in enumerate(available_categories)}
available_subcategories_mapper = {subcategory: i for i, subcategory in enumerate(available_subcategories)}


def prepare_data(behavior, news, config):
    x, y = [], []

    logging.info('Preparing data...')

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

    logging.info('Tokenizing...')
    x = tokenize_dataset(x)

    return x, y


@app.route('/index/')
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/')
def predict():
    news_ids = random.sample(list(news_test.keys()), 10)
    news = [news_test[news_id] for news_id in news_ids]

    behavior = [
        0, 0, 0, [], [[i, 0] for i in range(10)]
    ]

    MODEL_PATH = 'pr/10701_project/baseline'

    with open(os.path.join(MODEL_PATH, 'config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    x_test, _ = prepare_data(behavior, news, config)

    with open(os.path.join(MODEL_PATH, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    feature_names = list(vectorizer.get_feature_names())

    x_test_title = vectorizer.transform([features['title'] for features in x_test]).todense()

    all_features_test = [x_test_title]

    if 'categories' in config['features']:
        x_test_categories = np.vstack([features['category'] for features in x_test])
        feature_names += ['category_' + category for category in available_categories]

        all_features_test.append(x_test_categories)

    if 'subcategories' in config['features']:
        x_test_subcategories = np.vstack([features['subcategory'] for features in x_test])
        feature_names += ['subcategory_' + subcategory for subcategory in available_subcategories]

        all_features_test.append(x_test_subcategories)

    if 'category_history' in config['features']:
        x_test_category_history = np.vstack([features['category_history'] for features in x_test])
        feature_names += ['category_history_' + category for category in available_categories]

        all_features_test.append(x_test_category_history)

    x_test = np.hstack(all_features_test)

    # Load xgboost model
    model = xgb.Booster()
    model.load_model(os.path.join(MODEL_PATH, 'model.json'))

    y_pred = model.predict(xgb.DMatrix(x_test))

    # Sort news by predicted score
    news = [(news, score) for score, news in sorted(zip(y_pred, news), reverse=True)]

    return render_template('predict.html', news=news)
