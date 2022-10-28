import os
import logging
from utils import load_dataset, IMPRESSIONS, TITLE

from tqdm.autonotebook import tqdm

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from text_utils import tokenize_dataset

logging.basicConfig(level=logging.INFO)

logging.info('Loading dataset...')

news_train, behaviors_train = load_dataset(os.path.join(os.environ['MINDPATH'], 'MINDsmall_train'))
news_test, behaviors_test = load_dataset(os.path.join(os.environ['MINDPATH'], 'MINDsmall_dev'))

print('Number of news in train:', len(news_train))
print('Number of behaviours in train:', len(behaviors_train))

print('Number of news in test:', len(news_test))
print('Number of behaviours in test:', len(behaviors_test))


def prepare_data(behaviors, news):
    x, y = [], []

    logging.info('Preparing data...')

    for behavior in tqdm(behaviors, leave=False):
        for news_id, action in behavior[IMPRESSIONS]:
            x.append(news[news_id][TITLE])
            y.append(int(action))

    print('Average target:', np.mean(y))

    SELECT_FIRST = 500_000

    x = x[:SELECT_FIRST]
    y = y[:SELECT_FIRST]

    logging.info('Tokenizing...')
    x = tokenize_dataset(x)
    x = [' '.join(sent) for sent in x]

    return x, y


x_train, y_train = prepare_data(behaviors_train, news_train)
x_test, y_test = prepare_data(behaviors_test, news_test)

logging.info('Vectorizing...')
MAX_FEATURES = 512
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

logging.info('Training...')
clf = xgb.XGBClassifier(n_estimators=1_000, max_depth=6, n_jobs=-1)
clf.fit(x_train, y_train)

logging.info('Evaluating...')
print('Train score:', clf.score(x_train, y_train))
print('Test score:', clf.score(x_test, y_test))

print('Train AUC:', roc_auc_score(y_train, clf.predict_proba(x_train)[:, 1]))
print('Test AUC:', roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))

print('Most important words:')

importance = clf.feature_importances_
indices = np.argsort(importance)[::-1]

TOP_N = 10
for i in range(TOP_N):
    print(vectorizer.get_feature_names()[indices[i]], importance[indices[i]])
