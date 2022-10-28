import os
from tqdm.autonotebook import tqdm

NEWS_ID = 0
CATEGORY = 1
SUBCATEGORY = 2
TITLE = 3
ABSTRACT = 4
URL = 5
TITLE_ENTITIES = 6
ABSTRACT_ENTITIES = 7

IMPRESSION_ID = 0
USER_ID = 1
TIME = 2
HISTORY = 3
IMPRESSIONS = 4


def load_news(path):
    with open(path, 'r') as f:
        for line in f:
            splitted_line = line.strip().split('\t')
            news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities = splitted_line
            yield news_id, (category, subcategory, title, abstract, url, title_entities, abstract_entities)


def load_behaviors(path):
    with open(path, 'r') as f:
        for line in f:
            splitted_line = line.strip().split('\t')
            impession_id, user_id, time, history, impressions = splitted_line
            history = history.split(' ')
            impressions = impressions.split(' ')
            impressions = [impression.split('-') for impression in impressions]

            yield impession_id, user_id, time, history, impressions


def load_dataset(path):
    news = dict(tqdm(load_news(os.path.join(path, 'news.tsv')), leave=False))
    behaviors = list(tqdm(load_behaviors(os.path.join(path, 'behaviors.tsv')), leave=False))
    return news, behaviors
