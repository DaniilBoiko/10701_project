import string

from tqdm.autonotebook import tqdm

from joblib import Parallel, delayed

from nltk import word_tokenize
from nltk.corpus import stopwords

from utils import TITLE

STOP_WORDS = set(stopwords.words('english') + list(string.punctuation))


def tokenize(features):
    features['title'] = [i for i in word_tokenize(features['title'].lower()) if i not in STOP_WORDS]
    features['title'] = ' '.join(features['title'])
    return features

def tokenize_dataset(dataset):
    return Parallel(n_jobs=-1)(delayed(tokenize)(features) for features in tqdm(dataset, leave=False))

def tokenize_article(article):
    article = list(article)
    article[TITLE] = [i for i in word_tokenize(article[TITLE].lower()) if i not in STOP_WORDS]
    article[TITLE] = ' '.join(article[TITLE])
    return article

def tokenize_news(news):
    keys, values = zip(*news.items())
    values = Parallel(n_jobs=-1)(delayed(tokenize_article)(article) for article in tqdm(values, leave=False))
    for k, v in zip(keys, values):
        news[k] = v
    return news