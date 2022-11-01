import string

from tqdm.autonotebook import tqdm

from joblib import Parallel, delayed

from nltk import word_tokenize
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english') + list(string.punctuation))


def tokenize(features):
    features['title'] = [i for i in word_tokenize(features['title'].lower()) if i not in STOP_WORDS]
    features['title'] = ' '.join(features['title'])
    return features


def tokenize_dataset(dataset):
    return Parallel(n_jobs=-1)(delayed(tokenize)(features) for features in tqdm(dataset, leave=False))
