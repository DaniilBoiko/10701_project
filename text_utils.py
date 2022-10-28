import string

from tqdm.autonotebook import tqdm

from joblib import Parallel, delayed

from nltk import word_tokenize
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english') + list(string.punctuation))


def tokenize(sent):
    return [i for i in word_tokenize(sent.lower()) if i not in STOP_WORDS]


def tokenize_dataset(dataset):
    return Parallel(n_jobs=-1)(delayed(tokenize)(sent) for sent in tqdm(dataset, leave=False))
