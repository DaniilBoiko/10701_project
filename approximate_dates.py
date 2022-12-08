import os
import logging
import pickle as pkl
from datetime import datetime

from tqdm import tqdm

from utils import load_dataset, IMPRESSIONS, TIME

logging.basicConfig(level=logging.INFO)

logging.info('Loading dataset...')
news_train, behaviors_train = load_dataset(os.path.join(os.environ['MINDPATH'], 'MINDsmall_train'))
news_test, behaviors_test = load_dataset(os.path.join(os.environ['MINDPATH'], 'MINDsmall_dev'))

news2dates = {}


def parse_datetime(time):
    return datetime.strptime(time, '%m/%d/%Y %I:%M:%S %p')


for behavior in tqdm(behaviors_train):
    for news_id, action in behavior[IMPRESSIONS]:
        if news_id not in news2dates:
            news2dates[news_id] = parse_datetime(behavior[TIME])
        else:
            news2dates[news_id] = min(news2dates[news_id], parse_datetime(behavior[TIME]))

for behavior in tqdm(behaviors_test):
    for news_id, action in behavior[IMPRESSIONS]:
        if news_id not in news2dates:
            news2dates[news_id] = parse_datetime(behavior[TIME])
        else:
            news2dates[news_id] = min(news2dates[news_id], parse_datetime(behavior[TIME]))

# Save the dates
logging.log(logging.INFO, 'Saving dates...')
with open('news2dates.pkl', 'wb') as f:
    pkl.dump(news2dates, f)
