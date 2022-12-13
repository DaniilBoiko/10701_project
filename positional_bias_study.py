import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_dataset, IMPRESSIONS

news_train, behaviors_train = load_dataset(os.path.join(os.environ['MINDPATH'], 'MINDsmall_train'))

bins = np.zeros(10)

for behabior in tqdm(behaviors_train):
    for i, (news_id, action) in enumerate(behabior[IMPRESSIONS]):
        if i == 10:
            break

        bins[i] += int(action)

bins /= len(behaviors_train)

plt.figure(figsize=(10, 10))
plt.bar(range(1, 11), bins)
plt.xlabel('Position')
plt.ylabel('Click-through rate')

plt.savefig('ctr.png', dpi=300, bbox_inches='tight')
plt.show()

print(bins)