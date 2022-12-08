import numpy as np
from utils import CATEGORY, encode_distribution


class Diversity:
    def __init__(self, category_mapper, k=5):
        self.k = k
        self.category_mapper = category_mapper

    def compute(self, recommendations):
        categories = [self.category_mapper[recommendation[CATEGORY]] for recommendation in recommendations]
        distribution = encode_distribution(categories, self.category_mapper)
        distribution = np.array(distribution)

        # Compute entropy
        entropy = -np.sum(distribution * np.log(distribution + 1e-10))

        return entropy


class Novelty:
    def __init__(self, news_date_mapper):
        self.news_date_mapper = news_date_mapper

    def compute(self, recommendation_ids, current_date):
        novelty = 0
        for recommendation_id in recommendation_ids:
            novelty += max((current_date - self.news_date_mapper[recommendation_id]).days, 0)

        novelty /= len(recommendation_ids)

        return novelty
