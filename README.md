# Building a news recommender system

This is a project for the course "Introduction to Machine Learning" (10-701) at Carnegie Mellon University.

Before starting set the enivorment variable:
```bash
export MINDPATH="../"
```

## File description

- **train_classifier.py** is the main file for training the classifier
- **train_classifier_gpu.py** is the main file for training the classifier on GPU

Additional experiments:
- **positional_bias_study.py** calculates ctr's for different page positions

Advanced metric calculation:
- **approximate_dates.py** calculates dates of first impression for the news articles, thus approximating their publication dates
- **additional_metrics_computation** contains classes to compute the metrics
- **use_unaware_amc_example** shows how to use the advanced metric calculation classes to compute the metrics

Flask app:
- **app.py** is the main file for the flask app
- **templates/** contains the html templates for the flask app

Utils:
- **utils.py** contains utility functions
- **text_utils.py** contains utility functions for text processing