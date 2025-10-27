# context_ai.py - Contextual AI for Adaptive Search and Learning

import pickle
from sklearn.linear_model import RidgeCV
import numpy as np
class ContextHistory:
    """Store and use historical optimization data"""
    def __init__(self, filename='context_history.pkl'):
        self.filename = filename
        try:
            with open(filename, 'rb') as f:
                self.history = pickle.load(f)
        except:
            self.history = []

    def store(self, metrics):
        self.history.append(metrics)
        with open(self.filename, 'wb') as f:
            pickle.dump(self.history, f)

    def suggest_region(self, search_space):
        """Lightweight predictor for promising subspaces."""
        if len(self.history) < 2:
            return None  # Not enough data
        # Build a regressor over past param-fitness pairs
        X = np.array([m['params'] for m in self.history])
        y = np.array([m['fitness'] for m in self.history])
        model = RidgeCV(alphas=[1.0, 10.0, 100.0])
        model.fit(X, y)
        preds = model.predict(search_space)
        return search_space[np.argmin(preds)]