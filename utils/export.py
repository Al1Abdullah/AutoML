import pickle
import os

def save_model(model, path='model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path='model.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None
