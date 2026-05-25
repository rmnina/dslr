from utils import load
import numpy as np
import random
from OneVsRestClassifier import OneVsRestClassifier as OVR
import pandas as pd

THRESHOLD = 0.5
SEED = 21
EVAL_DATASET_SIZE = 0.2

def split_dataset(X: np.ndarray, y: np.ndarray, eval_size: float) -> tuple[np.ndarray]:
    if not 0 < EVAL_DATASET_SIZE < 1:
        raise Exception("split_dataset(): EVAL_DATASET_SIZE must be between 0 and 1.")
    m = X.shape[0]
    random.seed(SEED)
    eval_idx = random.sample(range(m), int(m * eval_size))
    train_idx = list(set((range(m))) - set(eval_idx))

    X_train = X.iloc[train_idx, :]
    y_train = y.iloc[train_idx]
    X_eval = X.iloc[eval_idx, :]
    y_eval = y.iloc[eval_idx]

    return X_train, X_eval, y_train, y_eval
    
    
def ft_normalize(X: np.ndarray, min_train: float, max_train:float) -> np.ndarray:
    return (X - min_train) / ((max_train - min_train) + 1e-20) 


def test(X_eval, y_eval, W, b) -> None:

    class_mapping = {"Gryffindor": 0, "Hufflepuff": 1, "Ravenclaw": 2, "Slytherin": 3}
    y_eval_mapped = pd.Series(y_eval).map(class_mapping).to_numpy()
    preds = OVR.predict(X_eval, W, b)
    success = 0

    for i in range(len(X_eval)):
        if preds[i] == y_eval_mapped[i]:
            success += 1
    print(f"success rate: {(success / len(X_eval) * 100):.2f}%")


def normalize_features(X_train, X_eval) -> tuple[np.ndarray]:
    min_train = X_train.min(axis=0)
    max_train = X_train.max(axis=0)

    X_train = ft_normalize(X_train, min_train, max_train)
    X_eval = ft_normalize(X_eval, min_train, max_train)
    return (X_train, X_eval)

def main():
    try:
        df = load("datasets/dataset_train.csv")
        features = [
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Divination",
            "History of Magic",
            "Transfiguration",
            "Charms",
            "Flying",
            "Hogwarts House",
        ]
        df = df[features]
        label = features.pop()
        X = df[features]
        y = df.loc[:, label]
        
        np.random.seed(SEED)
        X_train, X_eval, y_train, y_eval = split_dataset(X, y, EVAL_DATASET_SIZE)
        X_train.fillna(X_train.mean(), inplace=True)
        X_eval.fillna(X_eval.mean(), inplace=True)
        
        X_train = X_train.to_numpy()
        X_eval = X_eval.to_numpy()
        y_train = y_train.to_numpy()
        y_eval = y_eval.to_numpy()

        X_train, X_eval = normalize_features(X_train, X_eval)    


        ovr = OVR(X_train, y_train, X_eval, y_eval, seed=SEED)
        ovr.fit()
        test(X_eval, y_eval, ovr.W, ovr.b)

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")

if __name__ == "__main__":
    main()
