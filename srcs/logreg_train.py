from utils import load
import numpy as np
import matplotlib.pyplot as plt
from srcs.LogisticRegression import LogisticRegression
import random

THRESHOLD = 0.5
SEED = 42
EVAL_DATASET_SIZE = 0.2

def split_dataset(X: np.ndarray, y: np.ndarray, eval_size: float) -> tuple[np.ndarray]:
    #TODO: check if EVAL_DATASET_SIZE between 0 and 1
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
    return (X - min_train) / (max_train - min_train) 


def test(X_test, y_test, W, b):
    # test_preds = ft_predict(X_test, W, b)
    test_preds = LogisticRegression.predict(X_test, W, b)
    success = 0
    for i in range(len(X_test)):
        if test_preds[i] >= 0.5:
            griffondor = 1
        else:
            griffondor = 0
        if griffondor == y_test[i]:
            success += 1
    #     print(f"Pred: {test_preds[i]} / truth: {y_test[i]}")
    print(f"success rate: {(success / len(X_test) * 100):.2f}%")


def one_versus_all(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray) -> None:
    pass
    

def main():
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
    house_mapping = {"Gryffindor": 1, "Slytherin": 0, "Ravenclaw": 0, "Hufflepuff": 0}
    y = df.loc[:, label].map(house_mapping)
    
    np.random.seed(SEED)
    X_train, X_eval, y_train, y_eval = split_dataset(X, y, EVAL_DATASET_SIZE)
    X_train.fillna(X_train.mean(), inplace=True)
    X_eval.fillna(X_eval.mean(), inplace=True)
    
    X_train = X_train.to_numpy()
    X_eval = X_eval.to_numpy()
    y_train = y_train.to_numpy()
    y_eval = y_eval.to_numpy()
    
    print(X_train.shape, X_eval.shape, y_train.shape, y_eval.shape)
    
    min_train = X_train.min(axis=0)
    max_train = X_train.max(axis=0)

    X_train = ft_normalize(X_train, min_train, max_train)
    X_eval = ft_normalize(X_eval, min_train, max_train)

    model = LogisticRegression(X_train, y_train, X_eval, y_eval, seed=SEED)
    model.fit()
    test(X_eval, y_eval, model.W, model.b)


if __name__ == "__main__":
    main()