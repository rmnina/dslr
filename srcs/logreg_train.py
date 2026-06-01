from utils import load, ft_normalize
import numpy as np
import random
from OneVsRestClassifier import OneVsRestClassifier as OVR
import pandas as pd
from Metrics import Metrics

THRESHOLD = 0.5
SEED = 13
EVAL_DATASET_SIZE = 0.2
LEARNING_RATE = 2e-2
ITERATION = 5000


def split_dataset(X: np.ndarray, y: np.ndarray, eval_size: float) -> tuple[np.ndarray]:
    """
    Splits each of the two original datasets into, for each original:
        - One training dataset
        - One evaluating dataset
    The split is operated according to the _eval_size_ proportion.
    Example:
        _eval_size_ = 0.2 -> The resulting evaluating datasets will consist
        in 20% of the original datasets. Therefore, the resulting training datasets
        will consist in the remaining 80% of the original datasets.
    The split is made at random to ensure representativity in each resulting dataset.

    Args:
        X (np.ndarray): The original dataset to split, containing the labels.
        y (np.ndarray): The original dataset to split containing the classes.
        eval_size (float): The proportion of the resulting eval datasets, relatively to
        the original datasets.

    Returns:
        X_train, y_train, X_eval, y_eval (tuple[np.ndarray]) : The training and evaluating
        datasets retrieved from the two original datasets.
    """
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


def test(X_eval, y_eval, W, b) -> None:
    class_mapping = {"Gryffindor": 0, "Hufflepuff": 1, "Ravenclaw": 2, "Slytherin": 3}
    y_eval_mapped = pd.Series(y_eval).map(class_mapping).to_numpy()
    preds = OVR.predict(X_eval, W, b)

    target_names = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    metrics = Metrics(y_true=y_eval_mapped, y_pred=preds, target_names=target_names)
    print("\nClassification report:")
    print(metrics.classification_report())


def main():
    try:
        df = load("datasets/dataset_train.csv")
        features = [
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Divination",
            "Muggle Studies",
            "History of Magic",
            "Transfiguration",
            "Charms",
            "Flying",
            "Ancient Runes",
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
        
        min_train = X_train.min(axis=0)
        max_train = X_train.max(axis=0)
        X_train = ft_normalize(X_train, min_train, max_train)    
        X_eval = ft_normalize(X_eval, min_train, max_train)    

        ovr = OVR(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            min_train=min_train,
            max_train=max_train,
            seed=SEED,
            learning_rate=LEARNING_RATE,
            iteration=ITERATION
                )
        ovr.fit()
        test(X_eval, y_eval, ovr.W, ovr.b)
        ovr.save_model("model.pkl")
        print("model saved in model.pkl")

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")


if __name__ == "__main__":
    main()
