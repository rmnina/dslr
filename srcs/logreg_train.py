from utils import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Trainer import Trainer

THRESHOLD = 0.5
SEED = 42
LR = 2e-2
ITERATIONS = 5000

def ft_normalize(X: np.ndarray, min_train: float, max_train:float) -> np.ndarray:
    return (X - min_train) / (max_train - min_train) 


def test(X_test, y_test, W, b):
    # test_preds = ft_predict(X_test, W, b)
    test_preds = Trainer.predict(X_test, W, b)
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


def main():
    np.random.seed(SEED)
    df = load("datasets/dataset_train.csv")
    df = df.dropna(axis=0, how="any")
    features = [
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "History of Magic",
        "Transfiguration",
        "Charms",
        "Flying"
    ]
    X = df[features].to_numpy()
    house_mapping = {"Gryffindor": 1, "Slytherin": 0, "Ravenclaw": 0, "Hufflepuff": 0}
    y = df.loc[:, "Hogwarts House"].map(house_mapping).to_numpy()
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=SEED)

    min_train = X_train.min(axis=0)
    max_train = X_train.max(axis=0)

    X_train = ft_normalize(X_train, min_train, max_train)
    X_eval = ft_normalize(X_eval, min_train, max_train)

    model = Trainer(X_train, y_train, X_eval, y_eval)
    model.fit()
    test(X_eval, y_eval, model.W, model.b)


if __name__ == "__main__":
    main()