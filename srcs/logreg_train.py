from utils import load, ft_normalize, split_dataset
import numpy as np
from OneVsRestClassifier import OneVsRestClassifier as OVR
import pandas as pd
from Metrics import Metrics
import argparse

THRESHOLD = 0.5
SEED = 13
EVAL_DATASET_SIZE = 0.2
LEARNING_RATE = 2e-2
ITERATION = 5000
SAVE_CM = True


def test(X_eval, y_eval, W, b) -> None:
    class_mapping = {"Gryffindor": 0, "Hufflepuff": 1, "Ravenclaw": 2, "Slytherin": 3}
    y_eval_mapped = pd.Series(y_eval).map(class_mapping).to_numpy()
    preds = OVR.predict(X_eval, W, b)

    target_names = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    metrics = Metrics(y_true=y_eval_mapped, y_pred=preds, target_names=target_names)

    print("\nClassification report:")
    print(metrics.classification_report())

    cm = Metrics.get_confusion_matrix(y_true=y_eval_mapped, y_pred=preds, class_count=len(target_names))
    Metrics.plot_confusion_matrix(confusion_matrix=cm, target_names=target_names, save=SAVE_CM)
    if SAVE_CM:
        print("Confusion matrix saved in 'data_visualization/confusion_matrix.png'")


def parse_arguments() -> str:
    parser = argparse.ArgumentParser("Training script, takes train dataset as parameter.")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Path of dataset")
    args = parser.parse_args()
    return args.dataset


def main():
    try:
        dataset_path = parse_arguments()
        df = load(dataset_path)
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
        X_train, X_eval, y_train, y_eval = split_dataset(X, y, EVAL_DATASET_SIZE, SEED)
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
        print("Model saved in 'model.pkl'")

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")


if __name__ == "__main__":
    main()
