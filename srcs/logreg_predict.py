from OneVsRestClassifier import OneVsRestClassifier as OVR
from utils import load, ft_normalize
import pandas as pd
import argparse


def parse_arguments() -> str:
    parser = argparse.ArgumentParser("Prediction script, takes test dataset as parameter.")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Path of dataset")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path of model")
    args = parser.parse_args()
    return args.dataset, args.model


def main():
    dataset_path, model_path = parse_arguments()
    obj = OVR.load_model(model_path)
    W = obj["weights"]
    b = obj["bias"]
    min_train = obj["min_train"]
    max_train = obj["max_train"]
    
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
    ]
    X = df[features]
    X.fillna(X.mean(), inplace=True).to_numpy()
    X_test = ft_normalize(X, min_train, max_train)
    
    preds = OVR.predict(X_test, W, b)
    
    class_mapping = {0: "Gryffindor", 1: "Hufflepuff", 2: "Ravenclaw", 3: "Slytherin"}
    preds_dataframe = pd.DataFrame(pd.Series(preds).map(class_mapping)).reset_index()
    preds_dataframe.columns = ["Index", "Hogwarts House"]

    preds_dataframe.to_csv("houses.csv", columns=["Index", "Hogwarts House"], index=False)
    print("Predictions saved in 'houses.csv'")

if __name__ == "__main__":
    main()