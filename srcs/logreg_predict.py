from OneVsRestClassifier import OneVsRestClassifier as OVR
from utils import load, ft_normalize

def main():
    
    obj = OVR.load_model("model.pkl")
    W = obj["weights"]
    b = obj["bias"]
    min_train = obj["min_train"]
    max_train = obj["max_train"]
    print(W, b, min_train, max_train)
    df = load("datasets/dataset_test.csv")
    features = [
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "History of Magic",
        "Transfiguration",
        "Charms",
        "Flying",
    ]
    X = df[features].to_numpy()
    X_test = ft_normalize(X, min_train, max_train)
    preds = OVR.predict(X_test, W, b)
    print(preds)


if __name__ == "__main__":
    main()