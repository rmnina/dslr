from utils import load
import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = 0.5
SEED = 42

# z will be equal to W dot X + b
# g(z) >= 0.5 when z >= 0, then W dot X + b >= 0 then y = 1
def ft_sigmoid(z):
    return 1 / (1 + np.exp(-z))

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
    X_train = df[features].to_numpy()
    house_mapping = {"Gryffindor": 1, "Slytherin": 0, "Ravenclaw": 0, "Hufflepuff": 0}
    y_train = df.loc[:, "Hogwarts House"].map(house_mapping).to_numpy()
    print("X_train\n\n", X_train)
    print("\ny_train\n\n", y_train)
    m, n = X_train.shape
    b = np.random.rand()
    W = np.random.rand(8)
    z = np.dot(X_train, W) + b
    sig = ft_sigmoid(z)
    print(sig)


if __name__ == "__main__":
    main()