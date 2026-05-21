from utils import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

THRESHOLD = 0.5
SEED = 42
LR = 2e-2
ITERATIONS = 5000

# z will be equal to W dot X + b
# g(z) >= 0.5 when z >= 0, then W dot X + b >= 0 then y = 1
def ft_sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ft_normalize(X_train: np.ndarray) -> np.ndarray:
    return (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0)) 


def ft_cost(y_train: np.ndarray, pred: np.ndarray, m: int) -> int:
    # print(y_train.shape)
    # print(pred.shape)
    return (-y_train * np.log(pred) - (1 - y_train) * np.log(1 - pred)).sum() / m


def ft_predict(X_train, W, b):
    z = np.dot(X_train, W) + b
    preds = ft_sigmoid(z)
    return preds

            

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
    X = ft_normalize(df[features].to_numpy())
    house_mapping = {"Gryffindor": 1, "Slytherin": 0, "Ravenclaw": 0, "Hufflepuff": 0}
    y = df.loc[:, "Hogwarts House"].map(house_mapping).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    m, n = X_train.shape
    m_test, n_test = X_test.shape
    b = np.random.rand()
    W = np.random.rand(n)
    W = W.reshape((n, 1))
    y_train = y_train.reshape((m, 1))
    y_test = y_test.reshape((m_test, 1))
    
    # print("X_train shape", X_train.shape)
    # print("X_test shape", X_test.shape)
    # print("y_train shape", y_train.shape)
    # print("y_test shape", y_test.shape)
    
    W, b = gradient_descent(X_train, y_train, X_test, y_test, m, n, W, b)
    test(X_test, y_test, W, b)
    # preds = ft_predict(X_train, W, b)
    # cost = ft_cost(y_train, preds, m)
    # print(cost)
    # W_derivatives, b_derivative = compute_gradients(X_train, y_train, preds, m, n, W)
    



if __name__ == "__main__":
    main()