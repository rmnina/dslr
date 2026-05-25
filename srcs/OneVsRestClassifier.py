import numpy as np
import pandas as pd
from LogisticRegression import LogisticRegression


class OneVsRestClassifierException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class OneVsRestClassifier:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        seed: int = 42,
        learning_rate: float = 2e-2,
        iteration: int = 5000,
                ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.seed = seed
        self.learning_rate = learning_rate
        self.iteration = iteration

        self.m, self.n = X_train.shape
        self.m_eval, self.n_eval = X_eval.shape

        self.classes_train = np.unique_counts(self.y_train)
        self.classes_eval = np.unique_counts(self.y_eval)

        np.random.seed(self.seed)
        self.class_count = self.get_class_count()

        self.W = [0] * self.class_count
        self.b = [0] * self.class_count

        self.y_train_mapped, self.y_eval_mapped = self.label_mapping()
        self.y_train_one_hot, self.y_eval_one_hot = self.one_hot_encoding()
        self.models = [0] * self.class_count


    def get_class_count(self) -> int:
        if not np.array_equal(self.classes_train.values, self.classes_eval.values):
            raise OneVsRestClassifierException("Different class count between train set and eval set")
        class_count = len(self.classes_train.values)
        return class_count

    def get_distribution(self) -> None:
        total_train = self.y_train.size
        total_eval = self.y_eval.size

        print("Class distribution in training dataset:")

        for i, cls in enumerate(self.classes_train.values):
            count = self.classes_train.counts[i]
            percentage = count / total_train * 100
            print(f"{cls} : {percentage:.2f}% ({count}/{total_train})")

        print("\nClass distribution in eval dataset:")

        for i, cls in enumerate(self.classes_eval.values):
            count = self.classes_eval.counts[i]
            percentage = count / total_eval * 100
            print(f"{cls} : {(percentage):.2f}% ({count}/{total_eval})")


    def label_mapping(self) -> tuple[np.ndarray]:
        class_mapping = {}
        for i, cls in enumerate(self.classes_train.values):
            class_mapping[cls] = i
        y_train_mapped = pd.Series(self.y_train).map(class_mapping).to_numpy()
        y_eval_mapped = pd.Series(self.y_eval).map(class_mapping).to_numpy()
        return (y_train_mapped, y_eval_mapped)


    def one_hot_encoding(self) -> tuple[np.ndarray]:
        y_train_one_hot = np.zeros((self.y_train.size, self.class_count))
        y_eval_one_hot = np.zeros((self.y_eval.size, self.class_count))

        y_train_one_hot[np.arange(self.y_train.size), self.y_train_mapped] = 1
        y_eval_one_hot[np.arange(self.y_eval.size), self.y_eval_mapped] = 1

        return (y_train_one_hot, y_eval_one_hot)

    def fit(self) -> None:

        for i in range(self.class_count):
            print(f"\nTraining of classifier '{self.classes_train.values[i]}':")
            self.models[i] = LogisticRegression(
                self.X_train,
                self.y_train_one_hot[:, i],
                self.X_eval,
                self.y_eval_one_hot[:, i],
                seed=self.seed,
                learning_rate=self.learning_rate,
                iteration=self.iteration
                )
            self.models[i].fit()
            self.W[i] = self.models[i].W
            self.b[i] = self.models[i].b
            
    @staticmethod
    def predict(X, W, b) -> np.ndarray:
        P = np.ndarray((len(W), X.shape[0]))

        for i in range(len(W)):
            P[i] = LogisticRegression.predict(X, W[i], b[i])
        preds = np.argmax(P, axis=0)
        return (preds)
