import numpy as np
import pandas as pd


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
        iteration: int = 4000,
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
        self.get_class_count()
        self.get_distribution()
        self.W = [np.random.rand(self.n) for _ in range(self.class_count)]
        self.b = [np.random.rand() for _ in range(self.class_count)]
        # print(self.class_count)
        # print(f"W = {self.W}")
        # print(f"B = {self.b}")
        self.one_hot_encoding()
        # self.Y_one_hot

    def get_class_count(self) -> int:
        print(self.classes_train.values, self.classes_eval.values)
        if not np.array_equal(self.classes_train.values, self.classes_eval.values):
            raise OneVsRestClassifierException("Different class count between train set and eval set")
        self.class_count = len(self.classes_train.values)

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

    def one_hot_encoding(self) -> np.ndarray:
        class_mapping = {}
        for i, cls in enumerate(self.classes_train.values):
            class_mapping[cls] = i
        y_train_mapped = pd.Series(self.y_train).map(class_mapping).to_numpy()
        self.y_train_one_hot = np.zeros((self.y_train.size, self.class_count))
        self.y_train_one_hot[np.arange(self.y_train.size), y_train_mapped] = 1
