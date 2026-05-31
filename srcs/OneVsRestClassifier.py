import numpy as np
import pandas as pd
import pickle
from LogisticRegression import LogisticRegression


class OneVsRestClassifierException(Exception):
    """
    Custom Exception class for OneVsRestClassifier.
    """
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class OneVsRestClassifier:
    """
    Docstring.
    """
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

        self.W = np.ndarray((self.class_count, self.n))
        self.b = np.ndarray(self.class_count)

        self.y_train_mapped, self.y_eval_mapped = self.label_mapping()
        self.y_train_one_hot, self.y_eval_one_hot = self.one_hot_encoding()
        self.models = [0] * self.class_count


    def get_class_count(self) -> int:
        """
        Get classes count in dataset.
        Raises:
            OneVsRestClassifierException: Returns an exception if classes in training dataset don't match classes in evaluation dataset. 

        Returns:
            class_count (int): The count of classes.
        """
        if not np.array_equal(self.classes_train.values, self.classes_eval.values):
            raise OneVsRestClassifierException("Different class count between train set and eval set")
        class_count = len(self.classes_train.values)
        return class_count

    def get_distribution(self) -> None:
        """
        Prints distribution of each class for training and evaluation datasets.
        """
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
        """
        Map classes labels to unique int array.
        Example:
            [A, B, B, C] becomes -> [0, 1, 1, 2]
        Returns:
            tuple[np.ndarray]: Mapped arrays of labels
        """
        class_mapping = {}
        for i, cls in enumerate(self.classes_train.values):
            class_mapping[cls] = i
        y_train_mapped = pd.Series(self.y_train).map(class_mapping).to_numpy()
        y_eval_mapped = pd.Series(self.y_eval).map(class_mapping).to_numpy()
        return (y_train_mapped, y_eval_mapped)


    def one_hot_encoding(self) -> tuple[np.ndarray]:
        """
        Encodes class labels as a one-hot numeric array.
        Example:
                [A, B, B, C] becomes -> [[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]]
        Returns:
            tuple[np.ndarray]: One-hot encoded arrays of labels.
        """
        y_train_one_hot = np.zeros((self.y_train.size, self.class_count))
        y_eval_one_hot = np.zeros((self.y_eval.size, self.class_count))

        y_train_one_hot[np.arange(self.y_train.size), self.y_train_mapped] = 1
        y_eval_one_hot[np.arange(self.y_eval.size), self.y_eval_mapped] = 1

        return (y_train_one_hot, y_eval_one_hot)

    def fit(self) -> None:
        """
        Trains a number _n_ of LogisticRegression model according to class count in dataset.
        
        Stores:
            self.W: The parameters _W_ in a (n, m) array, _m_ being the number of features in the dataset.
            self.b: The bias _b_ in a (n,) array.
        """
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
    def predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Returns predictions for each row in X dataset in form of int labelled class.
        Example:
            **X** is represented as: \\
            [x11, x12, ... x1n]\\
            [x21, x22, ... x2n]\\
            [xm1, xm2, ... xmn]
            
            For _m_ rows of _n_ features, makes a logistic regression model for each class _c_,
            then return indexes of max confidence value in interval [0; c)
            in an array (m,).
            
        Args:
            X (np.ndarray): The dataset for which labels must be predicted
            W (np.ndarray): The parameters _W_ of each logistic regression models in a (c, n) array
            b (np.ndarray): The bias _b_ of each logistic regression models in a (c,) array.

        Returns:
            preds (np.ndarray): indexes of max confidence value corresponding to each mapped label
        """
        P = np.ndarray((len(W), X.shape[0]))

        for i in range(len(W)):
            P[i] = LogisticRegression.predict(X, W[i], b[i])
        preds = np.argmax(P, axis=0)
        return preds

    def save_model(self, model_path: str):
        with open(model_path, "wb") as file:
            pickle.dump(obj=(self.W, self.b), file=file)
            
    @staticmethod
    def load_model(model_path: str):
        with open(model_path, "rb") as file:
            W, b = pickle.load(file)
        return W, b