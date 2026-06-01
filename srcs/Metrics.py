import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import ft_normalize


class Metrics():

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: list,
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.target_names = target_names
        self.class_count = len(target_names)
        self.TP, self.FP, self.TN, self.FN = self.count_positive_negative()
        self.support = np.unique_counts(y_true).counts

        self.precision = self.get_precision()
        self.global_accuracy = self.get_accuracy(global_accuracy=True)
        self.recall = self.get_recall()
        self.f1 = self.get_F1_score()
        self.f1_macro = self.get_F1_macro()

    def count_positive_negative(self) -> tuple[np.ndarray]:
        TP = [0] * self.class_count
        FP = [0] * self.class_count
        TN = [0] * self.class_count
        FN = [0] * self.class_count
        for i in range(self.class_count):
            for j in range(len(self.y_pred)):
                if self.y_pred[j] == self.y_true[j] and self.y_true[j] == i:
                    TP[i] += 1
                if self.y_pred[j] != i and self.y_true[j] != i:
                    TN[i] += 1
                if self.y_pred[j] == i and self.y_true[j] != i:
                    FP[i] += 1
                if self.y_pred[j] != i and self.y_true[j] == i:
                    FN[i] += 1
        return (np.array(TP), np.array(FP), np.array(TN), np.array(FN))

    def classification_report(self, digits: int = 4, width: int = 12):
        width = "<" + str(width)
        float_prec = "." + str(digits) + "f"
        buffer = f"\n{" ":{width}}{"precision":{width}}"
        buffer += f"{"recall":{width}}"
        buffer += f"{"f1-score":{width}}"
        buffer += f"{"support":{width}} \n"

        for i, target in enumerate(self.target_names):
            buffer += f"{target:{width}}"
            buffer += f"{self.precision[i]:{width}{float_prec}}"
            buffer += f"{self.recall[i]:{width}{float_prec}}"
            buffer += f"{self.f1[i]:{width}{float_prec}}"
            buffer += f"{self.support[i]:{width}}\n"
        buffer += f"\n{"accuracy":{width}}{"":{width}}{"":{width}}"
        buffer += f"{self.global_accuracy:{width}{float_prec}}"
        buffer += f"{sum(self.support):{width}}\n"

        buffer += f"{"f1-macro":{width}}{"":{width}}{"":{width}}"
        buffer += f"{self.f1_macro:{width}{float_prec}}"
        buffer += f"{sum(self.support):{width}}\n"
        return buffer

    def get_recall(self):
        return (self.TP / (self.TP + self.FN + 1e-20))

    def get_accuracy(self, global_accuracy: bool = False):
        if global_accuracy:
            return self.TP.sum() / self.support.sum()
        return (self.TP + self.TN) / self.support.sum()

    def get_precision(self):
        return self.TP / (self.TP + self.FP + 1e-20)

    def get_F1_score(self):
        return (2 * (self.precision * self.recall) / (self.precision + self.recall + 1e-20))

    def get_F1_macro(self):
        return (self.f1.sum() / self.class_count)

    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> pd.DataFrame:
        confusion_matrix = pd.DataFrame(
            0,
            index=range(class_count),
            columns=range(class_count)
            )

        for i in range(class_count):
            for j in range(len(y_pred)):
                if y_pred[j] == i and y_pred[j] == y_true[j]:
                    confusion_matrix.iloc[i, i] += 1
                if y_pred[j] == i and y_pred[j] != y_true[j]:
                    confusion_matrix.iloc[y_true[j], i] += 1

        return confusion_matrix

    @staticmethod
    def plot_confusion_matrix(confusion_matrix: pd.DataFrame, target_names: list, save: bool = True):
        df_norm = ft_normalize(
            confusion_matrix.to_numpy(),
            confusion_matrix.min().to_numpy(),
            confusion_matrix.max().to_numpy()
            )
        df = pd.DataFrame(df_norm)
        df = df.set_axis(target_names, axis='index')
        df = df.set_axis(target_names, axis='columns')

        sns_plot = sns.heatmap(
            df,
            annot=confusion_matrix,
            fmt='g',
            cmap='Blues'
            )
        sns_plot.set_title(
            label="Confusion Matrix of Hogwarts House predicted"
            )
        plt.xlabel("Predictions")
        plt.ylabel("Ground truth")
        fig = sns_plot.get_figure()
        if save:
            fig.savefig("data_visualization/confusion_matrix.png")
        else:
            plt.show()
