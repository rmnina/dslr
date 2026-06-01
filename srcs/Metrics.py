import numpy as np
import matplotlib.pyplot as plt


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
        return (self.TP / (self.TP + self.FN))

    def get_accuracy(self, global_accuracy: bool = False):
        if global_accuracy:
            return self.TP.sum() / self.support.sum()
        return (self.TP + self.TN) / self.support.sum()

    def get_precision(self):
        return self.TP / (self.TP + self.FP)

    def get_F1_score(self):
        return (2 * (self.precision * self.recall) / (self.precision + self.recall))

    def get_F1_macro(self):
        return (self.f1.sum() / self.class_count)
