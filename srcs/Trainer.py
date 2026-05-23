import numpy as np

class Trainer:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        seed: int = 42,
        learning_rate: float = 2e-2,
        iteration: int = 4000
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
        self.W = np.random.rand(self.n)
        self.b = np.random.rand()
        

    # z will be equal to W dot X + b
    # g(z) >= 0.5 when z >= 0, then W dot X + b >= 0 then y = 1
    @staticmethod
    def sigmoid(z) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    
    @staticmethod
    def predict(X, W, b) -> np.ndarray:
        z = np.dot(X, W) + b
        preds = Trainer.sigmoid(z)
        return preds


    def compute_cost(self, y: np.ndarray, pred: np.ndarray) -> int:
        return (-y * np.log(pred) - (1 - y) * np.log(1 - pred)).sum() / self.m_eval


    def fit(self) -> None:
        for i in range(self.iteration):
            preds = Trainer.predict(self.X_train, self.W, self.b)
            errors = preds - self.y_train
            dW = np.dot(self.X_train.T, errors) / self.m
            db = errors.sum() / self.m
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            
            if i % (self.iteration // 20) == 0:
                test_preds = self.predict(self.X_eval, self.W, self.b)
                cost = self.compute_cost(self.y_eval, test_preds)
                print(cost)
