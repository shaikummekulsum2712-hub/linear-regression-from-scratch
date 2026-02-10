import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        n = len(X)

        for _ in range(self.epochs):
            y_pred = self.w * X + self.b

            dw = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return self.w * X + self.b


# Testing the model
if __name__ == "__main__":
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    print("Predictions:", predictions)
