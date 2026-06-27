
import pandas as pd
import math

# Read data
df = pd.read_csv("data.csv", sep=r"\s+")

# Features and Target
X = df[["MonthlyCharges", "Tenure"]].values
Y = df["Churn"].values

# Hyperparameters
learning_rate = 0.001
epochs = 1000


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def train(X, Y, learning_rate, epochs):
    w1 = 0
    w2 = 0
    b = 0

    for epoch in range(epochs):

        total_loss = 0

        for i in range(len(X)):

            monthly_charges = X[i][0]
            tenure = X[i][1]
            actual = Y[i]

            # Forward Pass
            z = w1 * monthly_charges + w2 * tenure + b
            prediction = sigmoid(z)

            # Binary Cross Entropy Loss
            loss = -(actual * math.log(prediction + 1e-9) +
                     (1 - actual) * math.log(1 - prediction + 1e-9))

            total_loss += loss

            # Gradients
            error = prediction - actual

            dw1 = error * monthly_charges
            dw2 = error * tenure
            db = error

            # Update Parameters
            w1 -= learning_rate * dw1
            w2 -= learning_rate * dw2
            b -= learning_rate * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss = {total_loss:.4f}")

    return w1, w2, b


def predict(monthly_charges, tenure, w1, w2, b):
    z = w1 * monthly_charges + w2 * tenure + b
    return sigmoid(z)


# Train the model
w1, w2, b = train(X, Y, learning_rate, epochs)

print("\nTraining Complete")
print(f"w1 = {w1}")
print(f"w2 = {w2}")
print(f"b  = {b}")

# Test on a new customer
monthly_charges = 55
tenure = 14

prediction = predict(monthly_charges, tenure, w1, w2, b)

print(f"\nProbability of Churn: {prediction:.4f}")

if prediction >= 0.5:
    print("Prediction: Customer will Churn")
else:
    print("Prediction: Customer will Stay")

