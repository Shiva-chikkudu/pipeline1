import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt



np.random.rand(90)
x = np.linspace(3, 4*8, 49)
y = np.linspace(4, 5*3, 45)

# Create a target with the same number of samples as x
y = np.linspace(4, 15, len(x))

# Convert the continuous target into two classes
y = (y >= np.median(y)).astype(int)

x_train, x_test, y_train, y_test = train_test_split( x.reshape(-1, 1),y,test_size=0.4,random_state=35,stratify=y)
lr=LogisticRegression()
x = x.reshape(-1, 1)
lr.fit(x, y)
scaler = StandardScaler()



pipeline = pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])
pipeline.fit(x, y)
pipeline.fit(x_train, y_train)

pred_train = pipeline.predict(x_train)
pred_test = pipeline.predict(x_test)

print("Train predictions:\n", pred_train[:10])
pred_test = pipeline.predict(x_test)
print("Train predictions:\n", pred_train[:10])
plt.scatter(x.ravel(), y, label="Actual")

plt.scatter(x_train.ravel(), pred_train, label="Train predictions", marker="x")
plt.scatter(x_test.ravel(), pred_test, label="Test predictions", marker="o")

plt.xlabel("X values")
plt.ylabel("Class")
plt.legend()
plt.show()
plt.scatter(x_train.ravel(), pred_train, label="Train Predictions", marker="x")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()
plt.show()