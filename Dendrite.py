import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


class DendriticNeuron:
    def __init__(self, input_size):
        self.N1 = np.random.randn(input_size)
        self.N2 = np.random.randn(input_size)
        self.N3 = np.random.randn(input_size)

    def forward(self, values):
        values = np.asarray(values, dtype=float)
        d1 = np.maximum(0, np.dot(values, self.N1))
        d2 = np.maximum(0, np.dot(values, self.N2))
        d3 = np.maximum(0, np.dot(values, self.N3))
        integrated = d1 + d2 * d3 / (d1 + 1e-8) + d2 + d3
        return np.maximum(0, integrated)


class NonDendriticNeuron:
    def __init__(self, input_size):
        self.M1 = np.random.randn(input_size)
        self.M2 = np.random.randn(input_size)
        self.M3 = np.random.randn(input_size)

    def forward(self, values):
        values = np.asarray(values, dtype=float)
        return np.sum(
            [np.tanh(np.dot(values, weights))
             for weights in (self.M1, self.M2, self.M3)]
        )


class NeuronLayer1:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)

    def forward(self, values):
        z = np.dot(np.asarray(values, dtype=float), self.weights)
        return 0.5 * z * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3))
        )


class NeuronLayer2:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)

    def forward(self, values):
        z = np.dot(np.asarray(values, dtype=float), self.weights)
        return np.maximum(0, z)


class CentralNeuronLayer:
    def activation(self, first, second):
        z = np.asarray(first, dtype=float) + np.asarray(second, dtype=float)
        relu = np.maximum(0, z)
        tanh = np.tanh(z)
        gelu = 0.5 * z * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3))
        )
        return relu, tanh, gelu


# Create numeric feature data for the neuron and classifier code.
x = np.asarray([
    [0.75, 1.0],
    [0.65, 0.0],
    [0.50, 0.0],
    [0.80, 1.0],
    [0.60, 1.0],
], dtype=float)

y = np.asarray([1.0, 0.0, 0.0, 1.0, 1.0], dtype=float)
x_array = x
neuron = DendriticNeuron(x_array.shape[-1])
print("Dendritic output:", neuron.forward(x))

non_dendritic = NonDendriticNeuron(len(x))
# Use the number of features (2), not the number of samples (5).
non_dendritic = NonDendriticNeuron(x.shape[1])
print("Non-dendritic output:", non_dendritic.forward(x))

# Each layer needs one weight per feature, not per sample.
layer1 = NeuronLayer1(x.shape[1])
layer2 = NeuronLayer2(x.shape[1])

print("Layer 1 output:", layer1.forward(x))
print("Layer 2 output:", layer2.forward(x))

central_layer = CentralNeuronLayer()
print("Central activations:", central_layer.activation(x, y[:, np.newaxis]))

X = x
Y = y

tree_pipeline = Pipeline([
    ("classifier", DecisionTreeClassifier(random_state=42))
])
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    (NeuronLayer1, PCA(n_components=3)),
    (NeuronLayer2, PCA(n_components=3))
])
# X contains 3 samples and 1 feature, so PCA can retain at most 1 component.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=1)),
])
x_transformed = pipeline.fit_transform(np.asarray(X, dtype=float))
tree_pipeline.fit(
    np.asarray(X, dtype=float),
    np.asarray(Y).ravel(),
)
print("Predictions:", tree_pipeline.predict(X)) 
print(x_transformed)  


def backward(self, CentralNeuronLayer, NeuronLayer2):
    z = np.asarray(CentralNeuronLayer, dtype=float) + np.asarray(NeuronLayer2, dtype=float)
    relu = np.minimum(0, z)
    tanh = np.tanh(z)
    return relu, tanh     
# Convert Y to a column so it broadcasts across the two features.
y_column = Y[:, np.newaxis]

central_output = central_layer.activation(X, y_column)[0]
layer1_output = layer1.forward(X)[:, np.newaxis]
layer2_output = layer2.forward(X)[:, np.newaxis]

Neurons = central_output + layer1_output + layer2_output
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=1)),
    ("classifier", DecisionTreeClassifier(random_state=42)),
])

pipeline.fit(X, Y)
print("Predictions:", pipeline.predict(Neurons))
print(Neurons)