import numpy as np
import json

data = np.load("mnist/mnist_24.npz")

H2 = []
H3 = []
output = []
x_train = data["x_train"].reshape(-1, 576) / 255.0
y_train = data["y_train"]
x_test = data["x_test"].reshape(-1, 576) / 255.0
y_test = data["y_test"]
L1 = 64
L2 = 64

label_list = {
    "0": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "1": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "2": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "3": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "4": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "5": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "6": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "7": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "8": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "9": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculateNode(initValues, Weights, bias):
    presig = 0
    i = 0
    while i < len(initValues):
        sumSig = initValues[i]*Weights[i]
        presig += sumSig
        i += 1
    sig = sigmoid(presig + bias)
    return sig

def calculateLayer(inputs, W, B):
    outputs = []
    numNodes = len(W[0])

    j=0
    while j < numNodes:
        nodeWeights = []
        i = 0
        while i < len(inputs):
            nodeWeights.append(W[i][j])
            i += 1

        nodeValue = calculateNode(inputs, nodeWeights, B[j])
        outputs.append(nodeValue)
        j += 1
    
    return outputs

def initWeights():
    global L1, L2
    W1temp = np.random.randn(576, L1) * np.sqrt(1/576)
    b1temp = np.zeros(L1)
    W2temp = np.random.randn(L1, L2) * np.sqrt(1/16)
    b2temp = np.zeros(L2)
    W3temp = np.random.randn(L2, 10) * np.sqrt(1/16)
    b3temp = np.zeros(10)
    return W1temp, b1temp, W2temp, b2temp, W3temp, b3temp

def calculateCost(output, perfTest):
    cost = 0
    i = 0
    while i < len(output):
        diff = output[i] - perfTest[i]
        cost += diff * diff
        i += 1
    return cost

def outputDeltas(output, target):
    deltas = []
    k = 0
    while k <len(output):
        error = 2 * (output[k] - target[k])
        slope = output[k] * (1-output[k])
        deltas.append(error * slope)
        k += 1

    return deltas

def hiddenDeltas(layerValues, W_next, nextDeltas):
    deltas = []
    j = 0
    while j < len(layerValues):
        blame = 0
        k = 0
        while k < len(nextDeltas):
            blame += W_next[j][k] * nextDeltas[k]
            k += 1
        slope = layerValues[j] * (1 - layerValues[j])
        deltas.append(blame * slope)
        j += 1

    return deltas

def updateLayer(W, b, inputs, deltas, lr):
    i = 0
    while i < len(inputs):
        j = 0
        while j < len(deltas):
            W[i][j] -= lr * inputs[i] * deltas[j]
            j += 1
        i += 1
    j = 0
    while j < len(deltas):
        b[j] -= lr * deltas[j]
        j += 1

def saveWeights(W1, b1, W2, b2, W3, b3, path="weights.json"):
    model = {
        "W1": W1.tolist(), "b1": b1.tolist(),
        "W2": W2.tolist(), "b2": b2.tolist(),
        "W3": W3.tolist(), "b3": b3.tolist(),
    }
    with open(path, "w") as f:
        json.dump(model, f)


def train():
    W1, b1, W2, b2, W3, b3 = initWeights()
    lr = 0.5
    
    for epoch in range(5):
        order = np.random.permutation(len(x_train))
        totalCost = 0
        n = 0
        while n < len(x_train):
            x = x_train[order[n]]
            target = label_list[str(y_train[order[n]])]

            H2 = calculateLayer(x, W1, b1)
            H3 = calculateLayer(H2, W2, b2)
            output = calculateLayer(H3, W3, b3)
            totalCost += calculateCost(output, target)

            d3 = outputDeltas(output, target)
            d2 = hiddenDeltas(H3, W3, d3)
            d1 = hiddenDeltas(H2, W2, d2)

            updateLayer(W3, b3, H3, d3, lr)
            updateLayer(W2, b2, H2, d2, lr)
            updateLayer(W1, b1, x, d1, lr)
            n += 1

        saveWeights(W1, b1, W2, b2, W3, b3)
        print ("epoch", epoch, "avg cost", totalCost / len(x_train))
    return W1, b1, W2, b2, W3, b3

def describeGuess(output):
    ranked = np.argsort(output)[::-1]       # digit indices, highest output first
    total = sum(output)
    lines = []
    for digit in ranked[:3]:
        pct = output[digit] / total * 100
        bar = "#" * int(pct / 5)
        lines.append(f"  {digit}: {pct:5.1f}%  {bar}")
    return ranked[0], "\n".join(lines)

def showImage(x):
    chars = " .:*#"
    for row in range(24):
        print("".join(chars[int(x[row * 24 + col] * 4.99)] for col in range(24)))


def test(index):
    data = {}
    with open("weights.json", "r") as file:
        data = json.load(file)

    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]
    W3 = data["W3"]
    b3 = data["b3"]
    x = x_test[index]
    target = label_list[str(y_test[index])]

    H2 = calculateLayer(x, W1, b1)
    H3 = calculateLayer(H2, W2, b2)
    output = calculateLayer(H3, W3, b3)
    cost = calculateCost(output, target)

    guess, breakdown = describeGuess(output)
    showImage(x)
    print(f"Guess: {guess}   Actual: {y_test[index]}   {'✓' if guess == y_test[index] else '✗'}")
    print(breakdown)
    print(f"Cost: {cost:.4f}")

train()