# MNIST From Scratch

A multilayer perceptron that reads handwritten digits, written in plain Python and NumPy. No PyTorch or TensorFlow, the forward pass, backpropagation and weight updates are all hand-coded.

Built after watching 3Blue1Brown's neural network series.

## Results

95.6% accuracy on the 10,000 image MNIST test set (96.3% on train) after 5 epochs.

| Digit | Test accuracy |
|---|---|
| 0 | 99.2% |
| 1 | 98.2% |
| 2 | 92.8% |
| 3 | 96.5% |
| 4 | 91.3% |
| 5 | 94.3% |
| 6 | 97.0% |
| 7 | 96.6% |
| 8 | 95.4% |
| 9 | 94.5% |

## How it works

- Input: 24x24 grayscale image flattened to 576 values
- 2 hidden layers of 64 sigmoid nodes
- 10 sigmoid outputs, one per digit
- Squared error cost, plain SGD (one image at a time), learning rate 0.5

The math is written out with loops on purpose so every step of backprop is visible. It's slow to train (a while per epoch) but that's the point.

## Files

- `main.py` - the network, training loop, and a `test(index)` function that draws a test image in the terminal and shows the top 3 guesses
- `evaluate.py` - vectorized accuracy check of `weights.json` on the full train and test sets
- `weights.json` - trained weights
- `mnist/convert_mnist.py` - converts the raw MNIST files to 24x24 `.npz` archives
- `digit_collector.py` - tkinter app for drawing my own 24x24 digits, saved into `dataset/` (not included in this repo)

## Running it

```bash
pip install -r requirements.txt
python3 mnist/convert_mnist.py   # builds mnist/mnist_24.npz
python3 evaluate.py              # accuracy of the saved weights
python3 main.py                  # retrains from scratch and overwrites weights.json
```

## Next

- Train and test on my own handwritten digits
- Softmax + cross-entropy output
- A CNN version
