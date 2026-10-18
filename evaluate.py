import numpy as np, json
d = np.load("mnist/mnist_24.npz")
m = {k: np.array(v) for k, v in json.load(open("weights.json")).items()}
sig = lambda z: 1/(1+np.exp(-z))
def acc(x, y):
    x = x.reshape(-1, 576)/255.0
    out = sig(sig(sig(x@m["W1"]+m["b1"])@m["W2"]+m["b2"])@m["W3"]+m["b3"])
    p = out.argmax(1)
    return (p==y).mean(), p
tr,_ = acc(d["x_train"], d["y_train"])
te,p = acc(d["x_test"], d["y_test"])
y = d["y_test"]
print(f"train acc {tr*100:.2f}%  ({len(d['y_train'])})")
print(f"test  acc {te*100:.2f}%  ({len(y)})")
for k in range(10):
    s = y==k
    print(f"  digit {k}: {(p[s]==k).mean()*100:5.1f}%  n={s.sum()}")
