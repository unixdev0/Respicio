import numpy as np

class Config:
    alpha = 0.01

def nonlin(x,deriv=False):
    if(deriv==True):
        return x * (1-x)
    return 1/(1+np.exp(-x))

def predict(model, x):
    l0 = x
    l1 = nonlin(np.dot(l0,model))
    return l1


def train(X,y):
    m = X.shape[0]
    np.random.seed(1)
    syn0 = 2 * np.random.random((3,1)) - 1

    for iter in xrange(1000):
        l0 = X
        l1 = nonlin(np.dot(l0,syn0)) #4x1
        delta2 = l1 - y #4x1
        l1_grad = delta2 * l0
        DeltaGrad = np.sum(l1_grad, axis=0)/m
        dgrad = np.array([DeltaGrad])

        syn0 -= Config.alpha * dgrad.T
        #l1_delta = l1_err * nonlin(l1, True)
        #l1_delta = l1_err.dot(syn0)*nonlin(l1,True)
        #syn0 += np.dot(l0.T, l1_delta)
    return syn0


X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]])

y = np.array([[0,0,1,1]]).T
model = train(X, y)

for i in range(X.shape[0]):
    l1 = predict(model, X[i,:])
    print "Output after training:"
    if l1 >= 0.5 :
        pred = 1
    else:
        pred = 0
    print pred, y[i, 0]

sample = np.array([
    [1,0,0],
    [1,1,0],
    [0,0,0]])

for s in range(sample.shape[0]):
    l1 = predict(model, sample[s])
    print("Prediction  [%d]:  %f" % (s, l1))
    print sample[s]
    if l1 >= 0.75:
        pred = 1
    elif l1 < 0.75 and l1 > 0.25:
        pred = l1
    else:
        pred = 0
    print ("%f" % pred)
