import numpy as np
import matplotlib.pyplot as plt


class Config:
    n_passes = 5000
    alpha = 0.01
    reg = 0.001
    activation = 'tanh'
    epsilon = 1e-4

J_hist = np.zeros(shape=(Config.n_passes/100, 2))

def nonlin(x, deriv=False):
    if Config.activation == 'sigmoid':
        if(deriv==True):
            return x*(1-x)
        else:
            return 1/(1+np.exp(-x))
    elif Config.activation == 'tanh':
        if(deriv==True):
            return 1.0 - x**2
        else:
            return np.tanh(x)

X = np.array([
    [0,0,1],
    [1,0,1],
    [0,1,1],
    [1,1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

def cost_J(model, h, y, m, conf=Config):
    part1 = -y * np.log(h)
    part0 = (1 - y) * np.log(1 - h)
    diff = np.subtract(part1, part0)
    J = np.sum(diff)/m
    # Calculating the loss
    # Add regulatization term to loss (optional)
    syn1 = model['syn1']
    syn2 = model['syn2']
    p1 = np.sum(np.power(syn1, 2))
    p2 = np.sum(np.power(syn2, 2))
    reg = p1 + p2
    t = conf.reg*reg/(2 * m)
    J_reg = J + t
    return J_reg

def build_model():
    np.random.seed(1)
    syn1 = 2 * np.random.random((3,4)) - 1
    syn2 = 2 * np.random.random((4,1)) - 1
    model = {'syn1': syn1, 'syn2': syn2}
    return model

def predict(model, x):
    a1 = x
    z2 = np.dot(a1, model['syn1'])
    a2 = nonlin(z2)
    z3 = np.dot(a2, model['syn2'])
    a3 = nonlin(z3)
    return a3

def train(model, x, y, conf=Config):
    m = len(x)
    syn1 = model['syn1']
    syn2 = model['syn2']
    for t in xrange(conf.n_passes):
        a1 = X
        z2 = np.dot(a1, syn1)
        a2 = nonlin(z2)
        z3 = np.dot(a2, syn2)
        a3 = nonlin(z3)

        d3 = a3 - y
        d2 = d3.dot(syn2.T) * nonlin(a2, True)
        grad2 = (d3.T.dot(a2) + conf.reg*np.sum(syn2))/m
        grad1 = (d2.T.dot(a1) + conf.reg*(np.sum(syn1)))/m

        syn1 -= conf.alpha * grad1.T
        syn2 -= conf.alpha * grad2.T
        model['syn1'] = syn1
        model['syn2'] = syn2

        if t % 100 == 0:
            J = cost_J(model, a3, y, m, conf)
            J_hist[t / 100, 0] = t/100
            J_hist[t/100, 1] = J
            print ("Cost function J[t]=%f"%J)

    model['syn1'] = syn1
    model['syn2'] = syn2
    return model

def visualize(j):
    j0 = j[:,0]
    j1 = j[:,1]

    plt.plot(j0, j1, 'ro')
    plt.xlabel('i')
    plt.ylabel('Cost J')
    plt.legend(['i - iterations x100'])
    plt.title('Cost J per x100 iterations')
    plt.show()


def main():
    conf = Config()
    model = build_model()
    print "Model generated:"
    print model
    conf.alpha = 0.03
    model = train(model, X, y, conf)
    print "Model trained:"
    print model


    for i in range(len(X)):
        pred = predict(model, X[i, :])
        print X[i, :], pred, y[i]

if __name__ == "__main__":
    main()