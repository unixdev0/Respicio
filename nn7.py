import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

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

class Config:
    #nn_input_dim = 5  # input layer dimensionality
    #nn_output_dim = 4  # output layer dimensionality
    #nn_hidden_dim = 5

    LI = 10
    LH1 = 20
    LH2 = 40
    LH3 = 60
    LH4 = 40
    LO = 4

    num_passes = 3000
    num_samples=1000
    num_train=950
    num_test=50
    noise=0.05
    debug = True
    reg = 0.0001
    alpha = 0.9
    activation = 'sigmoid'
    epsilon = 1e-4
    epsilon_init = 0.12
    moons = True

def generate_data(conf=Config):
    np.random.seed(0)
    if conf.moons == True:
        X, y = datasets.make_moons(conf.num_samples, noise=0.20)
    else:
        X, y = datasets.make_classification(n_samples=conf.num_samples, n_features=conf.LI, n_informative=3, n_redundant=2, n_repeated=0,
                                             n_classes=conf.LO, n_clusters_per_class=1, weights=None, flip_y=0.01,
                                             class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True,
                                             random_state=1)
    return X, y

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def random_Epsilon_calculation(L_in, L_out, debug=False):
    epsilon = np.sqrt(6)/ np.sqrt(L_in + L_out)
    if debug:
        assert isinstance(epsilon, object)
        print("epsilon %f" % epsilon)
    return epsilon

def initialize_Weights(L_in, L_out, debug=False):
    Weights = 2 * Config.epsilon_init * np.random.randn(L_in, L_out) - Config.epsilon_init
    if debug:
        print("Weights %d x %d" % (L_in, L_out))
    return Weights

def build_model(conf=Config):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = initialize_Weights(conf.LI, conf.LH1)
    b1 = np.ones((conf.LH1, 1))
    W2 = initialize_Weights(conf.LH1, conf.LH2)
    b2 = np.ones((conf.LH2, 1))
    W3 = initialize_Weights(conf.LH2, conf.LH3)
    b3 = np.ones((conf.LH3, 1))
    W4 = initialize_Weights(conf.LH3, conf.LO)
    b4 = np.ones((conf.LO, 1))

    # This is what we return at the end
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3, 'W4': W4, 'b4': b4}
    return model

def cost_J(model, h, y, m, conf=Config):
    part1 = -y * np.log(h)
    part0 = (1 - y) * np.log(1 - h)
    diff = np.subtract(part1, part0)
    J_unreg = np.sum(diff)/m
    W1 = model['W1']
    W2 = model['W2']
    W3 = model['W3']
    W4 = model['W4']
    p1 = np.sum( np.power(W1, 2) )
    p2 = np.sum( np.power(W2, 2) )
    p3 = np.sum(np.power(W3, 2))
    p4 = np.sum(np.power(W4, 2))
    reg = conf.reg * ( p1 + p2 + p3 + p4) / (2 * m)
    J = J_unreg + reg
    if conf.debug == True:
        print "Cost J = ", J, ", reg = ", reg
    return J

def train_model(model, X, y, conf=Config):
    # Initialize the parameters to random values. We need to learn these.
    m = len(X)
    lowest_J = tuple((999,0))
    # Gradient descent. For each batch...
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    W3 = model['W3']
    b3 = model['b3']
    W4 = model['W4']
    b4 = model['b4']

    for i in range(0, conf.num_passes):
        # Forward propagation
        a1 = X
        z2 = np.dot(a1, W1) + b1.T
        a2 = nonlin(z2)
        z3 = np.dot(a2, W2) + b2.T
        a3 = nonlin(z3)
        z4 = np.dot(a3, W3) + b3.T
        a4 = nonlin(z4)
        z5 = np.dot(a4, W4) + b4.T
        a5 = nonlin(z5)

        d5 = a5 - y
        d4 = d5.dot(W4.T) * nonlin(a4, True)
        d3 = d4.dot(W3.T) * nonlin(a3, True)
        d2 = d3.dot(W2.T) * nonlin(a2, True)

        grad4 = (d5.T.dot(a4) + conf.reg * (np.sum(W4))) / m
        grad3 = (d4.T.dot(a3) + conf.reg * (np.sum(W3))) / m
        grad2 = (d3.T.dot(a2) + conf.reg * (np.sum(W2))) / m
        grad1 = (d2.T.dot(a1) + conf.reg * (np.sum(W1))) / m

        db4 = np.sum(d5, axis=0, keepdims=True) / m
        db3 = np.sum(d4, axis=0, keepdims=True) / m
        db2 = np.sum(d3, axis=0) / m
        db1 = np.sum(d2, axis=0) / m

        W1 -= conf.alpha * grad1.T
        W2 -= conf.alpha * grad2.T
        W3 -= conf.alpha * grad3.T
        W4 -= conf.alpha * grad4.T
        b1 -= conf.alpha * db1.reshape(conf.LH1, 1)
        b2 -= conf.alpha * db2.reshape(conf.LH2, 1)
        b3 -= conf.alpha * db3.reshape(conf.LH3, 1)
        b4 -= conf.alpha * db4.reshape(conf.LO, 1)

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3':W3, 'b3':b3, 'W4': W4, 'b4': b4}
        if i % 1000 == 0:
            J = cost_J(model, a5, y, m, conf)
            if conf.debug:
                print("Loss after iteration %i: %f" % (i, J))
            if J < lowest_J[0]:
                lowest_J = (J, i)
    return model, lowest_J


def predict(model, x, conf=Config):
    W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4']
    # Forward propagation
    a1 = x
    z2 = np.dot(a1, W1) + b1.T
    a2 = nonlin(z2)
    z3 = np.dot(a2, W2) + b2.T
    a3 = nonlin(z3)
    z4 = np.dot(a3, W3) + b3.T
    a4 = nonlin(z4)
    z5 = np.dot(a4, W4) + b4.T
    a5 = nonlin(z5)
    a5.reshape(1, conf.LO)
    return a5


def main():
    conf = Config()
    conf.debug = True
    conf.num_passes = 3000
    conf.num_samples = 10000
    conf.num_train = 6000
    conf.num_test = 4000
    conf.reg = 0.
    conf.noise = 0.05
    conf.moons = False
    conf.nn_input_dim = 20  # input layer dimensionality
    conf.nn_output_dim = 4  # output layer dimensionality
    conf.nn_hidden_dim = 40

    conf.LI = 20
    conf.LH1 = 40
    conf.LH2 = 60
    conf.LH3 = 40
    conf.LO = 5

    J_history = list()
    alpha = [0.01, 0.03, 0.05, 0.45, 0.5, 0.6, 0.7]

    #alpha = [0.12]
    reg = [0.00005]
    #reg = [ 0, 0.00005, 0.00005, 0.0005, 0.005, 0.05, 0.5]
    for r in reg:
        model = build_model(conf)
        #conf.alpha = 0.03
        conf.alpha = 0.45
        conf.reg = r
        X, y = generate_data(conf)

        if conf.moons == True:
            y_v = np.eye(conf.nn_output_dim)[y, :]
        else:
            y_v = np.eye(conf.LO, dtype=int)[y, :]
        X_train = X[0:conf.num_train, :].copy()
        y_v_train = y_v[0:conf.num_train, :].copy()
        X_pred = X[conf.num_train:conf.num_train + conf.num_test, :].copy()
        y_pred = y_v[conf.num_train:conf.num_train + conf.num_test, :].copy()
        model, lowest_J = train_model(model, X_train, y_v_train, conf)
        error = 0.
        for i in xrange(conf.num_test):
            pred = predict(model, X_pred[i, :], conf)
            pred_tmp = pred.copy()
            pred[pred >= 0.48] = 1
            pred[pred < 0.48] = 0
            if np.array_equal(pred[0, :],y_pred[i]) == False:
                if conf.debug == True:
                    print "Prediction ERROR: raw=", pred_tmp, "expected=", y_pred[i, :], "predicted=", pred[0, :]
                error += 1
        total_error = (error / conf.num_test) * 100
        J_history.append((r, lowest_J, 100 - total_error))

    for j in J_history:
        print "Reg=", j[0], " Cost J=", j[1][0], " Iterations=", j[1][1], " Accuracy", j[2], "%"

if __name__ == "__main__":
    main()