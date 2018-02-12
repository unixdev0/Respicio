import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


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

def visualize(X, y, model):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x:predict(model,x), X, y)
    plt.title("Logistic Regression")


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    nn_hidden_dim = 3
    num_passes = 20000
    num_samples=2000
    num_train=1500
    num_test=500
    noise=0.05
    debug = True
    reg = 0.0001
    alpha = 0.9
    activation = 'sigmoid'
    epsilon = 1e-4
    epsilon_init = 0.12

def generate_data(num_of_samples):
    np.random.seed(0)
    X, y = datasets.make_moons(num_of_samples, noise=0.20)
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
    epsilon_init = random_Epsilon_calculation(L_in, L_out, debug)
    Weights = 2 * epsilon_init * np.random.randn(L_in, L_out) - epsilon_init
    # Weights = np.random.randn(L_in, L_out) / np.sqrt(L_in)
    if debug:
        print("Weights %d x %d" % (L_in, L_out))
    return Weights

def build_model(conf=Config):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = initialize_Weights(conf.nn_input_dim, conf.nn_hidden_dim)
    #W1 = 2 * np.random.random((Config.nn_input_dim, Config.nn_hidden_dim)) - 1
    b1 = np.ones((conf.nn_hidden_dim, 1))
    W2 = initialize_Weights(conf.nn_hidden_dim, conf.nn_output_dim)
    #W2 = 2 * np.random.random((Config.nn_hidden_dim, Config.nn_output_dim)) - 1
    b2 = np.ones((conf.nn_output_dim, 1))
    # This is what we return at the end
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    m = len(X)  # training set size
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    a1 = X
    z2 = a1.dot(W1) + b1.T
    a2 = sigmoid(z2)
    z3 = a2.dot(W2) + b2.T
    a3 = sigmoid(z3)
    part1 = -y * np.log(a3)
    part2 = (1 - y) * np.log(1 - a3)
    diff = np.subtract(part1, part2)
    J = np.sum(diff)/m
    # Calculating the loss
    # Add regulatization term to loss (optional)
    #reg = np.sum(np.power(W1, 2)) + np.sum(np.power(W2, 2))
    #reg = (Config.reg_lambda * reg)/(2*m)
    if Config.debug:
        print ("Cost function J(-reg): %f" % J)
    J_reg = J + (np.sum(np.square(W1)) + np.sum(np.square(W2)))*Config.reg/(2*m)
    if Config.debug:
        print ("Cost function J(+reg): %f" % J)
    #J += Config.reg_lambda / 2 * m * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return J, J_reg

def cost_J(model, h, y, m, conf=Config):
    part1 = -y * np.log(h)
    part0 = (1 - y) * np.log(1 - h)
    diff = np.subtract(part1, part0)
    J_unreg = np.sum(diff)/m
    W1 = model['W1']
    W2 = model['W2']
    p1 = np.sum( np.power(W1, 2) )
    p2 = np.sum( np.power(W2, 2) )
    reg = conf.reg * ( p1 + p2 ) / (2 * m)
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
    for i in range(0, conf.num_passes):
        # Forward propagation
        a1 = X
        z2 = np.dot(a1, W1)
        a2 = nonlin(z2)
        z3 = np.dot(a2, W2)
        a3 = nonlin(z3)

        d3 = a3 - y
        d2 = d3.dot(W2.T) * nonlin(a2, True)
        grad2 = (d3.T.dot(a2) + conf.reg * (np.sum(W2))) / m
        grad1 = (d2.T.dot(a1) + conf.reg * (np.sum(W1))) / m
        db2 = np.sum(d3, axis=0, keepdims=True) / m
        db1 = np.sum(d2, axis=0) / m

        W1 -= conf.alpha * grad1.T
        W2 -= conf.alpha * grad2.T
        b1 -= Config.alpha * db1.reshape(Config.nn_hidden_dim, 1)
        b2 -= Config.alpha * db2.reshape(Config.nn_output_dim, 1)

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        if i % 1000 == 0:
            J = cost_J(model, a3, y, m, conf)
            if conf.debug:
                print("Loss after iteration %i: %f" % (i, J))
            if J < lowest_J[0]:
                lowest_J = (J, i)
    return model, lowest_J


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    a1 = x
    z2 = np.dot(a1, W1)
    a2 = nonlin(z2)
    z3 = np.dot(a2, W2)
    a3 = nonlin(z3)

    if a3.size > 2:
        pred = np.argmax(a3, axis=1)
    else:
        a3.reshape(1,2)
        #print "a3: ", a3
        pred = np.argmax(np.abs(a3))
    return pred


def main():
    conf = Config()
    conf.debug = False
    conf.num_passes = 20000
    conf.num_samples = 3000
    conf.num_train = 2800
    conf.num_test = 200
    conf.reg = 0.0001
    conf.noise = 0.05
    conf.debug = True
    J_history = list()
    alpha = [0.85, 0.9, 0.95, 1]

    model = build_model(conf)
    conf.alpha = 0.95
    X, y = generate_data(conf.num_samples)
    y_v = np.eye(conf.nn_output_dim)[y, :]
    X_train = X[0:conf.num_train, :].copy()
    y_v_train = y_v[0:conf.num_train, :].copy()
    X_pred = X[conf.num_train:conf.num_train + conf.num_test, :].copy()
    y_pred = y[conf.num_train:conf.num_train + conf.num_test].copy()
    model, lowest_J = train_model(model, X_train, y_v_train, conf)
    error = 0.
    for i in xrange(conf.num_test):
        pred = predict(model, X_pred[i, :])
        # print "Prediction idx:[", pred, "] in ", y_v[i, :], " = ",  y_v[i, pred], " - factual: ", y[i]
        if pred != y_pred[i]:
            if conf.debug == True:
                print("Prediction ERROR: %f:%f - %f => %f" % (X_pred[i, 0], X_pred[i, 1], y_pred[i], pred))
            error += 1
    total_error = (error / conf.num_test) * 100
    #J_history.append((a, lowest_J, 100 - total_error))


    visualize(X, y, model)
    print " Accuracy: ", 100 - total_error
    #for j in J_history:
        #print "Learning rate alpha=", j[0], " Cost J=", j[1][0], " Iterations=", j[1][1], " Accuracy", j[2], "%"


    endthis = 0
if __name__ == "__main__":
    main()