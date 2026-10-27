import numpy as np
import random

class Perceptron:
    """A single neuron with different activation functions.
       Attributes:
          inputs: The number of inputs in the perceptron.
          bias:   The bias term. By default it's 0.0.
          z: store preactivation value (needed for RELU in the backpropagation process) """

    def __init__(self, inputs, bias = 0.0):
        """Return a new Perceptron object with the specified number of inputs and the bias.""" 
        # Init for sigmoid 
        # self.weights = (np.random.rand(inputs) * 2) - 1
        # Gaussian Xavier initalization for tanh
        # self.weights = np.random.randn(inputs) * np.sqrt(1 / inputs)
        # He init for RELU   
        self.weights = np.random.randn(inputs) * np.sqrt(2 / inputs)
        self.bias = bias

    def run(self, x, activation):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(x, self.weights) + self.bias
        self.z = x_sum  # for RELU
        if (activation == 'Sigmoid'):
            return self.sigmoid(x_sum)
        elif (activation == 'Tanh'):
            return self.tanh(x_sum)
        elif (activation == 'Relu'):
            return self.relu(x_sum)
        else: # identity/linear/ no activation
            return x_sum

    def set_weights(self, w_init):
        """Set the weights. w_init is a python list with the weights."""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1/(1+np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        # original formula 
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

        # piecewise tanh
        # Use it only if activations can reach very large values.
        # Standard tanh is safe for normal neural network ranges.
        #if x >= 0 :
        #    e = np.exp(-2 * x)
        #    return (1 - e)/(1 + e)
        #else:
        #    e = np.exp(2 * x)
        #    return (e - 1 )/ (e + 1)

class MultiLayerPerceptron:     
    """A multilayer perceptron class that uses the Perceptron class.
       Attributes:
          layers:  A python list with the number of elements per layer.
          eta:     The learning rate.
       The last hidden layer and the output layer use identity activation so the
       network can learn arithmetic functions without nonlinear activations
       distorting or squashing the values. """

    def __init__(self, layers, eta = 0.0003):
        """Return a new MLP object with the specified parameters.""" 
        self.layers = np.array(layers, dtype=object)
        self.eta = eta
        self.network = [] # The list of lists of neurons
        self.values = []  # The list of lists of output values
        self.d = []       # The list of lists of error terms (lowercase deltas)

        for i in range(len(self.layers)):
            self.values.append([])
            self.d.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:      #network[0] is the input layer, so it has no neurons
                for j in range(self.layers[i]): 
                    self.network[i].append(Perceptron(inputs = self.layers[i-1]))
        
        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)
        self.d = np.array([np.array(x) for x in self.d], dtype=object)

    def set_weights(self, w_init):
        """Set the weights. 
           w_init is a 3D list with the weights for all but the input layer."""
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])

    def print_weights(self):
        print()
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                print("Layer", i+1, "Neuron", j, self.network[i][j].weights)
        print()

    def run(self, x):
        """Feed a sample x into the MultiLayer Perceptron."""
        x = np.array(x, dtype=float)
        self.values[0] = x
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                if (i == (len(self.network) - 1) ):  # output layer activation function is identity
                    self.values[i][j] = self.network[i][j].run(self.values[i-1], 'Identity')
                elif (i == (len(self.network) - 2) ): # last hidden layer uses identity
                    self.values[i][j] = self.network[i][j].run(self.values[i-1], 'Identity')                        
                else:
                    self.values[i][j] = self.network[i][j].run(self.values[i-1], 'Relu')
        return self.values[-1]
    
    def bp(self, x, y):
        """Run a single (x,y) pair with the backpropagation algorithm."""
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        # Feed a sample to the network
        results = self.run(x)
        
        # Calculate the MSE
        MSE = 0.0
        for i in range(len(y)):
            MSE += (y[i] - results[i])**2
        MSE = MSE / len(y) 

        # Calculate the output error terms
        for i in range(len(y)):
            self.d[-1][i] = y[i] - results[i]

        #  Calculate the error term of each unit on each layer
        for i in reversed(range(1, len(self.network)-1)):
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i+1]): 
                    fwd_error += self.d[i+1][k]*self.network[i+1][k].weights[h]

                # Sigmoid  
                # self.d[i][h] = self.values[i][h] * (1 - self.values[i][h]) * fwd_error
                #Tanh
                # if (i == (len(self.network)-2)): # identity(linear) output layer
                #     self.d[i][h] = fwd_error    
                # else:
                #     self.d[i][h] = (1 - (self.values[i][h])**2 ) * fwd_error
                
                # Relu derivative (If this layer uses Relu)
                if i < len(self.network) - 2:
                    if self.network[i][h].z > 0:
                        self.d[i][h] = fwd_error
                    else:
                        self.d[i][h] = 0.0
                # If this layer uses Identity
                else:
                    self.d[i][h] = fwd_error

        # Calculate the deltas and update the weights
        for i in range(1, len(self.layers)):
            for j in range(self.layers[i]):
                # Update the weights
                for k in range(self.layers[i-1]):
                    delta = self.eta * self.d[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k] +=delta
                # Update the neuron's bias
                self.network[i][j].bias += self.eta * self.d[i][j]
        return MSE

# sample creation 1
samples = [
    ([1,2], [1]),
    ([2,2], [4]),
    ([3,2], [9]),
    ([4,2], [16]),
    ([5,2], [25]),
    ([6,2], [36]),
    ([7,2], [49]),
    ([8,2], [64]),
    ([9,2], [81]),
    ([10,2], [100]),

    ([2,3], [8]),
    ([3,3], [27]),
    ([4,3], [64]),
    ([5,3], [125]),
    ([6,3], [216]),
    ([7,3], [343]),
    ([8,3], [512]),
    ([9,3], [729]),
    ([10,3], [1000]),

    ([2,4], [16]),
    ([3,4], [81]),
    #([4,4], [256]),
    ([5,4], [625]),

    ([2,5], [32]),
    ([3,5], [243]),
    ([4,5], [1024]),

    ([2,6], [64]),
    ([3,6], [729]),

    ([2,7], [128]),
    ([3,7], [2187]),

    ([2,8], [256]),
    ([2,9], [512]),
    ([2,10], [1024]),

    ([3,1], [3]),
    ([4,1], [4]),
    ([5,1], [5]),
    ([6,1], [6]),
    ([7,1], [7]),
    ([8,1], [8]),
    ([9,1], [9]),
    ([10,1], [10])
]

# sample creation 2
X = np.random.uniform(0.001, 10.0, (100, 2)).astype(dtype=np.float64)
Y = np.power(X[:,0], X[:,1]).astype(dtype=np.float64)                  # target: x^y
samples = list(zip(X,Y[:, None]))

#test code
mlp = MultiLayerPerceptron(layers=[2,32,32,1], eta = 0.0005)
print("\nTraining Neural Network \n")

for i in range(3000):
    random.shuffle(samples)

    mse = 0.0
    for x, y in samples:
        y_train = np.log(y)
        mse +=mlp.bp(x, y_train)
    mse /= len(samples)
    
    if(i%500 == 0):
        print (i,     mse)

mlp.print_weights()
    
print("MLP:")

print ("3 3 = {0:.10f}".format(np.exp(mlp.run([3,3])[0])))
print ("10 2 = {0:.10f}".format(np.exp(mlp.run([10,2])[0])))
print ("7 2 = {0:.10f}".format(np.exp(mlp.run([7,2])[0])))
print ("4 4 = {0:.10f}".format(np.exp(mlp.run([4,4])[0])))
print ("2 11 = {0:.10f}".format(np.exp(mlp.run([2,11])[0])))
print ("10 5 = {0:.10f}".format(np.exp(mlp.run([10,5])[0])))