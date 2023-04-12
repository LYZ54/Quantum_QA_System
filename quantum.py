import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import torch
import timeit
from torch.autograd import Variable
from math import pi

# example 1
dev = qml.device('default.qubit', wires=2)
qubits = 2
wires = [0, 1]


@qml.qnode(dev, interface='autograd')
def circuit1(phi, theta):
    for i in range(qubits):
        qml.Hadamard(wires=wires[i])

    return qml.expval(qml.PauliZ(0))


qml.drawer.use_style("black_white")
fig, ax = qml.draw_mpl(circuit1)([0.5, 0.1], 0.1)
plt.show()

# example 2
# dev = qml.device('default.qubit', wires=5)
#
#
# @qml.qnode(dev)
# def circuit(data, weights):
#     qml.AmplitudeEmbedding(data, wires=[0, 1, 2], normalize=True)
#     qml.RX(weights[0], wires=0)
#     qml.RY(weights[1], wires=1)
#     qml.RZ(weights[2], wires=2)
#     qml.CNOT(wires=[0, 1])
#     qml.CNOT(wires=[0, 2])
#     return qml.expval(qml.PauliZ(0))
#
#
# rng = np.random.default_rng(seed=42)  # make the results reproducable
# data = rng.random([2 ** 3], requires_grad=True)
# weights = np.array([0.1, 0.2, 0.3], requires_grad=True)
# qml.grad(circuit)(data, weights=weights

# example 3
# dev = qml.device('default.qubit', wires=1)
#
#
# @qml.qnode(dev, interface='autograd')
# def circuit1(params1):
#     qml.RX(params1[0], wires=0)
#     qml.RY(params1[1], wires=0)
#     return qml.expval(qml.PauliZ(0))
#
#
# def cost(params):
#     return circuit1(params)
#
#
# # dcircuit = qml.grad(circuit1, argnum=0)
# # print(dcircuit([0.10, 0.11]))
#
# init_params = np.array([0.11, 0.12], requires_grad=True)
# # print(cost(init_params))
#
# opt = qml.GradientDescentOptimizer(stepsize=0.4)
#
# steps = 100
#
# params = init_params
#
# for i in range(steps):
#     params = opt.step(cost, params)
#
#     if (i+1) % 5 == 0:
#         print("cost after {: 5d} steps is {: .7f}".format(i+1, cost(params)))
#
# print("Optimized rotation angles is {}".format(params))

# example 4
# dev = qml.device('default.qubit', wires=3)
#
#
# @qml.qnode(dev, diff_method='parameter-shift', interface='autograd')
# def circuit(params):
#     qml.RX(params[0], wires=0)
#     qml.RY(params[1], wires=1)
#     qml.RZ(params[2], wires=2)
#     qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern='ring')
#
#     qml.RX(params[3], wires=0)
#     qml.RY(params[4], wires=1)
#     qml.RZ(params[5], wires=2)
#     qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern='ring')
#
#     return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))
#
#
# params = np.random.random([6], requires_grad=True)
# # print('params:', params)
# # print('expval:', circiut(params))
# #
# # qml.draw_mpl(circiut, decimals=2)(params)
# # plt.show()
#
# # def parameter_shift_term(qnode, params, i):
# #     shifted = params.copy()
# #     shifted[i] += np.pi / 2
# #     forward = qnode(shifted)  # forward evaluation
# #
# #     shifted[i] -= np.pi
# #     backward = qnode(shifted)  # backward evaluation
# #
# #     return 0.5 * (forward - backward)
# #
# #
# # # gradient with respect to the first parameter
# # print(parameter_shift_term(circuit, params, 0))
#
# grad = qml.gradients.param_shift(circuit)
# print((grad(params)))
#
# grad2 = qml.grad(circuit)
# print(grad2(params))

# example 5
# dev = qml.device("default.qubit", wires=4)
#
#
# @qml.qnode(dev, diff_method="backprop", interface="autograd")
# def circuit(params):
#     qml.StronglyEntanglingLayers(params, wires=[0, 1, 2, 3])
#     return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))
#
#
# # initialize circuit parameters
# param_shape = qml.StronglyEntanglingLayers.shape(n_wires=4, n_layers=15)
# params = np.random.normal(scale=0.1, size=param_shape, requires_grad=True)
# # print(params.size)
# # print(circuit(params))
#
# # qml.draw_mpl(circuit)(params)
# # plt.show()
# reps = 3
# num = 10
# times = timeit.repeat("circuit(params)", globals=globals(), number=num, repeat=reps)
# forward_time = min(times) / num
#
# print(f"Forward pass (best of {reps}): {forward_time} sec per loop")
#
# # create the gradient function
# grad_fn = qml.grad(circuit)
#
# times = timeit.repeat("grad_fn(params)", globals=globals(), number=num, repeat=reps)
# backward_time = min(times) / num
#
# print(f"Gradient computation (best of {reps}): {backward_time} sec per loop")

# example 6
# np.random.seed(42)
#
# # we generate a three-dimensional random vector by sampling
# # each entry from a standard normal distribution
# v = np.random.normal(0, 1, 3)
#
# # purity of the target state
# purity = 0.66
#
# # create a random Bloch vector with the specified purity
# bloch_v = Variable(
#     torch.tensor(np.sqrt(2 * purity - 1) * v / np.sqrt(np.sum(v ** 2))),
#     requires_grad=False
# )
#
# # array of Pauli matrices (will be useful later)
# Paulis = Variable(torch.zeros([3, 2, 2], dtype=torch.complex128), requires_grad=False)
# Paulis[0] = torch.tensor([[0, 1], [1, 0]])
# Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])
# Paulis[2] = torch.tensor([[1, 0], [0, -1]])
#
# # number of qubits in the circuit
# nr_qubits = 3
# # number of layers in the circuit
# nr_layers = 2
#
# # randomly initialize parameters from a normal distribution
# params = np.random.normal(0, np.pi, (nr_qubits, nr_layers, 3))
# params = Variable(torch.tensor(params), requires_grad=True)
#
#
# # a layer of the circuit ansatz
# def layer(params, j):
#     for i in range(nr_qubits):
#         qml.RX(params[i, j, 0], wires=i)
#         qml.RY(params[i, j, 1], wires=i)
#         qml.RZ(params[i, j, 2], wires=i)
#
#     qml.CNOT(wires=[0, 1])
#     qml.CNOT(wires=[0, 2])
#     qml.CNOT(wires=[1, 2])
#
#
# dev = qml.device("default.qubit", wires=3)
#
#
# @qml.qnode(dev, interface="torch")
# def circuit(params, A):
#     # repeatedly apply each layer in the circuit
#     for j in range(nr_layers):
#         layer(params, j)
#
#     # returns the expectation of the input matrix A on the first qubit
#     return qml.expval(qml.Hermitian(A, wires=0))
#
#
# # cost function
# def cost_fn(params):
#     cost = 0
#     for k in range(3):
#         cost += torch.abs(circuit(params, Paulis[k]) - bloch_v[k])
#
#     return cost
#
#
# # set up the optimizer
# opt = torch.optim.Adam([params], lr=0.1)
#
# # number of steps in the optimization routine
# steps = 200
#
# # the final stage of optimization isn't always the best, so we keep track of
# # the best parameters along the way
# best_cost = cost_fn(params)
# best_params = np.zeros((nr_qubits, nr_layers, 3))
#
# print("Cost after 0 steps is {:.4f}".format(cost_fn(params)))
#
# # optimization begins
# for n in range(steps):
#     opt.zero_grad()
#     loss = cost_fn(params)
#     loss.backward()
#     opt.step()
#
#     # keeps track of best parameters
#     if loss < best_cost:
#         best_cost = loss
#         best_params = params
#
#     # Keep track of progress every 10 steps
#     if n % 10 == 9 or n == steps - 1:
#         print("Cost after {} steps is {:.4f}".format(n + 1, loss))
#
# # calculate the Bloch vector of the output state
# output_bloch_v = np.zeros(3)
# for l in range(3):
#     output_bloch_v[l] = circuit(best_params, Paulis[l])
#
# # print results
# print("Target Bloch vector = ", bloch_v.numpy())
# print("Output Bloch vector = ", output_bloch_v)


# n_wires = 3
# dev = qml.device('default.qubit', wires=n_wires)
#
#
# @qml.qnode(dev)
# def circuit(weights):
#     qml.BasicEntanglerLayers(weights=weights, wires=range(n_wires), rotation=qml.RY)
#     return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]
#
#
# X = [[pi, pi, pi]]
# print(qml.draw(circuit, expansion_strategy="device")(X))
# qml.QNode()

# training_data = [
#     # Tags are: DET - determiner; NN - noun; V - verb
#     # For example, the word "The" is a determiner
#     ("the dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
#     ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
# ]
# word_to_ix = {}
#
# # For each words-list (sentence) and tags-list in each tuple of training_data
# for sent, tags in training_data:
#     for word in sent:
#         if word not in word_to_ix:  # word has not been assigned an index yet
#             word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
# print(word_to_ix)


input = np.array([1, 2, 3, 4])
input2 = np.square(input)
print(input2)
