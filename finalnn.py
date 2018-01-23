import numpy as np
import random
import copy
import math
# import matplotlib.pyplot as plt

class NN:

	def __init__(self, inputsIn, outputsIn, schemaIn, learningRate, activationFunction):
		self.inputs = inputsIn
		self.desiredOutputs = outputsIn
		self.schema = schemaIn
		self.activations = [self.inputs]+[0]*(len(self.schema)-1)
		for layer in range(1, len(self.schema)):
			thisLayer = []
			self.activations[layer] = thisLayer
		self.weights = []
		for layer in range(len(self.schema)-1):
			# self.weights += [np.random.rand(self.schema[layer], self.schema[layer+1])*np.random.rand]
			self.weights += [np.random.uniform(-1.0, 1.0, size=(self.schema[layer], self.schema[layer+1]))]
		self.activationFunction = activationFunction
		self.learningRate = learningRate

	def __repr__(self):
		for layer in range(len(self.weights)):
			if (layer == 0):
				print("Layer 0")
				print(self.activations[0])
			print("Weights "+str(layer))
			print(np.mat(self.weights[layer]))
			print()
			print("Layer "+str(layer+1))
			print(np.mat(self.activations[layer+1]))
		return ''

	def predict(self, data):
		prediction = data
		for layer in range(len(self.weights)):
			prediction = self.activationFunction(np.dot(prediction, self.weights[layer]))
		return prediction

	def forwardPropagation(self):
		#Change network values during propagation
		for layer in range(len(self.weights)):
			# self.activations[layer+1] = np.vectorize(self.activationFunction)(np.dot(self.activations[layer], self.weights[layer]))  #Use this if activation function is not numpy compatible
			self.activations[layer+1] = self.activationFunction(np.dot(self.activations[layer], self.weights[layer]))

	def backpropagation(self):
		slopes = []
		errors = []
		deltas = []
		for layer in range(1, len(self.activations)): #For every layer in the network (progressing left to right)
			tempSlope = self.activationFunction(self.activations[-layer], True) #Find the derivate slopes for each node at this layer (progressing right to left)
			slopes.insert(0, tempSlope) #Prepend into list of slopes
		
		#Compute output layer
		errors = [self.desiredOutputs-self.activations[-1]] #Compute L1 error for each output node, add to errors
		deltas = [errors[-1] * slopes[-1]] #Multiply by appropriate slope values, add to deltas

		#Compute hidden layers
		for layer in range(len(self.weights)-1, 0, -1): #For each hidden layer in network (progressing right to left)
			tempError = deltas[0].dot(self.weights[layer].T) #Compute hidden error for each node in this layer, prepend to errors
			errors.insert(0, tempError)
			tempDelta = errors[0] * slopes[layer-1] #Compute deltas for each hidden layer, prepend to deltas
			deltas.insert(0, tempDelta)

		#Adjust weights
		for layer in range(len(self.weights)-1, -1, -1):
			self.weights[layer] += np.array(self.activations[layer]).T.dot(deltas[layer]) * self.learningRate #Update weights

	def train(self):
		self.forwardPropagation()
		self.backpropagation()

	def trainFor(self, reps, verbose=True):
		for epoch in range(reps):
			self.train()
			if 100*(epoch / reps) % 10 == 0 and verbose:
				error = np.power(np.mean(np.abs(self.desiredOutputs - self.activations[-1])), 2)
				print(str(100*epoch/reps)+"% - Error: "+str(error))
		error = np.power(np.mean(np.abs(self.desiredOutputs - self.activations[-1])), 2)
		if verbose:
			print("100.0% - Error: "+str(error))
		return error

	def trainTo(self, errorThreshold, testingData=[]):
		errorRate = 1
		totalEpochs = 0
		while True:
			if errorRate <= errorThreshold:
				break
			self.trainFor(100, False)
			totalEpochs += 100
			if totalEpochs % 1000 == 0:
				if testingData == []:
					errorRate = np.power(np.mean(np.abs(np.array(testingData) - self.activations[-1])), 2)
				else:	
					errorRate = np.power(np.mean(np.abs(self.desiredOutputs - self.activations[-1])), 2)
				print("Error rate: "+str(round(errorRate, 3)))
		print("Total epochs: "+str(totalEpochs))
		return errorRate

	def testMode(self):
		print("======Test mode======")
		answer = ""
		while answer != "quit":
			answer = input("Enter parameters: ")
			answer = [float(n) for n in answer.split(',')]
			forwardProp = nn.predict(answer)
			if forwardProp is not None:
				print(forwardProp)

#Logistic sigmoid
def sigmoid(x, derivative=False):
	if derivative == True:
		return x*(1-x)
	else:
		return 1/(1+np.exp(-1*x))

#Arctan function
def tanh(x, derivative=False):
	if derivative == True:
		return 1 - np.power(np.tanh(np.longdouble(x)), 2)
	else:
		return np.tanh(np.longdouble(x))

#Linear activation past threshold 0
# def relu(x, derivative=False):
# 	if derivative == True:
# 		if x == 0:
# 			return 0.5
# 		else:
# 			return 1
# 	else:
# 		return np.maximum(0, x)

#Differentiable approximation of relu
# def softplus(x, derivative=False):
	# x = np.longdouble(x)
	# if derivative == True:
	# 	return 1/(1+np.exp(-x))
	# else:
	# 	return np.log(np.longdouble((1+np.exp(x))))

def splitData(folds, data):
	shuffledData = copy.deepcopy(data)
	np.random.shuffle(shuffledData)
	segments = list(map(list, zip(*[iter(shuffledData)]*folds))) 
	if divmod(len(shuffledData), folds)[1] != 0:
		segments += [data[folds*divmod(len(shuffledData), folds)[0]:]]
	return segments

def crossValidate(segmentSize, trainingLength, data, outputVectorDimension, schema, learningRate, activationFunction):
	np.set_printoptions(precision=5, suppress=True)
	segmentedData = splitData(segmentSize, data)
	testingError = []
	trainingError = []
	for k in range(len(segmentedData)):
		testing = segmentedData[k]
		training = []
		for i in range(len(segmentedData)):
			if i != k:
				training += segmentedData[k]
		inputs = np.array([tuple(line[:-outputVectorDimension]) for line in training])
		outputs = np.array([line[-outputVectorDimension:] for line in training])
		nn = NN(inputs, outputs, schema, learningRate, sigmoid)
		print("Simulation "+str(k+1)+"/"+str(len(segmentedData)))
		print("="*(len("Simulation "+str(k+1)+"/"+str(len(segmentedData)))))
		print("Training...")
		trainingErrorTerm = nn.trainFor(trainingLength)
		trainingError += [trainingErrorTerm]
		print("Finished!")
		print("Training Error: "+str(trainingErrorTerm))
		print("Running test data...")
		testInputs = np.array([[row[:-outputVectorDimension]] for row in testing])
		testOutputs = np.array([[row[-outputVectorDimension:]] for row in testing])
		subtestErrorSum = 0
		for i in range(len(testInputs)):
			result = nn.predict(testInputs[i])
			subtestError = np.mean(np.abs(np.array(testOutputs[i]) - np.array(result))) ** 2
			if len(testInputs) < 11:
				print("Answer: "+str(testOutputs[i])+", Prediction: "+str(result))
			else:
				if 3 > i or i > len(testInputs)-4:
					print("Answer: "+str(testOutputs[i])+", Prediction: "+str(result))
				if i == 4:
					print("\t......\n\t......\n\t......\n\t......")
			subtestErrorSum += subtestError
		testingError += [subtestErrorSum]
		print("Finished!")
		print("Testing Error: "+str(subtestErrorSum))
		print("="*(len("Simulation "+str(k+2)+"/"+str(len(segmentedData)))))
	print("Total training error: "+str(sum(trainingError))+" over "+str(len(segmentedData))+" trials.")
	print("Total testing error: "+str(sum(testingError))+" over "+str(len(segmentedData))+" trials.")


#XOR NN
file = open("xorData.txt", "r")
rawData = file.read().split('\n')
data = [[float(string) for string in (''.join(line)).split(',')] for line in rawData]
schema = [2,3,1]
learningRate = 9
outputVectorDimension = 1
activationFunction = sigmoid
inputs = np.array([tuple(line[:-outputVectorDimension]) for line in data])
outputs = np.array([line[-outputVectorDimension:] for line in data])
nn = NN(inputs, outputs, schema, learningRate, activationFunction)
print("\n=====XOR NN=====")
print("Final error: "+str(nn.trainFor(5000)))
input('Enter to continue on to next test...')

#Test NN
file = open("testData.txt", "r")
rawData = file.read().split('\n')
data = [[float(string) for string in (''.join(line)).split(',')] for line in rawData]
schema = [4,3,1]
learningRate = 7
outputVectorDimension = 1
activationFunction = sigmoid
inputs = np.array([tuple(line[:-outputVectorDimension]) for line in data])
outputs = np.array([line[-outputVectorDimension:] for line in data])
nn = NN(inputs, outputs, schema, learningRate, activationFunction)
print("\n=====Test NN=====")
print("Final error: "+str(nn.trainFor(2000)))
input('Enter to continue on to next test...')

#Iris NN
# https://archive.ics.uci.edu/ml/datasets/iris
file = open("IrisData.txt", "r")
rawData = file.read().split('\n')
data = [[float(string) for string in (''.join(line)).split(',')] for line in rawData]
schema = [4, 7, 3]
learningRate = 0.05
outputVectorDimension = 3
activationFunction = sigmoid
segmentSize = 10
trainingLength = 10000
print("\n=====Iris NN=====")
crossValidate(segmentSize, trainingLength, data, outputVectorDimension, schema, learningRate, activationFunction)
input('Enter to continue on to next test...')

#Earthquake Clean NN
#see AIMA
file = open("earthquake-clean.data.txt", "r")
rawData = file.read().split('\n')
data = [[float(string) for string in (''.join(line)).split(',')] for line in rawData]
schema = [2, 4, 3, 1]
learningRate = 0.01
outputVectorDimension = 1
activationFunction = sigmoid
segmentSize = 7
trainingLength = 20000
print("\n=====Earthquake Clean NN=====")
crossValidate(segmentSize, trainingLength, data, outputVectorDimension, schema, learningRate, activationFunction)
input('Enter to continue on to next test...')

#Earthquake Noisy NN
#see AIMA
file = open("earthquake-noisy.data.txt", "r")
rawData = file.read().split('\n')
data = [[float(string) for string in (''.join(line)).split(',')] for line in rawData]
schema = [2, 4, 3, 1]
learningRate = 0.01
outputVectorDimension = 1
activationFunction = sigmoid
segmentSize = 10
trainingLength = 20000
print("\n=====Earthquake Noisy NN=====")
crossValidate(segmentSize, trainingLength, data, outputVectorDimension, schema, learningRate, activationFunction)
input('Enter to continue on to next test...')

#Car NN
#http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
file = open("CarData.txt", "r")
rawData = file.read().split('\n')
data = [[float(string) for string in (''.join(line)).split(',')] for line in rawData]
schema = [6, 5, 3, 4]
learningRate = 0.005
outputVectorDimension = 4
activationFunction = sigmoid
segmentSize = 100
trainingLength = 20000
print("\n=====Car NN=====")
crossValidate(segmentSize, trainingLength, data, outputVectorDimension, schema, learningRate, activationFunction)


# CODE USED TO GENERATE DIAGRAMS
# ==============================

# fig, axes = plt.subplots(1, 1)
# bar_width = 0.5
# ind = np.arange(1-bar_width/2, max(len(testingError),len(trainingError))-1)
# axes.bar(ind, testingError[:-1], bar_width, color='g', label='Test Error')
# axes.bar(ind, trainingError[:-1], bar_width, color='b', label='Training Error')

# axes.legend(loc='best', frameon=False)
# plt.title('Error of Cross Validation Simulations')
# plt.xlabel('Simulation')
# plt.ylabel('L2 Error')
# plt.show()

# ============

# segmentedData = splitData(segmentSize, data)
# testingError = []
# trainingError = []
# k = 0
# testing = segmentedData[k]
# training = []
# for i in range(len(segmentedData)):
# 	if i != k:
# 		training += segmentedData[k]
# inputs = np.array([tuple(line[:-outputVectorDimension]) for line in training])
# outputs = np.array([line[-outputVectorDimension:] for line in training])
# testInputs = np.array([[row[:-outputVectorDimension]] for row in testing])
# testOutputs = np.array([[row[-outputVectorDimension:]] for row in testing])
# nn = NN(inputs, outputs, schema, learningRate, sigmoid)

# for j in range(1000):
# 	trainingError += [nn.trainFor(1, False)]
# 	subtestErrorSum = 0
# 	for i in range(len(testInputs)):
# 		result = nn.predict(testInputs[i])
# 		subtestError = np.mean(np.abs(np.array(testOutputs[i]) - np.array(result))) ** 2
# 		subtestErrorSum += subtestError
# 	testingError += [subtestErrorSum]

# fig, axes = plt.subplots(1, 1)
# ind = np.arange(0, 1000)
# axes.semilogx(ind, testingError, color='g', label='Test Error')
# axes.semilogx(ind, trainingError, color='b', label='Training Error')

# axes.legend(loc='best', frameon=False)
# plt.title('Error During Training')
# plt.xlabel('log(Epoch)')
# plt.ylabel('L2 Error')
# plt.grid(True)
# plt.show()