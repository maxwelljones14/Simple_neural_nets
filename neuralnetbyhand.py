import random
from math import tanh, cosh, sin, cos, pi
trainingSet = [[(0,0),0], [(1, 0), 1], [(1, 1), 0], [(0, 1), 1]]
xVals = [[0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]
numLayers = 8

def createNet(num):
    dict = {}
    for i in range(0,3):
        for j in range(num):
            dict[(i, j)] = random.randint(-1000, 1000)/1000
    l = []
    for i in range(num):
        l.append(random.randint(-1000,1000)/1000)
    return [dict, l]
def calculate(layers, network, first, second):

    inputlist = [first,second,1]
    xVals[0] = [first, second, 1]
    dict = network[0]
    middleLayer = []
    for i in range(layers):
        middleLayer.append(0)
    for key in dict:
        middleLayer[key[1]] += inputlist[key[0]]*dict[key]
    for j in range(layers):
        middleLayer[j] = tanh(middleLayer[j])
    l = network[1]
    ans = 0
    for w in range(layers):
        ans += middleLayer[w]*l[w]
    for y in range(layers):
        xVals[1][y] = 0
    for x in range(layers):
        xVals[1][x] = middleLayer[x]





    return ans
def derivative(x):
    return (1/cosh(x))**2
def backPropagate(layers, network, error):
    newError = []
    for i in range(layers):
       newError.append(derivative(xVals[1][i])*network[1][i]*error)
    for j in range(layers):
        network[1][j] += xVals[1][j]*error*0.1
    for key in network[0]:
        network[0][key] += xVals[0][key[0]]*newError[key[1]]*0.1
    return network

def trainNet(layers, network):
    # for i in range(10000):
    #     for key in trainingSet:
    #         error = calculate(numLayers, network, key[0][0], key[0][1]) - key[1]
    #         backPropagate(numLayers, network, -error)

    for i in range(40000):
        rand1, rand2 = random.randint(-125,125)/100,random.randint(-125,125)/100
        distance =  (rand1**2 + rand2**2)**0.5
        error = calculate(numLayers, net, rand1, rand2) - distance
        backPropagate(layers, network, -error)

def testNet(network):
    count = 0
    for i in range(1000):
        rand1, rand2 = random.randint(-125,125)/100, random.randint(-125,125)/100
        distance = (rand1**2 + rand2**2)**0.5
        guess = calculate(numLayers, network, rand1, rand2)
        if  ( guess < 1 and distance < 1) or (guess > 1 and distance > 1):
            count+=1

    return count

net = createNet(numLayers)
trainNet(numLayers, net)
print("percent correct:"+ str(testNet(net)/10) + "%")
print("network:")
print(net)
