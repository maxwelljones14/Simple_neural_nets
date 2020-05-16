import numpy as np
def perceptron(A,w,b,x):
    return A((w*x) + b)
def sigmoid(z):
    return 1 / (1 + np.e ** (-1 * z))
def sigmoid_deriv(z):
    return (1 / (1 + np.e ** (-1 * z))) * (1 - (1 / (1 + np.e ** (-1 * z))))
def step(x):
    if x > 0:
        return 1
    return 0
def task_four(num):
    w_0 = np.matrix(2 * np.random.rand(2, num) - 1)
    b_1 = np.matrix(2 * np.random.rand(1, num) - 1)
    w_1 = np.matrix(2 * np.random.rand(num, 1) - 1)
    b_2 = np.matrix(2 * np.random.rand(1, 1) - 1)

    vec_1 = np.vectorize(sigmoid)
    vec_2 = np.vectorize(sigmoid_deriv)

    lambda_val = 0.1

    count2 = 0
    count = 51

    while count > 50:
        count = 0
        f = open("10000_pairs.txt", "r")
        for r in f:
            list = []
            for index, item in enumerate(r.split(" ")):
                if index ==0:
                    list.append(float(item))
                else:
                    list.append(float(item[0:len(item)-2:]))
            y = 0
            if list[0]**2 + list[1]**2 >= 1:y=1

            a_0 = np.matrix(list)
            a_1 = vec_1((a_0 * w_0) + b_1)
            a_2 = vec_1((a_1 * w_1) + b_2)

            error = 0.5 * (np.linalg.norm(y - a_2)) ** 2
            #print("error before back prop: ", error)

            delta_2 = np.multiply(vec_2((a_1 * w_1) + b_2), y - a_2)
            delta_1 = np.multiply(vec_2((a_0 * w_0) + b_1), delta_2 * w_1.transpose())

            w_0 = w_0 + lambda_val * (a_0.transpose()) * delta_1
            b_1 = b_1 + lambda_val * delta_1

            w_1 = w_1 + lambda_val * (a_1.transpose()) * delta_2
            b_2 = b_2 + lambda_val * delta_2

        f.close()
        f = open("10000_pairs.txt", "r")

        for v in f:
            list = []
            for index, item in enumerate(v.split(" ")):
                if index ==0:
                    list.append(float(item))
                else:
                    list.append(float(item[0:len(item)-2:]))
            y = 0
            if list[0]**2 + list[1]**2 >= 1:y=1

            a_0 = np.matrix(list)
            a_1 = vec_1((a_0 * w_0) + b_1)
            a_2 = vec_1((a_1 * w_1) + b_2)
            if (a_2.item(0) < 0.5 and y == 1) or (a_2.item(0) > 0.5 and y == 0):
                count+=1
        f.close()
        count2+=1
        if count2>=8:
            lamda_val = 0.5 #lambda_val/1.5+.01
        print(count)



task_four(8)

