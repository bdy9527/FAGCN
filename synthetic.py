import numpy as np
import random
import scipy.sparse as sp
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as acc


def high_dim_gaussian(mu, sigma):
    if mu.ndim > 1:
        d = len(mu)
        res = np.zeros(d)
        for i in range(d):
            res[i] = np.random.normal(mu[i], sigma[i])
    else:
        d = 1
        res = np.zeros(d)
        res = np.random.normal(mu, sigma)
    return res


def generate_uniform_theta(Y, c):
    theta = np.zeros(len(Y), dtype='float')
    for i in range(c):
        idx = np.where(Y == i)
        sample = np.random.uniform(low=0, high=1, size=len(idx[0]))
        sample_sum = np.sum(sample)
        for j in range(len(idx[0])):
            theta[idx[0][j]] = sample[j] * len(idx[0]) / sample_sum
    return theta


def generate_theta_dirichlet(Y, c):
    theta = np.zeros(len(Y), dtype='float')
    for i in range(c):
        idx = np.where(Y == i)
        temp = np.random.uniform(low=0, high=1, size=len(idx[0]))
        sample = np.random.dirichlet(temp, 1)
        sample_sum = np.sum(sample)
        for j in range(len(idx[0])):
            theta[idx[0][j]] = sample[0][j] * len(idx[0]) / sample_sum
    return theta
    
def SBM(sizes, probs, mus, sigmas, noise,
        radius, feats_type='gaussian', selfloops=True):
    # -----------------------------------------------
    #     step1: get c,d,n
    # -----------------------------------------------
    c = len(sizes)
    if mus.ndim > 1:
        d = mus.shape[1]
    else:
        d = 1
    n = sizes.sum()
    all_node_ids = [ids for ids in range(0, n)]
    # -----------------------------------------------
    #     step2: generate Y with sizes
    # -----------------------------------------------
    Y = np.zeros(n, dtype='int')
    for i in range(c):
        class_i_ids = random.sample(all_node_ids, sizes[i])
        Y[class_i_ids] = i
        for item in class_i_ids:
            all_node_ids.remove(item)
    # -----------------------------------------------
    #     step3: generate A with Y and probs
    # -----------------------------------------------
    if selfloops:
        A = np.diag(np.ones(n, dtype='int'))
    else:
        A = np.zeros((n, n), dtype='int')
    for i in range(n):
        for j in range(i + 1, n):
            prob_ = probs[Y[i]][Y[j]]
            rand_ = random.random()
            if rand_ <= prob_:
                A[i][j] = 1
                A[j][i] = 1
    # -----------------------------------------------
    #     step4: generate X with Y and mus, sigmas
    # -----------------------------------------------
    X = np.zeros((n, d), dtype='float')
    for i in range(n):
        mu = mus[Y[i]]
        sigma = sigmas[Y[i]]
        X[i] = high_dim_gaussian(mu, sigma)

    return A, X, Y


def generate(p, q, idx):
    A, X, Y = \
        SBM(sizes=np.array([100, 100]),
        probs=np.array([[p, q], [q, p]]),
        mus=np.array([[-0.5]*20, [0.5]*20]),
        sigmas=np.array([[2]*20, [2]*20]),
        noise=[],
        radius=[],
        selfloops=False)
        
    return A, X, Y
        
        
def calculate(A, X, Y):

    A = sp.coo_matrix(A)
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    rowsum = np.array(A.sum(1)).clip(min=1)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    A = A.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    low = 0.5 * sp.eye(A.shape[0]) + A
    high = 0.5 * sp.eye(A.shape[0]) - A
    low = low.todense()
    high = high.todense()

    low_signal = np.dot(np.dot(low, low), X)
    high_signal = np.dot(np.dot(high, high), X)

    low_MLP = MLPClassifier(hidden_layer_sizes=(16), activation='relu', max_iter=2000)
    low_MLP.fit(low_signal[:100, :], Y[:100])
    low_pred = low_MLP.predict(low_signal[100:, :])

    high_MLP = MLPClassifier(hidden_layer_sizes=(16), activation='relu', max_iter=2000)
    high_MLP.fit(high_signal[:100, :], Y[:100])
    high_pred = high_MLP.predict(high_signal[100:, :])

    return acc(Y[100:], low_pred), acc(Y[100:], high_pred)


low_record = []
high_record = []


for i in range(1, 11):
    q = i * 0.01
    p = 0.05
    low_rec = []
    high_rec = []
    mlp_rec = []
    print(i, p, q)

    for j in range(10):
        A, X, Y = generate(p, q, 0)
        low, high, = calculate(A, X, Y)
        low_rec.append(low)
        high_rec.append(high)
    low_record.append([np.max(low_rec), np.min(low_rec), np.mean(low_rec)])
    high_record.append([np.max(high_rec), np.min(high_rec), np.mean(high_rec)])

print(low_record)
print(high_record)
