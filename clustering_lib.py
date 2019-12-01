import numpy as np
from numpy import genfromtxt
import pandas as pd
from random import randrange
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def randPU(n, K):
    '''

    :param n: len(matrix)
    :param K: number of cluster
    :return: misleading matrix
    '''
    # all 0 matrix
    Uzero = []
    for k in range(0, n):
        Uzero.append([0] * K)

    # random matrix of appartenence
    for row in range(0, n):
        Uzero[row][randrange(K)] = 1

    return Uzero


def diagSU(su):
    '''

    :param su: sum of U_0 columns
    :return: diagonal matrix
    '''
    Uzero = []
    for j in range(0, len(su)):
        Uzero.append([0] * len(su))

    for j in range(0, len(su)):
        Uzero[j][j] = 1 / su[j]

    return Uzero

def diagonalize(su):
    '''

    :param su: diagonal of a matrix
    :return: diagonal matrix
    '''

    Uzero = []
    for j in range(0, len(su)):
        Uzero.append([0] * len(su))

    for j in range(0, len(su)):
        Uzero[j][j] = su[j]
    Uzero = pd.DataFrame(Uzero)
    return Uzero


def mse(Xrow, Xmean0row1):
    min_dif = 0
    for r in range(0, len(Xrow)):
        min_dif += (Xrow[r] - Xmean0row1[r]) ** 2

    MSE = min_dif / len(Xrow)
    return MSE


def mse2(Xrow, Xmean0row1):
    min_dif = 0
    for r in range(0, len(Xrow)):
        min_dif += (Xrow[r] - Xmean0row1[r]) ** 2

    MSE = min_dif / len(Xrow)
    return MSE


def costf(X, Xmean_ott, U):
    cost = 0
    for i in range(0, len(X)):
        for j in range(0, len(U.iloc[0])):
            if U.iloc[i, j] == 1:
                cost += mse(X.iloc[i], Xmean_ott.iloc[j])
    return cost


def kmeans(X, K, Rndstart):
    '''

    :param X: data Matrix
    :param K: Number of cluster of the partition
    :param Rndstart: number of random start
    :return:
    '''

    maxiter = 100
    n = len(X)
    j = len(X.iloc[0])
    epsilon = 0.00001

    # find the best solution in a fixed number of random start partitions
    for loop in range(0, Rndstart):

        # initial partition U_0 is given
        U_0 = pd.DataFrame(randPU(n, K))

        # column frequency = random cluster
        sum_col = []
        for r in range(0, K):
            sum_col.append(sum(U_0[r]))

        # 1/su on diagonal of a NxN matrix
        su_diag = diagSU(sum_col)

        # given U, compute Xmean initial (centroids)
        Ut = U_0.transpose()
        dot1 = pd.DataFrame(np.dot(su_diag, Ut))

        Xmean0 = round(pd.DataFrame(np.dot(dot1, X)), 4)

        U = []
        for r in range(0, n):
            U.append([0] * K)

        for iter in range(1, maxiter):
            # given Xmean, assign each units to the closest cluster

            for r in range(0, n):

                min_dif = mse(X.iloc[r], Xmean0.iloc[0])

                posmin = 0
                for j in range(1, K):
                    dif = mse(X.iloc[r], Xmean0.iloc[j])
                    if dif < min_dif:
                        min_dif = dif
                        posmin = j
                U[r][posmin] = 1

            U = pd.DataFrame(U)

            # given a partition of units, so given U, compute Xmean uptaded (centroids)

            # update sum_col
            sum_col = []
            for t in range(0, K):
                sum_col.append(sum(U[t]))

            ## RARE CASE (BUT POSSIBLE) #############################################################
            # if there is some empty cluster we must split the cluster with max sum_col
            while sum([sum_col[h] == 0 for h in range(0, len(sum_col))]) > 0:  # some cluster is empty

                p1 = min(sum_col)
                p2 = max(sum_col)

                # select min column (empty cluster)
                for j in range(0, len(sum_col)):
                    if p1 == sum_col[j]:
                        c1 = j

                # select max column (cluster) for split its points to empty cluster
                for k in range(0, len(sum_col)):
                    if p2 == sum_col[k]:
                        c2 = k

                # list of units in max column (cluster)
                ind = []
                for row in range(0, len(U)):
                    if int(U.iloc[row, c2]) == 1:
                        ind.append(row)

                # split max cluster
                ind2 = []
                for row in range(0, p2 // 2):
                    ind2.append(row)

                for row in range(0, len(ind2)):
                    U.iloc[row, c1] = 1
                    U.iloc[row, c2] = 0

                sum_col = []
                for q in range(0, K):
                    sum_col.append(sum(U[q]))
            #################################################################################################

            # give U compute centroids
            _U = U.transpose()
            _dot1 = pd.DataFrame(np.dot(diagSU(sum_col), _U))
            Xmean = round(pd.DataFrame(np.dot(dot1, X)), 4)

            # compute ojective function
            BB = (np.dot(U, Xmean)) - X
            f = round(np.trace(np.dot(BB.transpose(), BB)), 4)

            # stopping rule
            dif = 0

            for k in range(0, K):
                dif += mse2(Xmean.iloc[k], Xmean0.iloc[k])

            if dif > epsilon:
                Xmean0 = Xmean
            else:
                break

        if loop == 0:
            U_ott = U
            f_ott = f
            Xmean_ott = Xmean

        if f < f_ott:
            U_ott = U
            f_ott = f
            Xmean_ott = Xmean

    # calculate cost
    cost = costf(X, Xmean_ott, U_ott)
    print('Done')
    return round(pd.DataFrame(U_ott), 4), f_ott, cost




def standardizeDataFrame(data):
    u = pd.DataFrame([1]*len(data))
    _u = u.transpose()
    _dot1 = pd.DataFrame(np.dot(u, _u))
    mat = round(pd.DataFrame(np.dot(1/len(data), _dot1)), 4)
    #centrature matrix
    Jc = np.identity(len(data)) - mat
    dot_data = pd.DataFrame(np.dot(Jc, data))
    dot_data2 = np.dot(dot_data.transpose(), dot_data)
    s_data = round(pd.DataFrame(np.dot(1/len(data), dot_data2)), 4)

    diagonal = np.array(np.diag(s_data))
    d2 = diagonalize(diagonal)**0.5
    d2_inv = pd.DataFrame(np.linalg.pinv(d2.values)) #pseudo inverse

    dot1 = pd.DataFrame(np.dot(data, d2_inv))
    stand = round(pd.DataFrame(np.dot(Jc, dot1)), 4)
    return stand


def clusterization(cluster, data):
    c1 = []
    c2 = []
    c3 = []
    for i in range(0, len(data)):
        if cluster[0].iloc[i][0] == 1:
            c1.append(data.iloc[i])
        if cluster[0].iloc[i][1] == 1:
            c2.append(data.iloc[i])
        if cluster[0].iloc[i][2] == 1:
            c3.append(data.iloc[i])

    c1 = pd.DataFrame(c1)
    c2 = pd.DataFrame(c2)
    c3 = pd.DataFrame(c3)
    return c1, c2, c3


def f_variables(X, U):
    U_inv = pd.DataFrame(np.linalg.pinv(U.values))

    fm = []
    for var in range(0, len(X.iloc[0])):
        z_vars = stats.zscore(X.iloc[:, var])
        Xm = np.dot(U_inv, z_vars)
        Db = np.dot(np.dot(Xm.transpose(), U.transpose()), np.dot(U, Xm))
        Dw = np.dot((z_vars - (np.dot(U, Xm))).transpose(), (z_vars - np.dot(U, Xm)))
        fm.append( (Db/(len(U.iloc[0]) - 1)) / (Dw/(len(X) - len(U.iloc[0]))))

    return fm


def dwdb(data, U, Xm, K):
    u = pd.DataFrame([1] * len(data))
    _u = u.transpose()
    _dot1 = pd.DataFrame(np.dot(u, _u))
    mat = round(pd.DataFrame(np.dot(1 / len(data), _dot1)), 4)
    # centrature matrix
    Jc = pd.DataFrame(np.identity(len(data)) - mat)

    data_c = round(pd.DataFrame(np.dot(Jc, data)), 4)
    _Xm = round(pd.DataFrame(np.dot(pd.DataFrame(np.linalg.pinv(U.values)), data)), 4)

    # WITHIN
    p = data_c - np.dot(U, _Xm)
    D_w = np.trace(np.dot(p.transpose(), p))

    # BETWEEN
    b = np.dot(U, _Xm)
    D_b = np.trace(np.dot(b.transpose(), b))

    return (D_b / (K - 1)) / (D_w / (len(data) - K))





