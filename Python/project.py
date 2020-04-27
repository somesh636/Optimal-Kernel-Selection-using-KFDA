
def cost_function(G, J, a):
    lambda_val = 10**(-8)
    I = np.identity(len(J))
    J_G = np.matmul(J,G)
    lambda_I = lambda_val*I
    J_G_J = np.matmul(J_G, J)
    value_1 = (lambda_I + J_G_J)
    J_G_a = np.matmul(J_G,a)
    G_J = np.matmul(G,J)
    aT_G_J = np.matmul(a.T,G_J)
    G_a = np.matmul(G,a)
    aT_G_a = np.matmul(a.T, G_a)
    value_1Inv = linalg.inv(value_1)
    aT_G_J_value1Inv = np.matmul(aT_G_J, value_1Inv)
    aT_G_J_value1Inv_J_G_a = np.matmul(aT_G_J_value1Inv, J_G_a)
    func_val = (1/lambda_val)*(aT_G_J_value1Inv_J_G_a - aT_G_a)
    return func_val

if __name__ == "__main__":

    import numpy as np 
    import pandas as pd 
    from sklearn.model_selection import train_test_split
    from scipy.spatial.distance import pdist, squareform 
    from scipy import linalg
    from scipy.optimize import linprog
    from scipy import optimize as optimize

    pd.set_option('display.max_rows',None)
    pd.set_option('display.max_columns', None)

    data_pima = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/PIMA/pima-indians-diabetes.csv')
    data_pima.rename(columns={'1':'Target', '6':'Pregnancies', '148':'Glucose', '72': 'BloodPressure', '35':'SkinThickness', '0': 'Insulin', '33.6': 'BMI', '0.627':'DiabeticPedigreeFunction', '50':'Age'},inplace=True)
    X = data_pima.loc[:,:'Age'].values
    y = data_pima['Target'].values
    data_pima_positive = data_pima.loc[(data_pima['Target'] > 0)]
    data_pima_negative = data_pima.loc[(data_pima['Target'] < 1)]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # cost_function() calculation 

    # Gram Matrix or the Kernel Matrix
    sq_dist = pdist(X, 'sqeuclidean')
    print("sq_dist: \n", sq_dist)
    sigma = [10**(0.1), 10**(-0.7), 10**(-0.4), 10**(-0.1), 10**(0.2), 10**(0.5), 10**(0.8), 10**(1.1), 10**(1.4), 10**(1.7)]
    theta = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


    G = 0
    for value in range(10):
        gamma = 1/(sigma[value]**2)
        gamma = -gamma * theta[value] 
        print("gamma: ", gamma)
        print('gamma.shape: ', gamma.shape)
        mat_sqr_dist = squareform(sq_dist)
        g = np.exp(gamma * mat_sqr_dist)
        G = np.add(G, g)
        print("G_inside:", G)
        print("G_inside: ",G.shape)

    print("G:", G)
    print("G: ",G.shape)
    # number of positive sample from the dataset
    m_plus = len(data_pima_positive.index)
    data_pima_positive = data_pima_positive.values
    m_minus = len(data_pima_negative.index)


    one_plus = np.ones(m_plus)
    # print("1_+: \n", one_plus)

    one_minus = np.ones(m_minus)
    # print("1_-: \n", one_minus)

    I_plus = np.identity(m_plus)
    #print("I: ", I)

    J_plus_1value = np.dot(one_plus, one_plus.T)
    J_plus = (1/np.sqrt(m_plus)) * (I_plus - (1/m_plus) * J_plus_1value)
    # print("J_+: \n", J_plus)
    # print("J_+.shape: \n", J_plus.shape)

    I_minus = np.identity(m_minus)

    J_minus_1value = np.dot(one_minus, one_minus.T)
    J_minus = (1/np.sqrt(m_minus)) * (I_minus - (1/m_minus) * J_minus_1value)

    J = linalg.block_diag(J_plus, J_minus)
    # print("J: ", J)
    # print("J.shape: ",J.shape)

    a_plus_1 = (1/m_plus)* one_plus
    # a_plus_1 = a_plus_1.T
    a_minus_1  = (1/m_minus)* one_minus
    # print("a_plus_1.shape: ",a_plus_1.shape)
    # print("a_plus_1: \n", a_plus_1)
    # print("a_minus_1: \n", a_minus_1)
    # print("a_minus_1.shape: \n", a_minus_1.shape)

    zeros_a_plus = np.zeros(len(a_minus_1))
    # print("zeros_a_plus: \n", zeros_a_plus)
    # print("zero.shape: ", zeros_a_plus.shape)

    a_plus = np.block([a_plus_1, zeros_a_plus])
    # print("a_plus: \n",a_plus)
    # print("a_plus.shape: ", a_plus.shape)

    zeros_a_minus = np.zeros(len(a_plus_1))
    # print("zeros_a_plus: \n", zeros_a_minus)
    # print("zero.shape: ", zeros_a_minus.shape)

    a_minus = np.block([zeros_a_minus, a_minus_1])
    # print("a_minus: ", a_minus)
    # print("a_minus.shape: ", a_minus.shape)

    a = a_plus - a_minus
    # print("A: ", a)
    # print("a.shape: ", a.shape)

    cost_function(G, J, a)
    print("pass:::::")
    # funtion to give optimal theta to the kernel based classifier
    result = optimize.minimize(cost_function(G,J,a), theta, method='Newton-CG', jac=True)
    print(result)