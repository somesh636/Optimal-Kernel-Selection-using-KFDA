def c_func_opt(theta):

    data_pima = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/PIMA/pima-indians-diabetes.csv')
    data_pima.rename(columns={'1':'Target', '6':'Pregnancies', '148':'Glucose', '72':'BloodPressure', '35':'SkinThickness', '0': 'Insulin','33.6': 'BMI', '0.627':'DiabeticPedigreeFunction','50':'Age'},inplace=True)
    X = data_pima.loc[:,:'Age'].values
    y = data_pima['Target'].values
    data_pima_positive = data_pima.loc[(data_pima['Target'] > 0)]
    data_pima_negative = data_pima.loc[(data_pima['Target'] < 1)]



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)

    sq_dist = pdist(X, 'sqeuclidean')
    sigma = [10**(0.1), 10**(-0.7), 10**(-0.4), 10**(-0.1), 10**(0.2), 10**(0.5), 10**(0.8), 10**(1.1), 10**(1.4), 10**(1.7)]
    mat_sqr_dist = squareform(sq_dist)
    # print("mat_sqr_dist: ", mat_sqr_dist)
    # print("mat_sqr_dist.shape: ", mat_sqr_dist.shape)
    

    # row = len(X)
    # col = len(X)

    
    # G_mat = [[0 for x in range(col)] for y in range(row)]
    # g_sum = 0
    
    # for index_a in range(len(X)):
    #     for index_b in range(len(X)): 
    #         for index_c in range(10): 
    #             # print("index_c:", index_c)
    #             sigma_sqr = np.square(sigma[index_c])
    #             diff_value = np.linalg.norm(X[index_a] - X[index_b])
    #             diff_value_squared = np.square(diff_value)
    #             expo_value = -(diff_value_squared/sigma_sqr)
    #             exponen_val = np.exp(expo_value)
    #             theta_expo_val = theta[index_c] * exponen_val
    #             g_sum += theta_expo_val
    #             # print("g_sum: ", g_sum)

    #         G_mat[index_a][index_b] = g_sum
    #         # print("g_sum_inside loop: ", g_sum)
    #         g_sum = 0

    # G_mat_array = np.array(G_mat)
    # print("G_mat: ", G_mat_array)
    # print("G_mat.shape: ", G_mat_array.shape)
    # # G = G_mat_array

    G = 0
    
    G_list = []
    
    for value in range(10):
        gamma = 1/(sigma[value]**2)
        gamma = -gamma 
        g = np.exp(gamma * mat_sqr_dist)
        G_g = theta[value] * g
        G_list.append(g)
        G = np.add(G, G_g)

    print("G: ", G)
    print("G.shape: ", G.shape)  

    # number of positive sample from the dataset
    m_plus = len(data_pima_positive.index)
    data_pima_positive = data_pima_positive.values
    m_minus = len(data_pima_negative.index)
    one_plus = np.ones(m_plus)
    one_minus = np.ones(m_minus)
    I_plus = np.identity(m_plus)
    J_plus_1value = np.dot(one_plus, one_plus.T)
    J_plus = (1/np.sqrt(m_plus)) * (I_plus - (1/m_plus) * J_plus_1value)
    I_minus = np.identity(m_minus)
    J_minus_1value = np.dot(one_minus, one_minus.T)
    J_minus = (1/np.sqrt(m_minus)) * (I_minus - (1/m_minus) * J_minus_1value)

    J = linalg.block_diag(J_plus, J_minus)
    a_plus_1 = (1/m_plus)* one_plus
    a_minus_1  = (1/m_minus)* one_minus
    zeros_a_plus = np.zeros(len(a_minus_1))
    a_plus = np.block([a_plus_1, zeros_a_plus])
    zeros_a_minus = np.zeros(len(a_plus_1))
    a_minus = np.block([zeros_a_minus, a_minus_1])
    a = a_plus - a_minus
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

    # sum_value = 0
    # for index_j in range(len(X)):
    #     sum_value += 
    # func_val = (1/lambda_val) * ()
    return func_val


def c_func(theta):

    data_pima = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/PIMA/pima-indians-diabetes.csv')
    data_pima.rename(columns={'1':'Target', '6':'Pregnancies', '148':'Glucose', '72':'BloodPressure', '35':'SkinThickness', '0': 'Insulin','33.6': 'BMI', '0.627':'DiabeticPedigreeFunction','50':'Age'},inplace=True)
    X = data_pima.loc[:,:'Age'].values
    y = data_pima['Target'].values
    data_pima_positive = data_pima.loc[(data_pima['Target'] > 0)]
    data_pima_negative = data_pima.loc[(data_pima['Target'] < 1)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)

    sq_dist = pdist(X, 'sqeuclidean')
    sigma = [10**(0.1), 10**(-0.7), 10**(-0.4), 10**(-0.1), 10**(0.2), 10**(0.5), 10**(0.8), 10**(1.1), 10**(1.4), 10**(1.7)]
    mat_sqr_dist = squareform(sq_dist)

    G = 0 
    for value in range(10):
        gamma = 1/(sigma[value]**2)
        gamma = -gamma  
        g = np.exp(gamma * mat_sqr_dist)
        G_g = theta[value] * g 
        G = np.add(G, G_g)
        
    # number of positive sample from the dataset
    m_plus = len(data_pima_positive.index)
    data_pima_positive = data_pima_positive.values
    m_minus = len(data_pima_negative.index)
    one_plus = np.ones(m_plus)
    one_minus = np.ones(m_minus)
    I_plus = np.identity(m_plus)
    J_plus_1value = np.dot(one_plus, one_plus.T)
    J_plus = (1/np.sqrt(m_plus)) * (I_plus - (1/m_plus) * J_plus_1value)
    I_minus = np.identity(m_minus)
    J_minus_1value = np.dot(one_minus, one_minus.T)
    J_minus = (1/np.sqrt(m_minus)) * (I_minus - (1/m_minus) * J_minus_1value)
    J = linalg.block_diag(J_plus, J_minus)
    a_plus_1 = (1/m_plus)* one_plus
    a_minus_1  = (1/m_minus)* one_minus
    zeros_a_plus = np.zeros(len(a_minus_1))
    a_plus = np.block([a_plus_1, zeros_a_plus])
    zeros_a_minus = np.zeros(len(a_plus_1))
    a_minus = np.block([zeros_a_minus, a_minus_1])
    # print("a_plus: ", a_plus)
    # print("a_plus.shape: ", a_plus.shape)

    # print("a_minus: ", a_minus)
    # print("a_minus.shape: ", a_minus.shape)
    a = a_plus - a_minus
    # print("a: ", a)
    # print("a.shape: ", a.shape)
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
    return sq_dist, sigma, a, J 


def classifier_KFDA(X, X_test, theta, sq_dist, sigma, a, J):

    lambda_val = 10**(-8)
    # the function is 

    # sigma_opt = 

    # h_x = np.matmul(sigma, K)
    # sq_dist = pdist(X, 'sqeuclidean')
    mat_sqr_dist = squareform(sq_dist)

    # print("len of G",len(J))
    G = 0 
    for value in range(10):
        gamma = 1/(sigma[value]**2)
        # print("gamma_pos: ", gamma)
        gamma = -gamma 
        # print("gamma: ", gamma)
        g = np.exp(gamma * mat_sqr_dist)
        G_g = theta[value] * g
        G = np.add(G, G_g)

    # print("G: \n", G)
    # print("G.shapeeee: ", G.shape)

    I = np.identity(len(J))
    J_G = np.matmul(J, G)
    J_G_J = np.matmul(J_G, J)
    lambda_I = lambda_val * I 
    # print("lambda_I: \n", lambda_I)
    # print("lambda_I.shape: ", lambda_I.shape)
    value_1 = np.add(lambda_I, J_G_J) 
    # print("value_1: \n", value_1)
    # print("value_1.shape: ", value_1.shape)
    inv_value = linalg.inv(value_1)
    # print("inv_value: \n", inv_value)
    # print("inv_value.shape: ", inv_value.shape)

    J_inv_value = np.matmul(J, inv_value)
    # print("J_inv_value: \n", J_inv_value)
    # print("J_inv_value.shape: ", J_inv_value.shape)
    J_inv_value_J_G = np.matmul(J_inv_value, J_G)
    substract_value = np.subtract(I, J_inv_value_J_G)
    # print("substract_value: \n", substract_value)
    # print("substract_value.shape: ", substract_value.shape)
    brac_value = np.matmul(substract_value, a)
    # print("brac_value: \n", brac_value)
    # print("brac_value.shape: ", brac_value.shape)
    lambda_inv = (1/lambda_val)
    alpha_opt_value = lambda_inv * brac_value
    print("alpha_opt_value: \n", alpha_opt_value)
    print("alpha_opt_value.shape: ", alpha_opt_value.shape)

    list_x_ProjValues = []
    
    list_of_values = []

    for index_a in range(len(X_test)): 
        value_x = 0
        for index_i in range(len(X)):
            for index_k in range(10):
                sigma_squared = np.square(sigma[index_k])
                value_X_diff = np.linalg.norm(X[index_i] - X_test[index_a])
                sqaured_value_X_diff = np.square(value_X_diff)
                # expo_value = -(sqaured_value_X_diff/sigma_squared)
                expo = np.exp(-(sqaured_value_X_diff/sigma_squared))
                # print("expo: ",expo)
                value_theta_k = theta[index_k] * expo 
                value_x += alpha_opt_value[index_i] * value_theta_k

        list_of_values.append(value_x)
        if value_x < 0: 
            list_x_ProjValues.append(0)
        else: 
            list_x_ProjValues.append(1)
    
    list_of_values = np.array(list_of_values)
    print("list_of_values: ", list_of_values)

    # print("list_x_ProjValues: \n",list_x_ProjValues)
    # print("list_x_ProjValues.shape: ", list_x_ProjValues.shape)
    list_x_ProjValues = np.array(list_x_ProjValues)
    print("list_x_ProjValues: \n",list_x_ProjValues)
    print("list_x_ProjValues.shape: ", list_x_ProjValues.shape)

    return list_x_ProjValues


        

if __name__ == "__main__":

    import numpy as np 
    import pandas as pd 
    from sklearn.model_selection import train_test_split
    from scipy.spatial.distance import pdist, squareform 
    from scipy import linalg
    from scipy.optimize import linprog
    from scipy import optimize as optimize
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder

    # PIMA DATASET Analysis
    data_pima = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/PIMA/pima-indians-diabetes.csv')
    data_pima.rename(columns={'1':'Target', '6':'Pregnancies', '148':'Glucose', '72':'BloodPressure', '35':'SkinThickness', '0': 'Insulin','33.6': 'BMI', '0.627':'DiabeticPedigreeFunction','50':'Age'},inplace=True)
    X = data_pima.loc[:,:'Age'].values
    y = data_pima['Target'].values
    data_pima_positive = data_pima.loc[(data_pima['Target'] > 0)]
    data_pima_negative = data_pima.loc[(data_pima['Target'] < 1)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)

    theta = np.array([0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.1])

    optimal_theta = np.array([0.10208656, 0.00984434, 0.09608656, 0.10208656, 0.10208656, 0.09971056, 0.10208656, 0.10208656, 0.09089424, 0.09988656])
        
    #[0.105, 0.01 , 0.1  , 0.1  , 0.1  , 0.1  , 0.1  , 0.1  , 0.09 , 0.1])
        
    #[0.09960701, 0.00995461, 0.10019745, 0.10085828, 0.10285725, 0.10042986, 0.10027806, 0.10022878, 0.09097063, 0.10042986])
    sq_dist, sigma, a, J = c_func(optimal_theta)
    value = c_func_opt(theta)

    # result = optimize.minimize(c_func_opt, theta, method='nelder-mead', options={'disp':True})
    # print(result)


    pred_values = classifier_KFDA(X, X_train, optimal_theta, sq_dist, sigma, a, J)
    acc_score = accuracy_score(y_train, pred_values)
    print("accuracy_score: ", acc_score*100)

    




    