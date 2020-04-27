def c_func_opt(theta):

    ######################################## PIMA Dataset ##################################################
    # data_pima = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/PIMA/pima-indians-diabetes.csv')
    # data_pima.rename(columns={'1':'Target', '6':'Pregnancies', '148':'Glucose', '72':'BloodPressure', '35':'SkinThickness', '0': 'Insulin','33.6': 'BMI', '0.627':'DiabeticPedigreeFunction','50':'Age'},inplace=True)
    # X = data_pima.loc[:,:'Age'].values
    # y = data_pima['Target'].values
    # data_pima_positive = data_pima.loc[(data_pima['Target'] > 0)]
    # data_pima_negative = data_pima.loc[(data_pima['Target'] < 1)]
    # data_positive = data_pima_positive
    # data_negative = data_pima_negative

    ########################################## Sonar Dataset ##############################################
    data_sonar = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/Sonar/sonar.all-data')
    data_sonar.rename(columns= {'R': 'Target'}, inplace = True)
    data_sonar['Target'] = data_sonar.Target.map({'R':0, 'M':1})
    X = data_sonar.iloc[:, 0:60].values
    y = data_sonar['Target'].values
    data_sonar_positive = data_sonar.loc[(data_sonar['Target'] > 0)]
    data_sonar_negative = data_sonar.loc[(data_sonar['Target'] < 1)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
    data_positive = data_sonar_positive
    data_negative = data_sonar_negative

    ####################################### Ionosphere Dataset ###########################################
    # data_ionos = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/Ionosphere/uci-ionosphere/ionosphere_data_kaggle.csv')
    # data_ionos.rename(columns={'label':'Target'}, inplace = True)
    # X = data_ionos.drop(['Target'], axis=1).values
    # data_ionos['Target'] = data_ionos.Target.map({'g':1, 'b':0})
    # y = data_ionos['Target'].values
    # data_ionos_positive = data_ionos.loc[(data_ionos['Target']>0)]
    # data_ionos_negative = data_ionos.loc[(data_ionos['Target']<1)]
    # data_positive = data_ionos_positive
    # data_negative = data_ionos_negative


    sq_dist = pdist(X, 'sqeuclidean')
    sigma = [10**(0.1), 10**(-0.7), 10**(-0.4), 10**(-0.1), 10**(0.2), 10**(0.5), 10**(0.8), 10**(1.1), 10**(1.4), 10**(1.7)]
    mat_sqr_dist = squareform(sq_dist)

    G = 0
    G_list = []
    
    for value in range(10):
        gamma = 1/(sigma[value]**2)
        gamma = -gamma 
        g = np.exp(gamma * mat_sqr_dist)
        G_g = theta[value] * g
        G_list.append(G_g)
        G = np.add(G, G_g)

    print("G: ", G)
    print("G.shape: ", G.shape)  
    
    m_plus = len(data_positive.index)
    m_minus = len(data_negative.index)
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
    return func_val

def calculate_parameters(theta, X, data_positive, data_negative):

    sq_dist = pdist(X, 'sqeuclidean')
    sigma = [10**(0.1), 10**(-0.7), 10**(-0.4), 10**(-0.1), 10**(0.2), 10**(0.5), 10**(0.8), 10**(1.1), 10**(1.4), 10**(1.7)]
    mat_sqr_dist = squareform(sq_dist)

    # Gradient calculation
    gradient_value = []
    G = 0 
    for value in range(10):
        gamma = 1/(sigma[value]**2)
        gamma = -gamma  
        g = np.exp(gamma * mat_sqr_dist)
        #G_g = theta[value] * g 
        G = g
        
        m_plus = len(data_positive.index)
        m_minus = len(data_negative.index)
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
        gradient_value.append(func_val)



    G = 0 
    for value in range(10):
        gamma = 1/(sigma[value]**2)
        gamma = -gamma  
        g = np.exp(gamma * mat_sqr_dist)
        G_g = theta[value] * g 
        G = np.add(G, G_g)
        
    m_plus = len(data_positive.index)
    m_minus = len(data_negative.index)
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

    



    return sq_dist, sigma, a, J, gradient_value 

def classifier_KFDA(X, X_test, theta, sq_dist, sigma, a, J):

    lambda_val = 10**(-8)
    mat_sqr_dist = squareform(sq_dist)

    G = 0 
    for value in range(10):
        gamma = 1/(sigma[value]**2)
        gamma = -gamma 
        g = np.exp(gamma * mat_sqr_dist)
        G_g = theta[value] * g
        G = np.add(G, G_g)

    I = np.identity(len(J))
    J_G = np.matmul(J, G)
    J_G_J = np.matmul(J_G, J)
    lambda_I = lambda_val * I 
    value_1 = np.add(lambda_I, J_G_J) 
    inv_value = linalg.inv(value_1)
    J_inv_value = np.matmul(J, inv_value)
    J_inv_value_J_G = np.matmul(J_inv_value, J_G)
    substract_value = np.subtract(I, J_inv_value_J_G)
    brac_value = np.matmul(substract_value, a)
    lambda_inv = (1/lambda_val)
    alpha_opt_value = lambda_inv * brac_value

    list_x_ProjValues = []
    list_of_values = []

    for index_a in range(len(X_test)): 
        value_x = 0
        for index_i in range(len(X)):
            for index_k in range(10):
                sigma_squared = np.square(sigma[index_k])
                value_X_diff = np.linalg.norm(X[index_i] - X_test[index_a])
                sqaured_value_X_diff = np.square(value_X_diff)
                expo = np.exp(-(sqaured_value_X_diff/sigma_squared))
                value_theta_k = theta[index_k] * expo 
                value_x += alpha_opt_value[index_i] * value_theta_k

        list_of_values.append(value_x)
        if value_x < 0: 
            list_x_ProjValues.append(0)
        else: 
            list_x_ProjValues.append(1)
    
    list_of_values = np.array(list_of_values)
    print("list_of_values: ", list_of_values)

    list_x_ProjValues = np.array(list_x_ProjValues)
    print("list_x_ProjValues: \n",list_x_ProjValues)
    print("list_x_ProjValues.shape: ", list_x_ProjValues.shape)

    return list_x_ProjValues

def gradient_value(grad):
    return grad

if __name__ == "__main__":

    import numpy as np 
    import pandas as pd 
    from sklearn.model_selection import train_test_split
    from scipy.spatial.distance import pdist, squareform 
    from scipy import linalg
    from scipy.optimize import linprog
    from scipy.optimize import LinearConstraint
    from scipy import optimize as optimize
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder

    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_column', None)
    theta = np.array([0.15, 0.11, 0.13, 0.1, 0.1, 0.19, 0.1, 0.19, 0.19, 0.19])
    # theta = np.array([0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.1])
    one_vec = np.ones(len(theta))
    # print("one_vec: \n", one_vec)
    one_vec_t_theta = np.matmul(one_vec.T, theta)
    # print("one_vec_t_theta: \n", one_vec_t_theta)
    con1 = {'type': 'eq', 'fun': lambda theta:  np.matmul(one_vec.T, theta)-1}   
    con2 = LinearConstraint(theta, lb=0, ub=0)
    cons = np.array([])
    cons = np.append(cons, con1)
    cons = np.append(cons, con2)


    ############################ PIMA Dataset ##############################################################
    # Optimization for Optimal Theta PIMA Dataset

    # result = c_func_opt(theta)
    # theta = np.array([0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.1])

    # gradient_val  = np.array([-7.00019903e-05, -1.34497621e-04, -1.34497621e-04,  2.67902020e-05,  -2.70382675e-05, -1.38515935e-05, -2.06434696e-05, -1.92680073e-05, -2.04748631e-04, -2.19010286e-01])
    # value = gradient_value(gradient_val)

    # one_vec = np.ones(len(theta))
    # # print("one_vec: \n", one_vec)
    # one_vec_t_theta = np.matmul(one_vec.T, theta)
    # # print("one_vec_t_theta: \n", one_vec_t_theta)
    # con1 = {'type': 'eq', 'fun': lambda theta:  np.matmul(one_vec.T, theta)-1}   
    # con2 = LinearConstraint(theta, lb=0, ub=0)
    # cons = np.array([])
    # cons = np.append(cons, con1)
    # cons = np.append(cons, con2)
    # # gradient_value = np.array(gradient_value)
    # # print("gradient_value: ", gradient_value)
    # result = optimize.minimize(c_func_opt, theta, method='CG', jac = gradient_value, hess= None, hessp = None, bounds=None, constraints= cons, options={'disp':True})
    # print("result: ",result)
        
    """ X_test  accuracy_score:  69.6969696969697 -->    optimal_theta = np.array([0.1 , 0.01, 0.1 , 0.01 , 0.09 , 0.1 , 0.01 , 0.09 , 0.09, 0.1 ])
     X_train accuracy_score: 65.86 --> 
    """
    # optimal_theta = np.array([0.1 , 0.01, 0.1 , 0.1 , 0.9 , 0.1 , 0.01 , 0.09 , 0.09, 0.1 ])


    #****Powell value*****#
    # optimal_theta = np.array([1.01456855, -1.62607429, 0.07771822, 0.09999544, 0.09999631, 0.1, 0.1, 0.10000017, 0.08999994, 0.09999996])
    
    
    # accuracy_score:  62.33766233766234 optimal_theta = np.array([0.10208656, 0.00984434, 0.09608656, 0.10208656, 0.10208656, 0.09971056, 0.10208656, 0.10208656, 0.09089424, 0.09988656])
        
    # acc = 32.8 % [0.1, 0.01, 0.1, 0.1, 0.1025, 0.1, 0.1, 0.1025, 0.09, 0.1])

    #optimal_theta = np.array([0.10208656, 0.00984434, 0.09608656, 0.10208656, 0.10208656, 0.09971056, 0.10208656, 0.10208656, 0.09089424, 0.09988656])
    # this is optimal with > 60 % acc [0.10208656, 0.00984434, 0.09608656, 0.10208656, 0.10208656, 0.09971056, 0.10208656, 0.10208656, 0.09089424, 0.09988656])
    # [0.10239446, 0.10010384, 0.10004884, 0.09982753, 0.10270696, 0.10001721, 0.09983196, 0.10011662, 0.10000203, 0.09997129])
    # [0.09976, 0.10020258, 0.099948, 0.09992538, 0.10029676, 0.09988788, 0.10028821, 0.10467176, 0.0998146 , 0.10009051])

    # data_pima = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/PIMA/pima-indians-diabetes.csv')
    # data_pima.rename(columns={'1':'Target', '6':'Pregnancies', '148':'Glucose', '72':'BloodPressure', '35':'SkinThickness', '0': 'Insulin','33.6': 'BMI', '0.627':'DiabeticPedigreeFunction','50':'Age'},inplace=True)
    # X = data_pima.loc[:,:'Age'].values
    # y = data_pima['Target'].values
    # data_pima_positive = data_pima.loc[(data_pima['Target'] > 0)]
    # data_pima_negative = data_pima.loc[(data_pima['Target'] < 1)]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
    # sq_dist, sigma, a, J, gradient_value  = calculate_parameters(optimal_theta, X, data_pima_positive, data_pima_negative)
    # gradient_value = np.array(gradient_value)
    # print("gradient_value: ", gradient_value)
    # pred_values = classifier_KFDA(X, X_train, optimal_theta, sq_dist, sigma, a, J)
    # acc_score = accuracy_score(y_train, pred_values)
    # print("accuracy_score: ", acc_score*100)
    

    ################################## Sonar DATASET ########################################################

    # theta = np.array([0.15, 0.11, 0.13, 0.1, 0.1, 0.19, 0.1, 0.19, 0.19, 0.19])
    # result = optimize.minimize(c_func_opt, theta, method='CG', jac = gradient_value, options={'disp':True}) # hess= None, hessp = None, bounds=None, constraints= cons,
    # print("result: ",result)
    
    # optimal_theta = np.array([0.15, 0.11, 0.13, 0.1, 0.1, 0.19, 0.1, 0.19, 0.19, 0.19])

    #optimal_theta = np.array([0.001, 0.0105, 0.009, 0.1, 0.1, 0.001, 0.009, 0.1, 0.09, 0.1])
    
    # # [0.10218658, 0.0099456, 0.101839, 0.09814699, 0.10097398, 0.10028149, 0.10410893, 0.0990997 , 0.09132846, 0.09895552])

    # data_sonar = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/Sonar/sonar.all-data')
    # data_sonar.rename(columns= {'R': 'Target'}, inplace = True)
    # data_sonar['Target'] = data_sonar.Target.map({'R':0, 'M':1})
    # X = data_sonar.iloc[:, 0:60].values
    # y = data_sonar['Target'].values
    # data_sonar_positive = data_sonar.loc[(data_sonar['Target'] > 0)]
    # data_sonar_negative = data_sonar.loc[(data_sonar['Target'] < 1)]

    # sq_dist, sigma, a, J, gradient_value = calculate_parameters(optimal_theta, X, data_sonar_positive, data_sonar_negative)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42) 

    # pred_value = classifier_KFDA(X, X_train, optimal_theta, sq_dist, sigma, a, J)
    # acc_score = accuracy_score(y_train, pred_value)
    # print("accuracy_score: ", acc_score*100)
    # print("acc_score: ", acc_score)
  

    #################################### Ionosphere dataset ################################################

    # Optimization for Optimal Theta Sonar Dataset
    # theta = np.array([0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.1])
    # result = optimize.minimize(c_func_opt, theta, method='nelder-mead', options={'disp':True})
    # print("result: ",result)

    # optimal_theta = np.array([0.10042432, 0.01010843, 0.09755272, 0.10108432, 0.0990964,  0.10108432, 0.10108432, 0.10053432, 0.09097589, 0.10108432])
    
    # # [0.10016358, 0.10002186, 0.10033923, 0.10009438, 0.10016358, 0.10008108, 0.09987294, 0.10453858, 0.09981708, 0.10009285])
    
    # data_ionos = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/Ionosphere/uci-ionosphere/ionosphere_data_kaggle.csv')
    # data_ionos.rename(columns={'label':'Target'}, inplace = True)
    # X = data_ionos.drop(['Target'], axis=1).values
    # data_ionos['Target'] = data_ionos.Target.map({'g':1, 'b':0})
    # y = data_ionos['Target'].values

    # data_ionos_positive = data_ionos.loc[(data_ionos['Target']>0)]
    # data_ionos_negative = data_ionos.loc[(data_ionos['Target']<1)]

    # sq_dist, sigma, a, J = calculate_parameters(optimal_theta, X, data_ionos_positive, data_ionos_negative)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42) 

    # pred_value = classifier_KFDA(X, X_train, optimal_theta, sq_dist, sigma, a, J)
    # acc_score = accuracy_score(y_train, pred_value)
    # print("accuracy_score: ", acc_score*100)
    # print("acc_score: ", acc_score)

    