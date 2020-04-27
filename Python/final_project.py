def c_func_opt(theta):

    ######################################## PIMA Dataset ##################################################
    data_pima = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/PIMA/pima-indians-diabetes.csv')
    data_pima.rename(columns={'1':'Target', '6':'Pregnancies', '148':'Glucose', '72':'BloodPressure', '35':'SkinThickness', '0': 'Insulin','33.6': 'BMI', '0.627':'DiabeticPedigreeFunction','50':'Age'},inplace=True)
    X = data_pima.loc[:,:'Age'].values
    y = data_pima['Target'].values
    data_pima_positive = data_pima.loc[(data_pima['Target'] > 0)]
    data_pima_negative = data_pima.loc[(data_pima['Target'] < 1)]
    data_positive = data_pima_positive
    data_negative = data_pima_negative

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
    data_ionos = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/Ionosphere/uci-ionosphere/ionosphere_data_kaggle.csv')
    data_ionos.rename(columns={'label':'Target'}, inplace = True)
    X = data_ionos.drop(['Target'], axis=1).values
    data_ionos['Target'] = data_ionos.Target.map({'g':1, 'b':0})
    y = data_ionos['Target'].values
    data_ionos_positive = data_ionos.loc[(data_ionos['Target']>0)]
    data_ionos_negative = data_ionos.loc[(data_ionos['Target']<1)]
    data_positive = data_ionos_positive
    data_negative = data_ionos_negative

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
    sigma = [10**(0.1), 10**(-0.7), 10**(-0.4), 10**(-0.1), 10**(0.2), 10**(0.5),
     10**(0.8), 10**(1.1), 10**(1.4), 10**(1.7)]
    mat_sqr_dist = squareform(sq_dist)
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
    return sq_dist, sigma, a, J 

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

        if value_x < 0: 
            list_x_ProjValues.append(0)
        else: 
            list_x_ProjValues.append(1)
    
    list_x_ProjValues = np.array(list_x_ProjValues)
    return list_x_ProjValues

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

    theta = np.array([0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.1])
    one_vec = np.ones(len(theta))
    one_vec_t_theta = np.matmul(one_vec.T, theta)
    con1 = {'type': 'eq', 'fun': lambda theta:  np.matmul(one_vec.T, theta)-1}   
    con2 = LinearConstraint(theta, lb=0, ub=0)
    cons = np.array([])
    cons = np.append(cons, con1)
    cons = np.append(cons, con2)

    ####################################################################### PIMA Dataset ##############################################################
    # Optimization for Optimal Theta PIMA Dataset
    
    result = c_func_opt(theta)
    result = optimize.minimize(c_func_opt, theta, method='nelder-mead', options={'disp':True})
    print("result: ",result)    
    optimal_theta = np.array([0.10208656, 0.00984434, 0.09608656, 0.10208656, 0.10208656, 0.09971056, 0.10208656, 0.10208656, 0.09089424, 0.09988656])

    data_pima = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/PIMA/pima-indians-diabetes.csv')
    data_pima.rename(columns={'1':'Target', '6':'Pregnancies', '148':'Glucose', '72':'BloodPressure', '35':'SkinThickness',
    '0': 'Insulin','33.6': 'BMI', '0.627':'DiabeticPedigreeFunction','50':'Age'},inplace=True)
    X = data_pima.loc[:,:'Age'].values
    y = data_pima['Target'].values
    data_pima_positive = data_pima.loc[(data_pima['Target'] > 0)]
    data_pima_negative = data_pima.loc[(data_pima['Target'] < 1)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
    sq_dist, sigma, a, J = calculate_parameters(optimal_theta, X, data_pima_positive, data_pima_negative)
    pred_values = classifier_KFDA(X, X_test, optimal_theta, sq_dist, sigma, a, J)
    acc_score = accuracy_score(y_test, pred_values)
    print("accuracy_score: ", acc_score*100)
    
    ###################################################################### Sonar DATASET ################################################################

    theta = np.array([0.15, 0.11, 0.13, 0.1, 0.1, 0.19, 0.1, 0.19, 0.19, 0.19])
    result = optimize.minimize(c_func_opt, theta, method='nelder-mead', options={'disp':True})
    print("result: ",result)
    optimal_theta = np.array([0.001, 0.0105, 0.009, 0.1, 0.1, 0.001, 0.009, 0.1, 0.09, 0.1])
    
    data_sonar = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/Sonar/sonar.all-data')
    data_sonar.rename(columns= {'R': 'Target'}, inplace = True)
    data_sonar['Target'] = data_sonar.Target.map({'R':0, 'M':1})
    X = data_sonar.iloc[:, 0:60].values
    y = data_sonar['Target'].values
    data_sonar_positive = data_sonar.loc[(data_sonar['Target'] > 0)]
    data_sonar_negative = data_sonar.loc[(data_sonar['Target'] < 1)]
    sq_dist, sigma, a, J = calculate_parameters(optimal_theta, X, data_sonar_positive, data_sonar_negative)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42) 
    pred_value = classifier_KFDA(X, X_test, optimal_theta, sq_dist, sigma, a, J)
    acc_score = accuracy_score(y_test, pred_value)
    print("accuracy_score: ", acc_score*100)

    #################################################################### Ionosphere dataset ###############################################################

    # Optimization for Optimal Theta Sonar Dataset
    theta = np.array([0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.1])
    result = optimize.minimize(c_func_opt, theta, method='nelder-mead', options={'disp':True})
    print("result: ",result)

    optimal_theta = np.array([0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.1])
        
    data_ionos = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/ECE_602/Project_final/Dataset/Ionosphere/uci-ionosphere/ionosphere_data_kaggle.csv')
    data_ionos.rename(columns={'label':'Target'}, inplace = True)
    X = data_ionos.drop(['Target'], axis=1).values
    data_ionos['Target'] = data_ionos.Target.map({'g':1, 'b':0})
    y = data_ionos['Target'].values
    data_ionos_positive = data_ionos.loc[(data_ionos['Target'] > 0)]
    data_ionos_negative = data_ionos.loc[(data_ionos['Target'] < 1)]

    sq_dist, sigma, a, J = calculate_parameters(optimal_theta, X, data_ionos_positive, data_ionos_negative)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42) 
    pred_value = classifier_KFDA(X, X_test, optimal_theta, sq_dist, sigma, a, J)
    acc_score = accuracy_score(y_test, pred_value)
    print("Accuracy Score: ", acc_score*100)


    