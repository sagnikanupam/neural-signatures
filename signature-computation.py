import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial, Chebyshev, Legendre, Laguerre, Hermite,HermiteE
import numpy as np
import pickle
import warnings
import iisignature as iisig
import preprocessing_funcs as pf

bins_before=4
bins_current=1
bins_after=5
poly_type_array = ['Chebyshev', 'Polynomial', 'Legendre', 'Laguerre', 'Hermite', 'HermiteE']
warnings.filterwarnings('ignore')

def evaluate(x_coord, y_coord, degree, polytype):
    '''
    Creates a polynomial from data of a given type and then evaluates how well the polynomial fits the two-dimensional path traced by the mouse in the hippocampus data. 

    Inputs: List of x-coordinates, list of corresponding y-coordinates, degree of polynomial, and polynomial type.
    Returns: A tuple where the first element is the MSE of the polynomial fit and the second element is the polynomial itself
    '''
    class_dict = {'Chebyshev': Chebyshev, 'Polynomial': Polynomial, 'Legendre': Legendre, 'Laguerre': Laguerre, 'Hermite': Hermite, 'HermiteE':HermiteE}
    p = class_dict[polytype].fit(x_coord, y_coord, degree)
    p_vals = [p(x) for x in x_coord]
    return (mean_squared_error(y_coord, p_vals), mean_absolute_error(y_coord, p_vals), r2_score(y_coord, p_vals), p)

def findBestPoly(x_coord, y_coord):
    '''
    Finds the best polynomial fit of each type of polynomial for a dataset of (x, y) coordinates.

    Inputs: List of x-coordinates and a list of corresponding y-coordinates 
    Returns: A dictionary where the keys are the polynomial types and dict[key] stores the tuple (Best MSE, Best MAE, Mean-R^2, and the polynomial generating Best MSE) pair for polynomial of type key.
    '''
    
    results_dict = {}
    for poly_type in poly_type_array:
        results_dict[poly_type] = [float('inf'), None]
    
    warnings.simplefilter('ignore', np.RankWarning)
    for deg in range(1, 10):
        curr_val = {}
        for poly_type in poly_type_array:
            curr_val[poly_type] = evaluate(x_coord, y_coord, deg, poly_type)
        for poly_type in curr_val.keys():
            if curr_val[poly_type][0] < results_dict[poly_type][0]:
                results_dict[poly_type] = curr_val[poly_type]
    return results_dict

def preprocessingData(neural_data, pos_binned):
    '''
    Preprocess data in accordance with Glaser et al's paper. 

    Inputs: neural_data is hippocampus data file in the form of a number of bins x number of neurons array. pos_binned is the positional data in the form of a number of bins x 2 array where the 2 coordinates refer to the x and y coordinates.
    Outputs: A tuple containing X (3D), X_flat (2D) and Y datasets for fitting models
    '''
    number_of_spikes = np.nansum(neural_data,axis=0)
    remove_neurons = np.where(number_of_spikes<100)
    neural_data=np.delete(neural_data, remove_neurons,1)
    X = pf.get_spikes_with_history(neural_data,bins_before,bins_after,bins_current)
    X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
    Y = pos_binned
    remove_times=np.where(np.isnan(Y[:,0]) | np.isnan(Y[:,1]))
    X=np.delete(X,remove_times,0)
    X_flat=np.delete(X_flat,remove_times,0)
    Y=np.delete(Y,remove_times,0) 
    return (X, X_flat, Y)

def meanMetrics(best_polynomials, reg_scores, signature_lengths):
    
    print(f"The mean R^2-value for signature linear maps is {np.mean(reg_scores)}.")
    
    for polytype in poly_type_array:
        MSE_list = []
        MAE_list = []
        R2_list = []
        for interval in best_polynomials:
            MSE, MAE, R2, _ = interval[polytype]
            MSE_list.append(MSE)
            MAE_list.append(MAE)
            R2_list.append(R2)

        print(f"The mean MSE-values for the polynomials of type {polytype} are {np.mean(MSE_list)}.")
        print(f"The mean MAE-values for the polynomials of type {polytype} are {np.mean(MAE_list)}.")
        print(f"The mean R^2-values for the polynomials of type {polytype} are {np.mean(R2_list)}.")

    print(f"The mean signature lengths for the computed signatures are {(np.mean([i[0] for i in signature_lengths]), np.mean([i[1] for i in signature_lengths]))}")

if __name__=="__main__":

    folder = 'decoding-data/'
    with open(folder+'example_data_hc.pickle','rb') as f:
        neural_data, pos_binned = pickle.load(f,encoding='latin1') 
    X, X_flat, Y = preprocessingData(neural_data, pos_binned)
    print(X.shape)
    print(X_flat.shape)
    print(Y.shape)

    best_polynomials = []
    reg_scores = []
    signature_lengths = []

    for i in range(bins_before, 300):#Y.shape[0]-bins_after):
        if i%10==0:
            print(f"Iteration {i} of {Y.shape[0]}...")
        Y_data = Y[i-bins_before:i+bins_after+1]
        x_coords = [i[0] for i in Y_data]
        y_coords = [i[1] for i in Y_data]
        best_polynomials.append(findBestPoly(x_coords, y_coords))
        signature = np.array(iisig.sig(X[i], 2, 1))
        reg = LinearRegression().fit(signature, Y_data[:-1])
        reg_scores.append(reg.score(signature, Y_data[:-1]))
        signature_lengths.append(signature.shape)
    
    meanMetrics(best_polynomials, reg_scores, signature_lengths)
    
    
    # signature = iisig.sig(x_data, 3, 2)
    # print((len(signature), len(signature[0])))
    # reg = LinearRegression().fit(signature, y_data[:-1])
    # print(reg.score(signature, y_data[:-1]))
    # plt.plot(time[:10], x_cd[:10])
    # plt.plot(time[:10], y_cd[:10])
    # plt.savefig('plot.png')
    # plt.show()