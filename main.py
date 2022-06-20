import numpy as np      
import pandas as pd        
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import MDS

import matplotlib
import matplotlib.pyplot as plt

from itertools import combinations
import copy

from funcs import *
from params import *


#--------------------------
# read data
df_data = pd.read_csv(DataPath, header=None)
df_data.columns = data_attribute

#---
# encode the categorical attributes
le = LabelEncoder()
LabelEncoder_Cols = categorical_attribute

df = df_data[LabelEncoder_Cols].copy()
d = defaultdict(LabelEncoder)
encoded_series = df.apply(lambda x: d[x.name].fit_transform(x))

df_data_processed = pd.concat([df_data.drop(LabelEncoder_Cols,axis=1), encoded_series], axis=1)
df_data_processed_bk = df_data_processed.copy() 

#--------------------------
print('label_Y is : '+label_Y)
print('label_O is : '+label_O)
print('the target threshold value of epsilon is:', epsilon_threshold_val)
epsilon_val = 1e10 # initialize the epsilon_val to a very large value
step_val = 0     # step_0 represents the raw data

print('Bias Mitigation Progress: Begin')
while epsilon_val> epsilon_threshold_val:
    
    print('---------------------------------------------')
    print('Current Step', step_val)
    try:
        exec_str = 'bias_mitigation_dict = bias_mitigation_dict_step_' + str(step_val)
        exec(exec_str)
    except:
        raise Exception('Error: check bias_mitigation_dict in params.py')
    
    # for each while loop, reset the data to 'df_data_processed_bk'
    # and then apply the bias mitigation process
    df_data_processed = df_data_processed_bk.copy()
    df_data_processed.drop(columns=[label_Y], inplace=True)
    
    
    #-----------------------------------------------------------------------------------
    # Step4: Bias Mitigation
    # bias_mitigation_dict: details shown in params.py
    bias_mitigation_keys_arr = list(bias_mitigation_dict.keys())
    for i_attribute in bias_mitigation_keys_arr:

        dict_attribute = bias_mitigation_dict[i_attribute]

        # numerical attribute: polynomial transformation  
        if i_attribute in numerical_attribute:
            y_name = i_attribute
            y_data = df_data_processed[y_name]**(dict_attribute)
            df_data_processed[y_name] = y_data  
        # categorical attribute: re-binning transformation
        elif i_attribute in categorical_attribute:
            y_name = i_attribute
            df_tmp = df_data_processed[y_name].copy() 
            keys_arr = list(dict_attribute.keys())
            for i_key in keys_arr:
                df_tmp[df_tmp==i_key] = dict_attribute[i_key]  
            df_data_processed[y_name] = df_tmp
        else:
            raise Exception('Error attribute', i_attribute)
    
    
    
    #-----------------------------------------------------------------------------------
    # Step1: Data Normalization
    comb_label_arr = list(combinations(list(set(df_data_processed[label_O].values)), 2))
    df_S_full = pd.DataFrame()
    for comb_label in comb_label_arr:

        # For multi-nary label_O, select a pair of data each time
        p_label, n_label = comb_label
        df_data_processed_p = df_data_processed[df_data_processed[label_O]==p_label].copy()
        df_data_processed_n = df_data_processed[df_data_processed[label_O]==n_label].copy()
        df_data_processed_p[label_O] = 1
        df_data_processed_n[label_O] = 0    
        df_data_processed_tmp = pd.concat([df_data_processed_p, df_data_processed_n])

        # calculate the min-max normalization
        for i_attribute in numerical_attribute:
            df_tmp = cal_min_max_normalization(df_data_processed_tmp[i_attribute])
            df_data_processed_tmp[i_attribute] = df_tmp

        ## compute attributes’ contribution
        df_S_tmp = cal_df_S(df_data_processed_tmp, label_O, numerical_attribute, categorical_attribute) 
        df_S_full[comb_label] = df_S_tmp    
    
    
    
    #-----------------------------------------------------------------------------------
    #Step 2: Distance Matrix Construction
    ## Compute sub-distance and overall distance
    attribute_num = df_data_processed.shape[1] - 1
    
    index_arr = list(df_S_full.index) + ['origin']
    distance_matrix = pd.DataFrame(index=index_arr, columns=index_arr, data=np.nan)
    for x_a in range(attribute_num+1):
        distance_matrix.iloc[x_a, x_a] = 0

    # x_empty represents the origin 
    x_empty = attribute_num
    wmax_arr = []
    for x_a in range(attribute_num+1):
        print('----------')
        print('calculating distance matrix row:', x_a, '/', attribute_num, ':', index_arr[x_a])    

        for x_b in range(x_a+1, attribute_num+1):
        
            if (x_a == x_empty) and (x_b == x_empty):
                pass
            else:
                #print('----------')
                #print('calculating:', index_arr[x_a], ':', x_a, ';', index_arr[x_b],  x_b)
                sub_set_arr = [x for x in range(attribute_num)]
                if x_a == x_empty :
                    pass
                else:
                    sub_set_arr.remove(x_a)
                if x_b == x_empty :
                    pass
                else:
                    sub_set_arr.remove(x_b) 

                #-------------------------------
                N_total = len(sub_set_arr)
                if h_order_val==-1:
                    h_order = N_total 
                elif h_order_val > N_total:
                    raise Exception('Error h_order_val', h_order_val)
                else:
                    h_order = h_order_val
                Terminal_H = N_total - h_order  

                
                # dist_val_arr is used to store the value of each sub-distance
                dist_val_arr = []
                for sub_set_size in range(N_total, Terminal_H-1, -1):

                    comb_arr = list(combinations([x for x in range(N_total)], sub_set_size))
                    #print('--------sub_distance--------')
                    #print(sub_set_size, len(comb_arr))

                    for i_comb in range(len(comb_arr)):
                        sub_set_arr_tmp = copy.deepcopy([sub_set_arr[i] for i in comb_arr[i_comb]]) 


                        full_set_arr = sub_set_arr_tmp + [x_a] + [x_b]
                        if x_empty in full_set_arr:
                            full_set_arr.remove(x_empty)    

                        #-----------------------------
                        # remove x_a    
                        full_set_arr_tmp = copy.deepcopy(full_set_arr)
                        if x_a == x_empty:
                            pass
                        else:
                            full_set_arr_tmp.remove(x_a)

                        wmax_a_arr = []
                        for comb_label in comb_label_arr:
                            df_S_input = df_S_full[comb_label]
                            df_tmp = df_S_input.iloc[full_set_arr_tmp].copy()
                            if df_tmp.shape[0]>0:
                                wmax_a = np.sqrt( (df_tmp**2).sum() / df_tmp.shape[0])
                            else:
                                wmax_a = 0
                        
                            wmax_a_arr.append(wmax_a)
                        wmax_a = np.max(wmax_a_arr)


                        #-----------------------------
                        # remove x_b
                        full_set_arr_tmp = copy.deepcopy(full_set_arr)
                        if x_b == x_empty:
                            pass
                        else:
                            full_set_arr_tmp.remove(x_b)

                        wmax_b_arr = []
                        for comb_label in comb_label_arr:
                            df_S_input = df_S_full[comb_label]
                            df_tmp = df_S_input.iloc[full_set_arr_tmp].copy()
                            if df_tmp.shape[0]>0:
                                wmax_b = np.sqrt( (df_tmp**2).sum() / df_tmp.shape[0])
                            else:
                                wmax_b = 0

                            wmax_b_arr.append(wmax_b)
                        wmax_b = np.max(wmax_b_arr)

                        #-----------------------------
                        dist_val = np.abs(wmax_a - wmax_b)
                        dist_val_arr.append(dist_val)

                # the full distance matrix is the un-weighted sum (i.e. mean) of all sub-distance
                distance_matrix.iloc[x_a, x_b] = np.mean(dist_val_arr)
                distance_matrix.iloc[x_b, x_a] = distance_matrix.iloc[x_a, x_b]


    
    
    
    

    #-----------------------------------------------------------------------------------
    #Step 3:  Bias Concentration Determination
    ## MDS
    random_state_val = 0
    stress = []
    # Max value for n_components
    max_range = 9
    for dim in range(1, max_range):
        mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=random_state_val, n_init=4, max_iter=10000, eps=1e-10)
        mds.fit(distance_matrix)
        stress.append(mds.stress_) 


    # elbow plot
    fontsize_val = 13
    fig = plt.figure()
    ax = pd.Series(index=range(1, max_range), data=stress).plot(style='o-', grid=True, fontsize=fontsize_val)
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax.set_xlabel('dim (MDS)', fontsize=fontsize_val)
    ax.set_ylabel('Stress', fontsize=fontsize_val)

    fig.savefig(ResultPath+'MDS_elbow_step'+str(step_val)+'.png',bbox_inches = 'tight')


    

    ## projected attribute space (2D)
    # here the MDS dimensition is fixed at 2; for higher dimension, process the 'distance_matrix_xxx.csv' in Result path
    fig = plt.figure()

    random_state_val = 1
    dim = 2
    mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=random_state_val, n_init=4, max_iter=10000, eps=1e-10)

    # Apply MDS
    # pts = mds.fit(distance_matrix) 
    pts = mds.fit_transform(distance_matrix)
    pts  = pts - pts[-1] 
    
    ax = pd.Series(index=pts[:, 0], data=pts[:, 1]).plot(style='o', grid=True, fontsize=fontsize_val)
    for i in range(len(distance_matrix)):
        ax.text(pts[i, 0]  , pts[i, 1] , distance_matrix.columns[i], fontsize=12)
    i = attribute_num
    pd.Series(index=[pts[i, 0]], data=[pts[i, 1]]).plot(ax=ax,style='rx')
    ax.text(pts[i, 0]  , pts[i, 1] , distance_matrix.columns[i], fontsize=12, color='red')
    ax.legend(['MDS Coordinates (dim=2)'], fontsize=fontsize_val)
    ax.grid()
    ax.set_xlabel('axis 1', fontsize=fontsize_val)
    ax.set_ylabel('axis 2', fontsize=fontsize_val)


    fig.savefig(ResultPath+'MDS_2D_step'+str(step_val)+'.png',bbox_inches = 'tight')



    #----------------------
    # calculate the distance to origin
    bias_concentration_matrix = pd.DataFrame(data=pts, index=distance_matrix.index).T
    df_dist2origin = (bias_concentration_matrix**2).sum().apply(np.sqrt)


    fig = plt.figure()
    ax = df_dist2origin.plot(style='o-')
    tickslabel_arr = list(df_dist2origin.index)
    tickslabel_arr = [x.replace('-', ' ') for x in tickslabel_arr]
    ax.set_xticks(range(len(tickslabel_arr)))
    ax.set_xticklabels(tickslabel_arr, rotation=90)
    ax.set_ylabel('distance to origin',fontsize=fontsize_val)
    ax.set_xticks(range(len(tickslabel_arr)))
    ax.set_xticklabels(tickslabel_arr, rotation=90) 
    
    ax.grid(axis="y")
    fig.savefig(ResultPath+'distance2origin_step'+str(step_val)+'.png', bbox_inches='tight')
    

    distance_matrix.to_csv(ResultPath+'distance_matrix_step_'+str(step_val)+'.csv')
    bias_concentration_matrix.to_csv(ResultPath+'bias_concentration_matrix_step_'+str(step_val)+'.csv')
    
    
    epsilon_val = df_dist2origin.max() 
    print('###---')
    print('maximum distance to origin: ',  epsilon_val)
    if epsilon_val < epsilon_threshold_val:
        print('the maximum distance to origin is smaller than the threshold epsilon value')
        print('Bias Mitigation Progress: Finish')
    
    step_val+=1