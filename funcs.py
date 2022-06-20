import numpy as np      
import pandas as pd      

def cal_min_max_normalization(df_input): 
    min_val = df_input.min()
    max_val = df_input.max()
    val_arr = df_input.values
    
    if max_val != min_val:
        df_output = pd.Series(index=df_input.index, data=(val_arr - min_val) / (max_val - min_val))   
        return df_output
    else:
        return df_input

def cal_S_numerical_and_categorical(X_input_full, y_input, numerical_attribute, categorical_attribute):
    #--------------------
    numerical_attribute_tmp = []
    for i_col in X_input_full.columns:
        if i_col in numerical_attribute:
            numerical_attribute_tmp.append(i_col)
    categorical_attribute_tmp = []
    for i_col in X_input_full.columns:
        if i_col in categorical_attribute:
            categorical_attribute_tmp.append(i_col)
    
    #---
    #numerical_attribute
    X_input = X_input_full[numerical_attribute_tmp].values
    
    #---
    X_pos = X_input[y_input==1]
    X_neg = X_input[y_input==0]
    
    X_pos_mean = np.mean(X_pos, axis=0)
    X_neg_mean = np.mean(X_neg, axis=0)
    df_S_numerical = pd.Series(index=numerical_attribute_tmp, data=np.abs(X_pos_mean - X_neg_mean ))
    
    #---
    #categorical_attribute 
    Si_arr = []
    for i_categorical in categorical_attribute_tmp:
        df_i_categorical = pd.DataFrame(X_input_full[i_categorical])
        
        df_i_categorical['label'] = y_input   
        df_freq = pd.DataFrame()
        df_p = df_i_categorical[df_i_categorical['label'] == 1].groupby(i_categorical).count()
        df_n = df_i_categorical[df_i_categorical['label'] == 0].groupby(i_categorical).count()
        
        if df_n.shape[0]>=df_p.shape[0]:
            df_freq['L'] = df_n
            df_freq['S'] = df_p
        else:
            df_freq['S'] = df_p
            df_freq['L'] = df_n
        
        df_freq.fillna(0, inplace=True)
        df_freq['L'] = df_freq['L']/df_freq['L'].sum()
        df_freq['S'] = df_freq['S']/df_freq['S'].sum()
        df_diff_each_categorical = np.abs( df_freq['L']  - df_freq['S'] )
        Si_val = df_diff_each_categorical.mean()

        Si_arr.append(Si_val)
            
    df_S_categorical = pd.Series(index=categorical_attribute_tmp, data=Si_arr)
    
    return df_S_numerical, df_S_categorical

def cal_df_S(df_data_processed_input, label_O, numerical_attribute, categorical_attribute):
    
    df_data_attribute = df_data_processed_input.drop(columns=[label_O]).copy()
    df_label_O = df_data_processed_input[label_O].copy() 
    label_O_val = df_label_O.values
    
    # calculate the S_m for numerical and categorical attributes 
    df_S_numerical_tmp, df_S_categorical_tmp = \
    cal_S_numerical_and_categorical(df_data_attribute, label_O_val, numerical_attribute, categorical_attribute)    
    df_S = pd.concat([df_S_numerical_tmp, df_S_categorical_tmp])
    df_S.to_frame().T
    
    return df_S