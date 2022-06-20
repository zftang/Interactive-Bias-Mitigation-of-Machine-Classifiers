#---------------------------
# data information
# raw data path
DataPath = 'Data/adult.data.csv'
# data attribute names (the order must match the 'adult.data.csv' file)
data_attribute = [
    'age', 'workclass', 'fnlwgt', 'education', 'education num',
    'marital status', 'occupation', 'relationship', 'race', 'sex',
    'capital gain', 'capital loss', 'hours per week', 'native country',
    'income'
]

numerical_attribute = [
    'age', 'fnlwgt', 'education num', 'capital gain', 'capital loss',
    'hours per week'
]

categorical_attribute = [
    'workclass', 'education', 'marital status', 'occupation', 'relationship',
    'race', 'sex', 'native country', 'income'
]

label_Y = 'income'
label_O = 'sex'

#---------------------------
ResultPath = 'Result/'

#---------------------------
# max sub-distance order for the distance matrix, -1 represent full order
h_order_val = -1

#---------------------------
# distance to origin threshold value
epsilon_threshold_val = 0.005

#---------------------------
# bias_mitigation_dict
# numerical attribute: polynomial transformation 
# for example 'hours per week': 5 means x^5
# categorical attribute: re-binning transformation
# for example
# 'occupation': {
#     3: 1
# }
# means change the class encoded 3 to 1
bias_mitigation_dict_step_0 = {}

bias_mitigation_dict_step_1 = {
    'relationship': {
        1: 0,
        4: 0,
        5: 0,
        3: 0
    }
}

bias_mitigation_dict_step_2 = {
    'relationship': {
        1: 0,
        4: 0,
        5: 0,
        3: 0
    },
    'marital status': {
        4: 2,
        0: 2,
        6: 2
    }
}

bias_mitigation_dict_step_3 = {
    'relationship': {
        1: 0,
        4: 0,
        5: 0,
        3: 0
    },
    'marital status': {
        4: 2,
        0: 2,
        6: 2
    },
    'hours per week': 5
}

bias_mitigation_dict_step_4 = {
    'relationship': {
        1: 0,
        4: 0,
        5: 0,
        3: 0
    },
    'marital status': {
        4: 2,
        0: 2,
        6: 2
    },
    'occupation': {
        3: 1
    },
    'hours per week': 5
}