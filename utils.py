import os
import dcor
import pickle
import numpy as np
from IPython.display import HTML
from IPython.display import display

def nice_table(dict, title=''):
    '''
    Given a dictionary, it returns an HTML tables with the key-value pairs arranged in rows or columns.
    '''
    # make a copy
    dict = dict.copy()
    html = f'<h2 style="text-align:left;">{title}</h2>'
    html += '<table style="width:50%; border-collapse: collapse; font-size: 16px; text-align:center; padding: 10px; border: 1px solid #fff;">'
    html += '<tr>'
    for key, value in dict.items():
        html += f'<td style="border: 1px solid #fff; text-align:center; padding: 10px; color: white; border-right: 1px solid #fff;">{key}</td>'
    html += '</tr>'
    
    # check if type of value is scalar and if it is, convert it to a list
    for key, value in dict.items():
        if not isinstance(value, list):
            if isinstance(value, float):
                if value < 1:
                    value = round(value, 5)
                else:
                    value = round(value, 3) 
            dict[key] = [value]

    for i in range(max([len(value) for value in dict.values()])):
        html += '<tr>'
        for key, value in dict.items():
            html += f'<td style="border: 1px solid #fff; text-align:center; padding: 10px; color: white; opacity: 0.8; border-left: 1px solid #fff;">{value[i]}</td>'
        html += '</tr>'
            
    return HTML(html)

def load_hyperparameters(model_name):
    '''
    Given model name, it returns the hyperparameters found by hyperparameter search.
    '''
    # if file exists
    if os.path.isfile(f'../../Saved/{model_name}_opt_params.pkl'):
        with open(f'../../Saved/{model_name}_opt_params.pkl', 'rb') as f:
            opt_params = pickle.load(f)
        return opt_params
    else:
        return {}

def save_hyperparameters(model_name, opt_params):
    '''
    Given model name and hyperparameters, it saves the hyperparameters found by hyperparameter search.
    '''
    with open(f'../../Saved/{model_name}_opt_params.pkl', 'wb') as f:
        pickle.dump(opt_params, f)

def load_model(model_name):
    '''
    Given model name, it returns the model.
    '''
    if not os.path.isfile(f'../../Saved/{model_name}.pkl'):
        return None
    with open(f'../../Saved/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def save_model(model_name, model):
    '''
    Given model name and model, it saves the model.
    '''
    with open(f'../../Saved/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)

def dist_corr(df,target):
    numerical_columns = [ col for col in df.columns if df[col].dtype == 'int64' or df[col].dtype =='float64']
    corr={}
    for col in numerical_columns:
        corr[col]=dcor.distance_correlation(df[col], target)
    display(nice_table(corr,title="Distance Correlation between Target Variable & Other Numeic KPIS"))

def corr_ratio(df,continous_col):
    categ_col = [ col for col in df.columns if df[col].dtype == 'object']
    ratio={}
    for col in categ_col:
        ratio[col]=correlation_ratio(df,col,continous_col)
    display(nice_table(ratio,title="Correlation Ratio between Target Variable & Other Categorical KPIS"))

def correlation_ratio(x_data, col1, col2):
    '''
    A measure of association between a categorical variable and a continuous variable.
    - Divide the continuous variable into N groups, based on the categories of the categorical variable.
    - Find the mean of the continuous variable in each group.
    - Compute a weighted variance of the means where the weights are the size of each group.
    - Divide the weighted variance by the variance of the continuous variable.

    It asks the question: If the category changes are the values of the continuous variable on average different?
    If this is zero then the average is the same over all categories so there is no association.
    '''
    categories = np.array(x_data[col1])
    values = np.array(x_data[col2])
    group_variances = 0
    for category in set(categories):
        group = values[np.where(categories == category)[0]]
        group_variances += len(group)*(np.mean(group)-np.mean(values))**2
    total_variance = sum((values-np.mean(values))**2)
    return (group_variances / total_variance)**.5