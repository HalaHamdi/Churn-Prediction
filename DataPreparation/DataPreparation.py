import os
import pickle
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import category_encoders as ce
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from analyzer import calc_outliers_range


def feature_histograms_analysis(df):

    '''
    Generates a series of three histogram/ bar chart plots for each group of columns of the same type (categorical, numerical).

    '''

    def grid_plot_generator(x_data,  rows, cols, title, figsize=(20, 10), numerical=False):
        """
        Generates a rows x cols grid of plots for the given data where each plot is a bar chart.
        If data is numerical, it uses a histogram instead.
        """
        ### Set up plt grid
        plt.style.use('dark_background')
        plt.rcParams['figure.dpi'] = 200        # increase plot resolution
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
        
        ### For each column plot in the right subplot of the grid
        for i, col in enumerate(x_data.columns):
            
            if rows == 1:
                ax = axs[i % cols]                      
            elif cols == 1:
                 ax = axs[i // cols]  
            else:
                ax = axs[i // cols, i % cols]
                
            if not numerical:                           # bar chart 
                x_data_hist = x_data.groupby(col).size().reset_index(name='count').sort_values(by='count', ascending=True)
                labels, counts = list(x_data_hist[col]), list(x_data_hist['count'])
                ax.bar(labels, counts, color='teal')
                if len(labels) > 15:
                    ax.set_xticklabels(labels, rotation=90)  # Rotate labels vertically
                    
            else:                                       # histogram 
                ax.hist(x_data[col], bins=30, color='teal')
                
            ax.set_title(col)
            ax.set_xlabel('')
            ax.tick_params(labelsize=9)

       
        if not numerical:
            plt.subplots_adjust(hspace=0.7, wspace=0.3)  # Increase space between subplots
                       
        fig.suptitle(title, fontsize=18)
        fig.subplots_adjust(top=0.90)

    warnings.filterwarnings('ignore')

    categ_col = [ col for col in df.columns if df[col].dtype == 'object' ]
    numeric_cols =[ col for col in df.columns if df[col].dtype == 'int64' or df[col].dtype =='float64']
   
    grid_plot_generator(df[categ_col], 1+(len(categ_col)//2), 2, "Distribution of Nominal Columns", figsize=(25,100))
    grid_plot_generator(df[numeric_cols], 1+(len(numeric_cols)//3), 3, "Distribution of Numerical Columns", figsize=(25,100), numerical=True)

def plot_boxplots(df):
    '''
    Boxplot for numerical variables
    '''
    numerical_columns = [ col for col in df.columns if df[col].dtype == 'int64' or df[col].dtype =='float64']
    # Impute missing arrival delays with their median
    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(25, 100))
    for i, numerical_col in enumerate(numerical_columns):
        ax = plt.subplot(1+(len(numerical_columns)//3), 3, i+1)
        ax.boxplot(df.dropna(subset=[numerical_col])[numerical_col])  # drop the rows that has Nans in this column
        ax.set_ylabel(numerical_col)
    plt.tight_layout()
    plt.show()

def plot_side_by_side_boxplots(df):
    '''
    plotting boxplot for the nominal variables categories wrt the target variable
    '''
    categ_col = [ col for col in df.columns if df[col].dtype == 'object' ]

    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(15, 15))
    # plt.subplots_adjust(hspace=0.9)
    for i, col in enumerate(categ_col):
        ax = plt.subplot(len(categ_col) , 1, i+1)
        target_categories = df[col].unique()
        ax.boxplot([df["offering_time"][df[col]==category] for category in target_categories])
        ax.set_xticklabels(df[col].unique(), rotation=90 )
        ax.set_ylabel("offering_time")
        ax.set_xlabel(col)
    plt.tight_layout()
    plt.show()

def association_bet_numeric_columns(df, method='pearson'):
    """
    Creates a heatmap to visualize the association between columns in a DataFrame.

    Parameters
    ----------
    df:  pandas data frame 
    method : str, optional
        The correlation method to be used. Default is 'pearson'.
    """
    numerical_columns = [ col for col in df.columns if df[col].dtype == 'int64' or df[col].dtype =='float64']

    df_numeric = df.loc[:, numerical_columns]
    corr = df_numeric.corr(method=method)

    # Create a heatmap of the correlation matrix
    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 200        # increase plot resolution
    plt.figure(figsize=(6, 5))  # Increase the figsize as desired
    sns.heatmap(corr, annot=True, cmap='cool', center=0, linewidths=0.5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.show()

def nominal_columns_dependency(df):
    """
    Calculates the dependency between nominal columns in a DataFrame using the chi-square test.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the entire data.
    Returns
    -------
    pandas.DataFrame
        A DataFrame showing the p-values of the chi-square test between each pair of nominal columns.
    """
    categ_col = [ col for col in df.columns if df[col].dtype == 'object' ]

    p_value_df = pd.DataFrame(index=categ_col, columns=categ_col)

    for col_i in categ_col:
        for col_j in categ_col:
            contingency_table = pd.crosstab(df[col_i], df[col_j]) # based on frequency
            # Perform chi-square test
            chi2, p_value, dof, ex = chi2_contingency(contingency_table)
            p_value_df.at[col_i, col_j] = p_value

    return p_value_df

def visualize_continuous_data(df):
    '''
    Plot all possible 4c2 pairs of continuous features in a scatter plot grid of 2x3. If HighD is True, 
    then plot all possible 4c3 pairs in a 1x4 grid of 3D scatter plots.
    '''

    numerical_columns = [col for col in df.columns if df[col].dtype == 'int64' or df[col].dtype =='float64']
    df_numeric = df.loc[:, numerical_columns]


    combinations = list(itertools.combinations(numerical_columns, 2)) #combinations so to avoid repetition of pairs
    plt.style.use('dark_background')
    fig, axes = plt.subplots(len(combinations)//3, 3, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.9)
    axes = axes.flatten()
    for i, (col1, col2) in enumerate(combinations):

        # for better visuals, lets remove outliers
        lower1,upper1=calc_outliers_range(df_numeric, col1)
        lower2,upper2=calc_outliers_range(df_numeric,col2)
        df_no_outliers=df_numeric[(df_numeric[col1]>lower1) & (df_numeric[col1]<upper1) & (df_numeric[col2]>lower2) & (df_numeric[col2]<upper2)]

        axes[i].scatter(df_no_outliers[col1], df_no_outliers[col2], color='pink', s=2, alpha=0.7, marker='.')
        axes[i].title.set_color('white')
        axes[i].set_xlabel(col1)
        axes[i].set_ylabel(col2)
        axes[i].legend()

    plt.show()





