import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def count_missing_values(df):
    """
    Count the number of missing values in each column of a PySpark DataFrame.

    Parameters:
        df : Pandas DataFrame to count missing values in.

    Returns:
        pandas.core.frame.DataFrame: A DataFrame showing the count of missing values for each column.
    """
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    return pd.DataFrame({'Missing Count': missing_counts}).T

def count_unique_elements_and_types(df):
    """
    Count the number of unique elements in each column of a Pandas DataFrame
    and get the data type of each column.

    Parameters:
        df : pandas.DataFrame
            DataFrame to analyze.

    Returns:
        pandas.DataFrame: A DataFrame showing the count of unique elements
                          and the data type for each column.
    """
    unique_counts = df.nunique()  # Count of unique elements
    data_types = df.dtypes        # Data types of each column

    # Create a new DataFrame to hold the results
    result_df = pd.DataFrame({
        'Unique Count': unique_counts,
        'Data Type': data_types
    }).T

    return result_df

def count_duplicate_rows(df):
    """
    Count the number of duplicate rows in a Pandas DataFrame.

    Parameters:
        df : Pandas DataFrame to count duplicate rows in.

    Returns:
        pandas.core.frame.DataFrame: A DataFrame showing the count of duplicate rows.
    """
    duplicate_counts = df.duplicated(keep=False).sum()  # Count total duplicates, including all occurrences
    return pd.DataFrame({'Duplicate Row Count': [duplicate_counts]})

def calc_outliers_range(df, column):
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR= Q3-Q1
    lower_limit= Q1 - 1.5*IQR
    upper_limit= Q3 + 1.5*IQR
    return lower_limit, upper_limit
    
def get_outliers(df, column):
    '''
    For a specific column in Pandas data frame, return the outliers upper and lower boundaries
    '''
    
    lower_limit,upper_limit=calc_outliers_range(df,column)
    lower_outliers_df=df[(df[column] < lower_limit )]
    upper_outliers_df= df[(df[column]> upper_limit)]
    return lower_outliers_df, upper_outliers_df

def count_outliers(df):
    '''
    Return a statistics of numeric values. It counts how many outliers in each numeric column in a dataframe.
    '''
    count_lower, count_upper, percentage={} , {}, {}
    numerical_columns = [col for col in df.columns if df[col].dtype == 'int64' or df[col].dtype =='float64']

    for col in numerical_columns:
        lower_outliers, upper_outliers = get_outliers(df, col)
        count_lower[col] = lower_outliers.shape[0]
        count_upper[col] = upper_outliers.shape[0]

        percentage[col]= round(((lower_outliers.shape[0]+upper_outliers.shape[0])/ df.shape[0])*100,2) 

    return pd.DataFrame({'Lower Outliers Count': count_lower,'Upper Outliers Count': count_upper ,"Outliers Percentage (%)": percentage}).T

def numerical_statistics(df):
    numerical_cols = [ col for col in df.columns if df[col].dtype == 'int64' or df[col].dtype =='float64']
    return df.loc[:, numerical_cols].describe().style.set_sticky(axis="index")

def vif_analysis(x_data):
    vif_data = pd.DataFrame()
    vif_data["feature"] = x_data.columns
    vif_data["VIF"] = [variance_inflation_factor(x_data.values, i) for i in range(len(x_data.columns))]
    # Sort the VIF data by VIF values in descending order
    vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)

    vif_dict = vif_data.set_index('feature')['VIF'].to_dict()
    return vif_dict