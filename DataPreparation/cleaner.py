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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler

def handle_nulls(x_data,y_data,module_dir,method='mix',split="train"):
    '''
    Deals with nans in the dataframe
    
    Parameters
    ----------
    x_data : pandas.DataFrame
        The DataFrame containing the features.

    y_data : pandas.Series or pandas.DataFrame
        The DataFrame or Series containing the target data, used when deleting rows due to nans.
    
    method: what action to take on nans. 
            ['drop', 'ffill','mode' , 'median' , 'mean', 'mix]
            if mix  is given, then a generic way of impuation is used.
    Returns
    -------
    None. everything is done inplace
    '''

    if method=='drop':
        # Drop rows with nulls in x_data and adjust y_data accordingly
        mask = x_data.notnull().all(axis=1)
        x_data.dropna(axis=0, how='any', inplace=True)
        y_data.drop(y_data.index[~mask], inplace=True)  # Drop corresponding rows in y_data

    if method=='ffill':
        x_data.fillna(method=method, inplace=True)

    if method=='mode':
        if split=="train" or split=='all':
            modes={}
            for col in x_data.columns:
                mode=x_data[col].mode()[0]
                modes[col] = mode
                x_data[col].fillna(mode, inplace=True) # mode could be more than one values, so we use the 1st

            with open(os.path.join(module_dir, '../Saved') + '/null_modes.pkl', 'wb') as f:
                pickle.dump(modes, f)
        if split=="test":
             with open(os.path.join(module_dir, '../Saved') + '/null_modes.pkl', 'rb') as f:
                modes = pickle.load(f)
                for col in x_data.columns:
                    x_data[col].fillna(modes[col], inplace=True)

    if method=='median':
        if split=="train"or split=='all':
            medians=x_data.median()
            x_data.fillna(medians, inplace=True)
            with open(os.path.join(module_dir, '../Saved') + '/null_medians.pkl', 'wb') as f:
                pickle.dump(medians, f)

        if split=="test":
            with open(os.path.join(module_dir, '../Saved') + '/null_medians.pkl', 'rb') as f:
                medians = pickle.load(f)
            x_data.fillna(medians, inplace=True)

    if method=='mean':
        if split=="train" or split=='all':
            means=x_data.mean()
            x_data.fillna(means, inplace=True)
            with open(os.path.join(module_dir, '../Saved') + '/null_means.pkl', 'wb') as f:
                pickle.dump(means, f)

        if split=="test":
            with open(os.path.join(module_dir, '../Saved') + '/null_means.pkl', 'rb') as f:
                means = pickle.load(f)
            x_data.fillna(means, inplace=True)

    if method=='mix':
        numerical_columns = [ col for col in x_data.columns if x_data[col].dtype == 'int64' or x_data[col].dtype =='float64']
        categ_col = [ col for col in x_data.columns if x_data[col].dtype == 'object' ]
        if split=='train' or split=='all':
            # in this case we handle nulls for categorical different than for numerical
            medians=x_data[numerical_columns].median()
            x_data[numerical_columns] = x_data[numerical_columns].fillna(medians)  # the numericals use median
            modes={}
            for col in categ_col:
                mode=x_data[col].mode()[0]
                modes[col]=mode
                x_data[col].fillna(mode, inplace=True)  # the categoricals use mode
            data=[medians, modes]
            with open(os.path.join(module_dir, '../Saved') + '/null_mix.pkl', 'wb') as f:
                pickle.dump(data, f)

        if split=='test':
            with open(os.path.join(module_dir, '../Saved') + '/null_mix.pkl', 'rb') as f:
                medians, modes = pickle.load(f)

            x_data[numerical_columns] = x_data[numerical_columns].fillna(medians)
            for col in categ_col:
                x_data[col].fillna(modes[col], inplace=True) 

def handle_diverse_categories(df,module_dir, class_ratio=0.001 , column_cardinaltiy=0.005, split='train'):
    '''
    A categorical column with high-cardinality [features with a large number of unique categories].
    These columns cause problems if a category is found in test set and does not exist in training.

    we can deal differently in this case:
        1. delete that column
        2. Change low frequent categories with "Other"

    Parameters
    df: Pandas data frame
    class_ratio : a threshold for the classes in a specific column. Below this threshold, this class needs to be replaced with 'Other'
    column_cardinaltiy: a threshold on the column itself. If this column has a a lot of catrgories compared to the size of the dataset
                        it means that this column is of less useful info. Either we dropp it  or group the minority classes in this col
                        in 'Other' category.
    ----------
    Returns
    -------
    None. Everything is done inplace
    '''

    categ_col = [ col for col in df.columns if df[col].dtype == 'object']
    if split=="train" or split=='all':
        unique_categ={}
        for col in categ_col:
            # The cardinality of this column: Number of classes over the size of training dataset
            cardinality= df[col].nunique()/df.shape[0]

            if cardinality==1: # this column id a unique ID
                df.drop([col], axis=1, inplace=True)

            # if this col has a higher cardinality than the predefined threshold, then there are a lot of classes 
            elif cardinality>column_cardinaltiy:
                ratios=df[col].value_counts(normalize=True) # count the ratio of each class in the column
                minority=ratios[ratios<class_ratio]
                minority=minority.reset_index().rename(columns={"index": col, col: "ratio"}) # a dataframe with the minority classes and their ratio
                minority_classes = minority[col].tolist() # the classes which are below the prededfined ratio
                label='Other'
                df[col].mask(df[col].isin(minority_classes), label, inplace=True) #change the label of minority classes to the new label

            unique_categ[col]=set(df[col])

        with open(os.path.join(module_dir, '../Saved') + '/diverge_categ.pkl', 'wb') as f:
            pickle.dump(unique_categ, f)
    if split=="test":
        with open(os.path.join(module_dir, '../Saved') + '/diverge_categ.pkl', 'rb') as f:
            unique_categ = pickle.load(f)

        for col in categ_col:
            if col not in unique_categ:
                df.drop([col], axis=1, inplace=True)
            else:
                classes=unique_categ[col]
                label='Other'
                df[col].mask(~df[col].isin(classes), label, inplace=True) # replace the unseen category with "Other"

def handle_categories(df, module_dir, encode='Binary', split='train'):
    '''
    Performs encoding on categorical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the entire data.
    
    module_dir : str
        Location of saved parameters (encoder objects).
    
    encode : str
        Type of encoding needed [Binary, OneHot, Ordinal, Frequency].
    
    split : str
        Indicates if encoding is performed on train or test data [train, test, all].
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame after encoding. The function modifies the DataFrame in-place.
    '''
    categ_col = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() > 2]
    
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() == 2:
            df[col] = df[col].map({df[col].unique()[0]: 0, df[col].unique()[1]: 1})
    
    if encode == 'Ordinal':
        if split == 'train' or split == 'all':
            label_encoders = {}
            for col in categ_col:
                df[col] = df[col].astype(str)
                encoder = ce.OrdinalEncoder(cols=[col])
                df[col] = encoder.fit_transform(df[col])
                label_encoders[col] = encoder

            # Save the encoders
            with open(os.path.join(module_dir, '../Saved') + '/label_encoders.pkl', 'wb') as f:
                pickle.dump(label_encoders, f)

        elif split == 'test':
            # Load the encoders
            with open(os.path.join(module_dir, '../Saved') + '/label_encoders.pkl', 'rb') as f:
                label_encoders = pickle.load(f)

            for col in categ_col:
                df[col] = df[col].astype(str)
                df[col] = label_encoders[col].transform(df[col])

    elif encode == 'OneHot':
        if split == 'train' or split == 'all':
            onehot_encoder = pd.get_dummies(df[categ_col], prefix=categ_col)
            df = pd.concat([df, onehot_encoder], axis=1)
            df.drop(categ_col, axis=1, inplace=True)

            # Save column names for one-hot encoded features
            with open(os.path.join(module_dir, '../Saved') + '/onehot_columns.pkl', 'wb') as f:
                pickle.dump(onehot_encoder.columns.tolist(), f)

        elif split == 'test':
            # Load the column names for one-hot encoded features
            with open(os.path.join(module_dir, '../Saved') + '/onehot_columns.pkl', 'rb') as f:
                onehot_columns = pickle.load(f)

            # Create dummy variables for test set (to ensure same columns)
            onehot_encoder = pd.get_dummies(df[categ_col], prefix=categ_col)
            df = pd.concat([df, onehot_encoder], axis=1)
            df.drop(categ_col, axis=1, inplace=True)

            # Ensure that the test set has the same columns as the training set
            for col in onehot_columns:
                if col not in df.columns:
                    df[col] = 0  # Add missing column with 0 value

            df = df[onehot_columns]  # Reorder columns to match training set

    elif encode == 'Frequency':
        if split == 'train' or split == 'all':
            freq_encoders = {}
            for col in categ_col:
                freq_encoding = df[col].value_counts() / len(df)
                df[col] = df[col].map(freq_encoding)
                freq_encoders[col] = freq_encoding

            # Save frequency encoders
            with open(os.path.join(module_dir, '../Saved') + '/freq_encoders.pkl', 'wb') as f:
                pickle.dump(freq_encoders, f)

        elif split == 'test':
            # Load frequency encoders
            with open(os.path.join(module_dir, '../Saved') + '/freq_encoders.pkl', 'rb') as f:
                freq_encoders = pickle.load(f)

            for col in categ_col:
                df[col] = df[col].map(freq_encoders[col])

    elif encode == 'Binary':
        if split == 'train' or split == 'all':
            encoder = ce.BinaryEncoder(cols=categ_col)
            df = encoder.fit_transform(df)
            with open(os.path.join(module_dir, '../Saved') + '/binary_encoder.pkl', 'wb') as f:
                pickle.dump(encoder, f)
        elif split == 'test':
            with open(os.path.join(module_dir, '../Saved') + '/binary_encoder.pkl', 'rb') as f:
                encoder = pickle.load(f)
            df = encoder.transform(df)

    return df

def handle_numericals(df,module_dir,method="standardize", split="train"):
    '''
    Let the numerical columns all within close scale to avoid the common probelms(e.g. slow convergence, sensitivity to scale)
    Parameters
    df:
                Pandas data frame
    module_dir:
                location of saved parameters
    method:
                either standardize  or normalize
    split:
                either train or test
    -------
    Returns
    -------
    None. Everything is done inplace
    '''
    numerical_columns = [ col for col in df.columns if df[col].dtype == 'int64' or df[col].dtype =='float64']
    if (split=='train' or split=='all') and method=='standardize':
        means, stds = [], []
        for col in numerical_columns:
                means.append(df[col].mean())
                stds.append(df[col].std())
                if df[col].std()!=0:
                    df[col] = (df[col] - df[col].mean())/df[col].std()
        # save the means and stds for later use
        np.save(os.path.join(module_dir, '../Saved') + '/means.npy', means)
        np.save(os.path.join(module_dir, '../Saved') + '/stds.npy', stds)

    if split=='test' and method=='standardize':
        means = np.load(os.path.join(module_dir, '../Saved') + '/means.npy')
        stds = np.load(os.path.join(module_dir, '../Saved') + '/stds.npy')
        for i,col in enumerate(numerical_columns):
            if stds[i]!=0:
                df[col] = (df[col]- means[i])/stds[i]

    if (split=='train' or split=='all') and method=='normalize':
        mins,maxs=[],[]
        for col in numerical_columns:
            min_val,max_val = df[col].min() ,df[col].max()
            mins.append(min_val)
            maxs.append(max_val)
            if min_val != max_val:
                df[col] = (df[col] - min_val)/(max_val - min_val)
        # save the mins and maxs for later use
        np.save(os.path.join(module_dir, '../Saved') + '/mins.npy', mins)
        np.save(os.path.join(module_dir, '../Saved') + '/maxs.npy', maxs)

    if split=='test' and method=='normalize':
        mins = np.load(os.path.join(module_dir, '../Saved') + '/mins.npy')
        maxs = np.load(os.path.join(module_dir, '../Saved') + '/maxs.npy')
        for i,col in enumerate(numerical_columns):
            if maxs[i] != mins[i]:
                df[col] = (df[col] - mins[i])/(maxs[i] - mins[i])

def handle_outliers(x_data, y_data, module_dir, method='median', split="train",skip=[]):
    '''
    Handles outliers in the dataset.
    
    Parameters
    ----------
    x_data : pandas.DataFrame
        The DataFrame containing the feature data.

    y_data : pandas.Series or pandas.DataFrame
        The DataFrame or Series containing the target data, used when deleting rows due to outliers.

    method: str
        The method to handle outliers. Options are:
        ['delete', 'cap', 'median', 'log_transform']

    split: str
        Whether to apply handling on the 'train' or 'test' split.
    
    module_dir: str
        The directory where the metrics (like thresholds or medians) are saved.

    Returns
    -------
    x_data : pandas.DataFrame
        The DataFrame with outliers handled.
    
    y_data : pandas.Series or pandas.DataFrame
        The target data with rows corresponding to deleted outliers removed (if 'delete' method is used).
    '''
    
    numerical_columns = [col for col in x_data.columns if x_data[col].dtype in ['int64', 'float64']]

    # Calculate or load the outlier ranges once
    if split == 'train' or split == 'all':
        outlier_ranges = {}
        for column_name in numerical_columns:
            lower, upper = calc_outliers_range(x_data, column_name)
            outlier_ranges[column_name] = (lower, upper)
        
        # Save the calculated outlier ranges
        with open(os.path.join(module_dir, '../Saved') + '/outlier_ranges.pkl', 'wb') as f:
            pickle.dump(outlier_ranges, f)

    elif split == 'test':
        # Load the outlier ranges calculated from the training set
        with open(os.path.join(module_dir, '../Saved') + '/outlier_ranges.pkl', 'rb') as f:
            outlier_ranges = pickle.load(f)

    # Apply the chosen method for handling outliers
    if method == 'delete':
        indices_to_drop = []
        for column_name in numerical_columns:
            if column_name in skip:
                continue
            lower, upper = outlier_ranges[column_name]
            mask = x_data[(x_data[column_name] < lower) | (x_data[column_name] > upper)].index
            indices_to_drop.extend(mask)
        
        # Drop rows with outliers in x_data and corresponding y_data
        indices_to_drop = list(set(indices_to_drop))  # Remove duplicates
        x_data.drop(indices_to_drop, inplace=True)
        y_data.drop(indices_to_drop, inplace=True)

    elif method == 'cap':
        for column_name in numerical_columns:
            if column_name in skip:
                continue
            lower, upper = outlier_ranges[column_name]
            x_data[column_name] = np.where(x_data[column_name] > upper, upper, x_data[column_name])
            x_data[column_name] = np.where(x_data[column_name] < lower, lower, x_data[column_name])

    elif method == 'median':
        if split == 'train' or split == 'all':
            medians = {}
            for column_name in numerical_columns:
                lower, upper = outlier_ranges[column_name]
                mask = (x_data[column_name] < lower) | (x_data[column_name] > upper)
                x_data.loc[mask, column_name] = x_data[column_name].median()
                medians[column_name] = x_data[column_name].median()

            # Save the medians for use during testing
            with open(os.path.join(module_dir, '../Saved') + '/outlier_medians.pkl', 'wb') as f:
                pickle.dump(medians, f)

        elif split == 'test':
            # Load medians from the training set
            with open(os.path.join(module_dir, '../Saved') + '/outlier_medians.pkl', 'rb') as f:
                medians = pickle.load(f)

            for column_name in numerical_columns:
                lower, upper = outlier_ranges[column_name]
                mask = (x_data[column_name] < lower) | (x_data[column_name] > upper)
                x_data.loc[mask, column_name] = medians[column_name]

    elif method == 'log_transform':
        
        # Log transform does not need outlier range calculation, just apply it directly
        for column_name in numerical_columns:
            if column_name in skip:
                continue
            if (x_data[column_name] < -1).any():
                print(f"Warning: Negative values detected in {column_name} which will result in NaNs.")
            x_data[column_name] = np.log(x_data[column_name] + 1)

    return x_data, y_data

def handle_oversampling(x_data, y_data,split ,method='smot', **kwargs):
    '''
    Handles class imbalance in the target variable by oversampling the minority class.
    
    Parameters
    ----------
    x_data : pandas.DataFrame
        The DataFrame containing the feature data.
    
    y_data : pandas.Series
        The Series containing the target variable.
    
    method : str
        The method to handle class imbalance ['smot', 'adasyn', 'random_oversampling'].
    
    Returns
    -------
    x_data : pandas.DataFrame
        The DataFrame with oversampling applied.
    
    y_data : pandas.Series
        The Series with oversampling applied.
    '''
    if split!='train':
        return x_data, y_data
    
    if method == 'smot':
        smot = SMOTE(sampling_strategy='minority')
        x_data, y_data = smot.fit_resample(x_data, y_data)
    
    elif method == 'adasyn':
        adasyn = ADASYN(sampling_strategy='minority')
        x_data, y_data = adasyn.fit_resample(x_data, y_data)
    
    elif method == 'random_oversampling':
        oversample = RandomOverSampler(sampling_strategy='minority')
        x_data, y_data = oversample.fit_resample(x_data, y_data)
    
    return x_data, y_data

def apply_pca(x_data, module_dir, variance_threshold=0.95, split="train"):
    '''
    Applies PCA to reduce dimensionality.

    Parameters
    ----------
    x_data : pandas.DataFrame
        The DataFrame containing the feature data, already preprocessed.

    variance_threshold: float
        The amount of variance to retain when applying PCA (default is 0.95).

    split: str
        Whether to apply PCA on the 'train' or 'test' split.

    module_dir: str
        The directory where the PCA model is saved.

    Returns
    -------
    x_data_pca : pandas.DataFrame
        The DataFrame with PCA applied, containing reduced dimensions.
    '''
    if variance_threshold== None:
        return x_data
    if split == 'train' or split == 'all':
        # Apply PCA and fit the model on training data
        pca = PCA(n_components=variance_threshold)
        x_data_pca = pca.fit_transform(x_data)
        
        # Save the PCA model for future use
        with open(os.path.join(module_dir, '../Saved') + '/pca_model.pkl', 'wb') as f:
            pickle.dump(pca, f)

    elif split == 'test':
        # Load the saved PCA model from the training phase
        with open(os.path.join(module_dir, '../Saved') + '/pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)

        # Apply PCA transformation on the test data
        x_data_pca = pca.transform(x_data)

    else:
        raise ValueError("Invalid split parameter. Use 'train','test' or 'all' .")
    
    return x_data_pca

def read_data(split="train", nulls="mix",outliers="cap", standardize="standardize",encode='Binary',pca_threshold=None,skip=[],oversample='smot',**kwargs):
    '''
    Reads the data from the CSV file and performs data cleaning and preprocessing.
    
    Parameters
    ----------
    split : str
        The split of the data to read ['train', 'test', 'all']. Default is 'train'.
    
    nulls : str
        The method to handle null values ['drop', 'ffill', 'mode', 'median', 'mean', 'mix']. Default is 'mix'.
    
    outliers : str
        The method to handle outliers ['delete', 'cap', 'median', 'log_transform']. Default is 'cap'.
        
    standardize : str   
        The method to standardize numerical data ['standardize', 'normalize']. Default is 'standardize'.
    
    encode : str
        The method to encode categorical data ['Binary', 'OneHot', 'Ordinal', 'Frequency']. Default is 'Binary'.
    
    Returns
    -------
    x_data : pandas.DataFrame
        The DataFrame containing the cleaned and preprocessed features.
    
    y_data : pandas.Series
        The Series containing the target variable.
    '''
    def process(x_data,y_data,module_dir,split="train", nulls="mix",outliers="cap", standardize="standardize",encode='Binary',pca_threshold=None,skip=[],oversample='smot',**kwargs):
       
        # data cleaning stage for all columns
        handle_nulls(x_data,y_data,module_dir,split=split,method=nulls)

        # transformations for numerical data
        x_data,y_data=handle_outliers(x_data,y_data,module_dir, method=outliers, split="train",skip=skip)
        handle_numericals(x_data,module_dir,method=standardize, split=split)  #the order of calling this and the above function matters

        # transformations for categorical data
        handle_diverse_categories(x_data,module_dir,split=split)
        x_data=handle_categories(x_data,module_dir,split=split, encode=encode) #the order of calling this and the above function matters
        
        if pca_threshold!=None:
            x_data=apply_pca(x_data,module_dir,variance_threshold=pca_threshold,split=split)
        
        x_data, y_data=handle_oversampling(x_data, y_data,split=split, method=oversample)
            
        return x_data, y_data

    module_dir = os.path.dirname(__file__)
    if split == "train" or split=="val":    path = os.path.join(module_dir, '../DataFiles/train.csv')
    elif split == "test":    path = os.path.join(module_dir, '../DataFiles/test.csv')
    elif split == "all":    path = os.path.join(module_dir, '../DataFiles/cell2celltrain.csv')
           
    target_variable='Churn'

    df = pd.read_csv(path)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # map the target variable to 0 and 1 for binary classification
    df[target_variable] = df[target_variable].map({'Yes': 1, 'No': 0})
    y_data = df[target_variable]
    x_data = df.drop([target_variable,"CustomerID"], axis=1)
    
    if split=='val':
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
        x_train, y_train= process( x_train, y_train,module_dir,split="train", nulls=nulls,outliers=outliers, standardize=standardize,encode=encode,pca_threshold=pca_threshold,skip=skip, oversample=oversample)
        x_test, y_test= process(x_test, y_test,module_dir,split="test", nulls=nulls,outliers=outliers, standardize=standardize,encode=encode,pca_threshold=pca_threshold,skip=skip, oversample=oversample)
        return x_train, x_test, y_train, y_test
    else:
        x_data, y_data = process(x_data, y_data,module_dir,split=split, nulls=nulls,outliers=outliers, standardize=standardize,encode=encode,pca_threshold=pca_threshold,skip=skip, oversample=oversample)
        return x_data, y_data, None, None


