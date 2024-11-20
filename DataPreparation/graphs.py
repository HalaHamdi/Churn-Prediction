import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import itertools

def plot_stacked_bar_churn_vs_categorical(data, target_column='Churn', threshold=10):
    """
    Plots stacked bar plots for each categorical feature against the churn column.
    Automatically detects categorical columns based on data type and number of unique values.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    target_column (str): The target column indicating churn (default is 'churn').
    threshold (int): Maximum unique values to consider a column as categorical.
    """

    # Convert churn column to binary integers
    data[target_column] = data[target_column].map({'Yes': 1, 'No': 0})

    # Automatically detect categorical columns based on data type and number of unique values
    categorical_columns = [col for col in data.select_dtypes(include=['object', 'category']).columns
                           if data[col].nunique() <= threshold and col != target_column]

    if target_column not in data.columns:
        raise ValueError(f"Churn column '{target_column}' not found in DataFrame")


    # Set the plot style and background
    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 200  # Increase plot resolution

    num_columns = len(categorical_columns)

    if num_columns == 0:
        print("No categorical columns found to plot.")
        return
    
    # Calculate number of rows needed to fit 3 plots in each row
    nrows = (num_columns + 2) // 3  # Ceiling division to ensure enough rows

    fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5 * nrows))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    for i, col in enumerate(categorical_columns):
        # Calculate value counts for churn = 1 and churn = 0
        churn_data = data.groupby([col, target_column]).size().unstack(fill_value=0)
        

        # Prepare for plotting
        labels = churn_data.index

        # Create the stacked bar plot
        ax = axs[i]  # Get the current axis

        churn_1 = churn_data.get(1, 0)  # Default to 0 if not found
        churn_0 = churn_data.get(0, 0)  # Default to 0 if not found

        ax.bar(labels, churn_1, color='#44a5c2', label='Churn')
        ax.bar(labels, churn_0, color='#024b7a', bottom=churn_1, label='No Churn')

        # Set the title and labels
        ax.set_title(col, fontsize=12)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.tick_params(axis='x', rotation=90)
        ax.legend()

    # Hide any unused subplots
    for j in range(num_columns, len(axs)):
        axs[j].axis('off')  # Turn off the unused axes

    fig.suptitle('Churn vs Categorical Features', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_box_churn_vs_numerical(data, target_column='Churn'):
    """
    Plots box plots for each numerical feature against the churn column.
    Automatically detects numerical columns based on data type.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    target_column (str): The target column indicating churn (default is 'churn').
    """
    
    # Convert churn column to binary integers if it contains 'Yes'/'No'
    if data[target_column].dtype == 'object':
        data[target_column] = data[target_column].map({'Yes': 1, 'No': 0})

    # Automatically detect numerical columns based on data type
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude the churn column from numerical columns if present
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)

    if target_column not in data.columns:
        raise ValueError(f"Churn column '{target_column}' not found in DataFrame")


    num_columns = len(numerical_columns)

    if num_columns == 0:
        print("No numerical columns found to plot.")
        return
    
    # Calculate number of rows needed to fit 3 plots in each row
    nrows = (num_columns + 2) // 3  # Ceiling division to ensure enough rows

    fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5 * nrows))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    for i, col in enumerate(numerical_columns):
        # Create the box plot
        ax = axs[i]  # Get the current axis
        data.boxplot(column=col, by=target_column, ax=ax, grid=False)

        # Set the title and labels
        ax.set_title(f'{col} vs Churn', fontsize=12)
        ax.set_xlabel('Churn', fontsize=10)
        ax.set_ylabel(col, fontsize=10)

    # Hide any unused subplots
    for j in range(num_columns, len(axs)):
        axs[j].axis('off')  # Turn off the unused axes

    fig.suptitle('Churn vs Numerical Features', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_pairplot_churn_vs_numerical(data, target_column='Churn'):
    """
    Plots pair plots for each numerical feature against the churn column.
    Automatically detects numerical columns based on data type.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    target_column (str): The target column indicating churn (default is 'churn').
    """
    
    # Convert churn column to binary integers if it contains 'Yes'/'No'
    if data[target_column].dtype == 'object':
        data[target_column] = data[target_column].map({'Yes': 1, 'No': 0})

    # Automatically detect numerical columns based on data type
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude the churn column from numerical columns if present
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)

    if target_column not in data.columns:
        raise ValueError(f"Churn column '{target_column}' not found in DataFrame")

    print(f"Detected numerical columns: {numerical_columns}")
    print(f"Churn column unique values: {data[target_column].unique()}")

    if len(numerical_columns) == 0:
        print("No numerical columns found to plot.")
        return
    
    # Include the churn column in the data for pair plotting
    data_subset = data[numerical_columns + [target_column]]
    
    # Set the plot style
    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 150  # Higher resolution for the plot
    
    # Create the pair plot using seaborn
    sns.pairplot(data_subset, hue=target_column, palette='coolwarm', diag_kind='kde', markers=["o", "s"])

    # Set the overall title for the pair plot
    plt.suptitle('Pair Plot: Numerical Features vs Churn', fontsize=16, y=1.02)
    
    plt.show()

def plot_pairplot_high_correlation(data, correlation_threshold=0.5, ncols=3):
    """
    Plots pair plots for numerical feature pairs that have a high correlation.
    Automatically detects numerical columns based on data type.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    correlation_threshold (float): The correlation threshold to filter feature pairs (default is 0.5).
    ncols (int): Number of columns for the grid layout (default is 3).
    """
    
    # Automatically detect numerical columns based on data type
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numerical_columns) < 2:
        print("Not enough numerical columns found to plot.")
        return

    # Compute the correlation matrix
    correlation_matrix = data[numerical_columns].corr()

    # Create a list to hold pairs of features with high correlation
    high_corr_pairs = []

    # Find pairs of features with correlation above the threshold
    for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                high_corr_pairs.append((numerical_columns[i], numerical_columns[j], correlation_matrix.iloc[i, j]))

    if len(high_corr_pairs) == 0:
        print("No pairs found with high correlation.")
        return

    # Sort pairs by absolute correlation value in descending order
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Set the plot style and size
    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 150  # Higher resolution for the plot

    # Create a grid for the plots
    nrows = int(np.ceil(len(high_corr_pairs) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 4))
    
    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Create pair plots for high correlation pairs
    for idx, pair in enumerate(high_corr_pairs):
        feature1, feature2, corr_value = pair
        
        # Create a scatter plot for each high correlation pair
        sns.scatterplot(data=data, x=feature1, y=feature2, ax=axes[idx], alpha=0.6)
        axes[idx].set_title(f'{feature1} vs {feature2}\nCorr: {corr_value:.2f}', fontsize=10)
        axes[idx].set_xlabel(feature1, fontsize=8)
        axes[idx].set_ylabel(feature2, fontsize=8)
        
        # Remove horizontal and vertical lines (grid)
        axes[idx].grid(False)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout for better fit
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(data, churn_column='churn'):
    """
    Plots a heatmap of the correlation matrix for numerical features against the churn column.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    churn_column (str): The target column indicating churn (default is 'churn').
    """

    # Convert churn column to binary integers if it contains 'Yes'/'No'
    if data[churn_column].dtype == 'object':
        data[churn_column] = data[churn_column].map({'Yes': 1, 'No': 0})

    # Automatically detect numerical columns based on data type
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude the churn column from numerical columns if present
    if churn_column in numerical_columns:
        numerical_columns.remove(churn_column)

    if churn_column not in data.columns:
        raise ValueError(f"Churn column '{churn_column}' not found in DataFrame")


    if len(numerical_columns) == 0:
        print("No numerical columns found to plot.")
        return

    # Create a DataFrame containing only numerical features and the churn column
    data_subset = data[numerical_columns + [churn_column]]

    # Compute the correlation matrix
    correlation_matrix = data_subset.corr()

    # Set the plot style
    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 150  # Higher resolution for the plot

    # Create a heatmap without annotations
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    
    # Set the title
    plt.title('Correlation Heatmap of Numerical Features vs Churn', fontsize=16)
    plt.show()

def plot_facetgrid_churn_vs_categorical(data, target_column='Churn',col2='HasCreditCard',threshold=10):
    """
    Plots facet grids for each categorical feature against the churn column.
    Automatically detects categorical columns based on data type and number of unique values.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    target_column (str): The target column indicating churn (default is 'Churn').
    threshold (int): Maximum unique values to consider a column as categorical.
    """
    warnings.filterwarnings("ignore")
    # Convert churn column to binary integers if it contains 'Yes'/'No'
    if data[target_column].dtype == 'object':
        data[target_column] = data[target_column].map({'Yes': 1, 'No': 0})

    # Automatically detect categorical columns based on data type and number of unique values
    categorical_columns = [col for col in data.select_dtypes(include=['object', 'category']).columns
                           if data[col].nunique() <= threshold and col != target_column]

    if target_column not in data.columns:
        raise ValueError(f"Churn column '{target_column}' not found in DataFrame")

    # Set the plot style and background
    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 200  # Increase plot resolution

    num_columns = len(categorical_columns)

    if num_columns == 0:
        print("No categorical columns found to plot.")
        return

    # Create a FacetGrid for each categorical feature
    for col in categorical_columns:
        # Check if the target column is in the current categorical column
        if col == target_column:
            continue

        # Create the FacetGrid with countplot
        # g = sns.FacetGrid(data, col=col, hue=target_column, col_wrap=1, height=4, aspect=2)
        g = sns.FacetGrid(data, col=col, hue=target_column)
        g.map(sns.countplot, col2)  # Change 'HasCreditCard' to the feature of interest if needed
        g.add_legend()
        
        # Set the title
        g.fig.suptitle(f'Churn vs {col}', fontsize=16, y=1.05)

    plt.show()

def plot_kde_churn_vs_numerical(data, target_column='Churn'):
    """
    Plots KDE plots for each numerical feature against the churn column.
    Automatically detects numerical columns based on data type.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    target_column (str): The target column indicating churn (default is 'Churn').
    """

    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Convert churn column to binary integers if it contains 'Yes'/'No'
    if data[target_column].dtype == 'object':
        data[target_column] = data[target_column].map({'Yes': 1, 'No': 0})

    # Automatically detect numerical columns based on data type
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude the churn column from numerical columns if present
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)

    if target_column not in data.columns:
        raise ValueError(f"Churn column '{target_column}' not found in DataFrame")

    num_columns = len(numerical_columns)

    if num_columns == 0:
        print("No numerical columns found to plot.")
        return
    
    # Calculate number of rows needed to fit 3 plots in each row
    nrows = (num_columns + 2) // 3  # Ceiling division to ensure enough rows

    fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5 * nrows))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    for i, col in enumerate(numerical_columns):
        # Create the KDE plots
        sns.kdeplot(data[data[target_column] == 1][col], label='Churn', ax=axs[i], shade=True)
        sns.kdeplot(data[data[target_column] == 0][col], label='No Churn', ax=axs[i], shade=True)

        # Set the title and labels
        axs[i].set_title(f'KDE Plot of {col} by Churn', fontsize=12)
        axs[i].set_xlabel(col, fontsize=10)
        axs[i].set_ylabel('Density', fontsize=10)
        axs[i].legend()

    # Hide any unused subplots
    for j in range(num_columns, len(axs)):
        axs[j].axis('off')  # Turn off the unused axes

    fig.suptitle('Churn vs Numerical Features - KDE Plots', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_pairplot_columns(data, columns, y_column='Churn', ncols=3):
    """
    Plots pair plots for every combination of the given columns with color coding based on the y_column.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    columns (list): List of columns to plot.
    y_column (str): Column name for color coding (default is 'Churn').
    ncols (int): Number of columns for the grid layout (default is 3).
    """
    if data[y_column].dtype == 'object':
        data[y_column] = data[y_column].map({'Yes': 1, 'No': 0})
        
    # Define colors based on the class
    colors = {0: '#667BC6', 1: '#EF5A6F'}
    
    # Generate all unique pairs of columns
    pairs = list(itertools.combinations(columns, 2))
    
    # Set the plot style and size
    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 150  # Higher resolution for the plot

    # Create a grid for the plots
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 4))
    
    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Create scatter plots for each pair
    for idx, (x_col, y_col) in enumerate(pairs):
        sns.scatterplot(data=data, x=x_col, y=y_col, hue=y_column, palette=colors, ax=axes[idx], alpha=0.8)
        axes[idx].set_title(f'{x_col} vs {y_col}', fontsize=10)
        axes[idx].set_xlabel(x_col, fontsize=8)
        axes[idx].set_ylabel(y_col, fontsize=8)
        
        # Remove horizontal and vertical lines (grid)
        axes[idx].grid(False)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout for better fit
    plt.tight_layout()
    plt.show()