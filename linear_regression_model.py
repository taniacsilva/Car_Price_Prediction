# Import Libraries
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


#Data Preparation
def data_preparation (filename):
    """This function imports the data from the csv file and performs data preparation

        Args:
            file_name (str): The filename

        Return:
            data (pandas.DataFrame): The pandas Dataframe
    """
    # Load the data
    data = pd.read_csv('car.csv')
    
    # Cleaning the names of the columns
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    
    # Cleaning the columns content
    string_col=list(data.dtypes[data.dtypes == 'object'].index)

    for col in string_col:
        data[col]=data[col].str.lower().str.replace(" ", "_")

    return data, string_col

def exploratory_data_analysis (data, string_col):
    """This function makes some exploratory data analysis

        Args:
            data (pandas.DataFrame): The pandas Dataframe

        Return:
            
    """
    data_columns = data.columns
    for col in data_columns:
        print("Column ", col)
        print(data[col].unique()[:5])
        print(data[col].nunique())

    # Plot for price distribution
    plot = sns.histplot(data.msrp [data.msrp < 100000], bins = 50)

    # Normalization
    # Add 1 to ensure log(0) doesn't happen
    price_logs = np.log1p(data.msrp)

    # Plot for price distribution (normalization)
    plot_log = sns.histplot(price_logs [price_logs < 100000], bins = 50)
    
    # Missing Values
    data.isnull().sum()

    return plt.show()


def split_train_val_test(data, val_s, test_s):
    """ This function splits our dataset into train, validation and test sets
        
        Args: 
            data (pandas.DataFrame): The pandas Dataframe
            val_s (float): The size of the validation set defined by the user
            test_s (float): The size of the test set defined by the user 

        Return: 
            x_train (pandas.DataFrame): Dataframe that includes the explanatory variables for the train set
            x_val (pandas.DataFrame): Dataframe that includes the explanatory variables for the validation set
            x_test (pandas.DataFrame): Dataframe that includes the explanatory variables for the test set
            y_train (numpy array): Array that includes the objective variable for the train set
            y_val (numpy array): Array that includes the objective variable for the validation set
            y_test (numpy array): Array that includes the objective variable for the test set
    """
    n = len(data)
    n_val = int(n*val_s)
    n_test = int(n*test_s)
    n_train = n - n_val - n_test

    # Creating the dataframes
    idx = np.arange(n)
    
    data.iloc[idx[:10]]

    np.random.seed(2) # to make sure that is reproducible
    np.random.shuffle(idx)

    x_train = data.iloc[idx[:n_train]]
    x_val = data.iloc[idx[n_train:n_train+n_val]]
    x_test = data.iloc[idx[n_train+n_val:]]

    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)

    y_train = np.log1p(x_train.msrp.values)
    y_val = np.log1p(x_val.msrp.values)
    y_test = np.log1p(x_test.msrp.values)

    del x_train['msrp']
    del x_val['msrp']
    del x_test['msrp']

    return x_train, x_val, x_test, y_train, y_val, y_test


def linear_regression_model (x_train, y_train):
    """ This function trains the model
            
            Args: 
                x_train (pandas.DataFrame): Dataframe that includes the explanatory variables for the train set
                y_train (numpy array): Array that includes the objective variable for the train set
            
            Return:
                w0 (float): constant obtained by training the linear regression model
                w (numpy array): array that contains the linear regression coefficients
    """    
    ones = np.ones(x_train.shape[0])
    x_train = np.column_stack([ones, x_train])

    XTX = x_train.T.dot(x_train)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(x_train.T).dot(y_train)

    w0 = w_full[0]
    w = w_full[1:]

    return w0, w     

def prepare_x(df, base):
    """ This function select the columns to be filled by 0 in case of existent missing values
    
        Args:
            df (pandas.DataFrame): Dataframe that have the columns that will be subject to this filling 
            base (list): List that contains the columns that will be selected
        
        Return:
            x_train_base (numpy array): Array that contains df values for the columns selected in base 
                                        with no missing values
    """
    df_num = df[base] 
    df_num = df_num.fillna(0) # Missing Values filled with 0
    x_train_base = df_num.values

    return x_train_base


def results_comparison_plot (y_pred, y_train):
    """ This function plots predictions and actual values and allows the comparison

            Args:
                y_test (numpy array): Array that includes the predictions for the train set
                y_train (numpy array): Array that includes the actual values for the train set
            
            Return:
    """
    sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
    sns.histplot(y_train, color='blue', alpha=0.5, bins=50)

    return plt.show()


def rmse(y, y_pred):
    """ This function computes the root mean square error between the predictions and the actuals 

        Args:
            y (numpy array): Array that includes the actuals
            y_pred (numpy array): Array that includes the predictions
    
        Return:
            rmse (float): float that represents the calculated root mean square error
    """
    error = y - y_pred
    se = error ** 2
    mse= se.mean()
    rmse = np.sqrt(mse)

    return rmse


def prepare_x_feature_age(df, base):
    """ This function select the columns to be filled by 0 in case of existent missing values
    
        Args:
            df (pandas.DataFrame): Dataframe that have the columns that will be subject to this filling 
            base (list): List that contains the columns that will be selected
        
        Return:
            x_train_base (numpy array): Array that contains df values for the columns selected in base 
                                        with no missing values
    """
    df=df.copy()
    df['age'] = 2017 - df.year
    features = base + ['age']
    df_num = df[features] 
    df_num = df_num.fillna(0) # Missing Values filled with 0
    x_train_base_feature = df_num.values

    return x_train_base_feature


def prepare_x_doors(df, base):
    """ This function select the columns to be filled by 0 in case of existent missing values
    
        Args:
            df (pandas.DataFrame): Dataframe that have the columns that will be subject to this filling 
            base (list): List that contains the columns that will be selected
        
        Return:
            x_train_base (numpy array): Array that contains df values for the columns selected in base 
                                        with no missing values
    """
    df=df.copy()
    features=base.copy()
    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2,3,4]:
        df[f'num_doors_{v}'] = (df.number_of_doors == v).astype('int')
        features.append(f'num_doors_{v}')

    df_num = df[features] 
    df_num = df_num.fillna(0) # Missing Values filled with 0
    x_train_base_feature = df_num.values

    return x_train_base_feature

def prepare_x_make(df, base):
    """ This function select the columns to be filled by 0 in case of existent missing values
    
        Args:
            df (pandas.DataFrame): Dataframe that have the columns that will be subject to this filling 
            base (list): List that contains the columns that will be selected
        
        Return:
            x_train_base (numpy array): Array that contains df values for the columns selected in base 
                                        with no missing values
    """
    df=df.copy()
    features=base.copy()
    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2,3,4]:
        df[f'num_doors_{v}'] = (df.number_of_doors == v).astype('int')
        features.append(f'num_doors_{v}')

    makes = list(df.make.value_counts().head().index)
    for v in makes:
        df[f'make_{v}'] = (df.make == v).astype('int')
        features.append(f'make_{v}')

    df_num = df[features] 
    df_num = df_num.fillna(0) # Missing Values filled with 0
    x_train_base_feature = df_num.values

    return x_train_base_feature


def prepare_x_categorical_variables(df, base):
    """ This function select the columns to be filled by 0 in case of existent missing values

        Args:
            df (pandas.DataFrame): Dataframe that have the columns that will be subject to this filling 
            base (list): List that contains the columns that will be selected
        
        Return:
            x_train_base (numpy array): Array that contains df values for the columns selected in base 
                                        with no missing values
    """
    df=df.copy()
    features=base.copy()
    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2,3,4]:
        df[f'num_doors_{v}'] = (df.number_of_doors == v).astype('int')
        features.append(f'num_doors_{v}')

    categorical_variables = ['make', 'engine_fuel_type', 'transmission_type', 'driven_wheels', 'market_category', 'vehicle_size', 'vehicle_style']
    categories = {}
    for c in categorical_variables:
        categories[c] = list (df[c].value_counts().head().index)

    for c, values in categories.items():
        for v in values:
            df[f'{c}_{v}'] = (df[c] == v).astype('int')
            features.append(f'{c}_{v}')

        

    df_num = df[features] 
    df_num = df_num.fillna(0) # Missing Values filled with 0
    x_train_base_feature = df_num.values

    return x_train_base_feature

def linear_regression_model_reg (x_train, y_train, r):
    """ This function trains the model using a regularization parameter to deal with correlated explanatory variables
            
            Args: 
                x_train (pandas.DataFrame): Dataframe that includes the explanatory variables for the train set
                y_train (numpy array): Array that includes the objective variable for the train set
                r (float): Regularization parameter, that will be used in the diagonal
                
            Return:
                w0 (float): constant obtained by training the linear regression model
                w (numpy array): array that contains the linear regression coefficients
    """    
    ones = np.ones(x_train.shape[0])
    x_train = np.column_stack([ones, x_train])

    XTX = x_train.T.dot(x_train)
    XTX =  XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(x_train.T).dot(y_train)

    w0 = w_full[0]
    w = w_full[1:]

    return w0, w  

def parse_arguments():
    """This function parses the argument(s) of this model

        Args:
            file_name: name of the command line field to insert on the runtime
            test_s: name of the command line field to insert on the runtime
            val_s: name of the command line field to insert on the runtime

        Return:
            args: Stores the extracted data from the parser run
    """

    parser = argparse.ArgumentParser(description="Process all the arguments for this model")
    parser.add_argument("file_name", help="The csv file name")
    parser.add_argument("test_s", help="The size of the test set, in percentage",type=float)
    parser.add_argument("val_s", help="The size of the validation set, in percentage",type=float)
    args = parser.parse_args()

    return args


def main():
    """This is the main function of this Linear Model Regression Implementation model"""
    args = parse_arguments()
    
    data, string_col = data_preparation(args.file_name)
    
    plt = exploratory_data_analysis (data, string_col)

    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(data, args.test_s, args.val_s)

    base=['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

    print('Columns Used: ', base)

    x_train_base = prepare_x(x_train, base)

    w0, w = linear_regression_model (x_train_base, y_train)

    y_pred_train = w0 + x_train_base.dot(w)

    plt = results_comparison_plot  (y_pred_train, y_train)

    print ('rmse y_pred_train : ', rmse(y_train, y_pred_train))

    x_val_base = prepare_x(x_val, base)

    y_pred_val = w0 + x_val_base.dot(w)   
    
    print ('rmse y_pred_val : ', rmse(y_val, y_pred_val))

    
    # Considering the 'age' as a feature
    x_train_base_feature = prepare_x_feature_age(x_train, base)
    w0, w = linear_regression_model (x_train_base_feature, y_train)
    x_val_base = prepare_x_feature_age(x_val, base)
    y_pred_val = w0 + x_val_base.dot(w)   
    print ('rmse y_pred_val_age : ', rmse(y_val, y_pred_val))
    plt = results_comparison_plot  (y_pred_val, y_val)

    # Considering the 'num_doors' as a feature (categorical)
    x_train_base_feature = prepare_x_doors (x_train, base)
    w0, w = linear_regression_model (x_train_base_feature, y_train)
    x_val_base = prepare_x_doors(x_val, base)
    y_pred_val = w0 + x_val_base.dot(w)   
    print ('rmse y_pred_val_doors : ', rmse(y_val, y_pred_val))

    # Considering the 'make' as a feature (categorical)
    x_train_base_feature = prepare_x_make (x_train, base)
    w0, w = linear_regression_model (x_train_base_feature, y_train)
    x_val_base = prepare_x_make(x_val, base)
    y_pred_val = w0 + x_val_base.dot(w)   
    print ('rmse y_pred_val_doors : ', rmse(y_val, y_pred_val))

    # Considering all categorical variables (top5) as a feature (categorical)
    x_train_base_feature = prepare_x_categorical_variables (x_train, base)
    w0, w = linear_regression_model (x_train_base_feature, y_train)
    x_val_base = prepare_x_categorical_variables (x_val, base)
    y_pred_val = w0 + x_val_base.dot(w)   
    print ('rmse y_pred_val_categorical_variables : ', rmse(y_val, y_pred_val))

    # Considering all categorical variables (top5) as a feature (categorical) and using regularization technique:
    for r in [0.0, 0.00001, 0.0001, 0.001, 0.1, 1, 10]:
        x_train_base_feature = prepare_x_categorical_variables (x_train, base)
        w0, w = linear_regression_model_reg (x_train_base_feature, y_train, r)
        x_val_base = prepare_x_categorical_variables (x_val, base)
        y_pred_val = w0 + x_val_base.dot(w)   
        print (f'rmse y_pred_val_categorical_var_reg_{r} : ', rmse(y_val, y_pred_val))
    
    # Using the model
    x_full_train = pd.concat([x_train, x_val])
    x_full_train = x_full_train.reset_index(drop=True)

    y_full_train = np.concatenate([y_train, y_val])

    r = 0.001
    x_train_base_feature = prepare_x_categorical_variables (x_full_train, base)
    w0, w = linear_regression_model_reg (x_train_base_feature, y_full_train, r)
    x_test = prepare_x_categorical_variables (x_test, base)
    y_pred_test = w0 + x_test.dot(w)   
    print (f'rmse y_pred_test_categorical_var_reg_{r} : ', rmse(y_test, y_pred_test))
    

if __name__ == '__main__':
    main()
