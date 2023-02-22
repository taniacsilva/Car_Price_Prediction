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
    ones = np.ones(x_train.shape[0])
    x_train = np.column_stack([ones, x_train])

    XTX = x_train.T.dot(x_train)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(x_train.T).dot(y_train)

    w0 = w_full[0]
    w = w_full[1:]

    return w0, w     


def base_plot (y_pred, y_train):
    sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
    sns.histplot(y_train, color='blue', alpha=0.5, bins=50)

    return plt.show()


def parse_arguments():
    """This function parses the argument(s) of this model

        Args:
            file_name: name of the command line field to insert on the runtime

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

    x_train_base = x_train[base].fillna(0).values # Missing Values filled with 0

    w0, w = linear_regression_model (x_train_base, y_train)

    y_pred = w0 + x_train_base.dot(w)

    plt = base_plot (y_pred, y_train)
    
    print(y_pred)
if __name__ == '__main__':
    main()
