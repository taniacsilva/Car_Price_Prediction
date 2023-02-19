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
    df_col = pd.DataFrame(data_columns, index = data_columns)
    for col in data_columns:
        #for col in range(len(data_columns)):
            #print(data_columns[col])
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


def split_train_val_test(data):
    n = len(data)
    n_val = int(n*0.2)
    n_test = int(n*0.2)
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


def parse_arguments():
    """This function parses the argument(s) of this model

        Args:
            file_name: name of the command line field to insert on the runtime

        Return:
            args: Stores the extracted data from the parser run
    """

    parser = argparse.ArgumentParser(description='Process all the arguments for this model')
    parser.add_argument('file_name', help='The csv file name')

    args = parser.parse_args()

    return args


def main():
    """This is the main function of this Linear Model Regression Implementation model"""
    args = parse_arguments()
    
    data, string_col = data_preparation(args.file_name)
    
    plt = exploratory_data_analysis (data, string_col)
   
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(data)
    print(len(x_train))
    print(len(x_test))
    print(len(x_val))
if __name__ == '__main__':
    main()
