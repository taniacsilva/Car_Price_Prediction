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

    return data

def exploratory_data_analysis (data):
    """This function makes some exploratory data analysis

        Args:
            data (pandas.DataFrame): The pandas Dataframe

        Return:
            
    """
    for col in data.columns:
        print(data[col].unique()[:5])
        print(data[col].nunique())

    plot=sns.histplot(data.msrp [data.msrp < 100000], bins=50)

    return plt.show()


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
    0
    data = data_preparation(args.file_name)
    
    plt = exploratory_data_analysis (data)
   
    for col in data.columns:
        print(data[col].unique()[:5])
        print(data[col].nunique())


if __name__ == '__main__':
    main()
