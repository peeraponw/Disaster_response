import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function to read csv files, messages and categories, and merge them into a dataframe.
    
    Args:
        messages_filepath (string): path to messages file
        categories_filepath (string): path to categories file
        
    Returns:
        df (dataframe): merged message and categories dataframe using 'id' as common column
    '''
    # read message file
    messages = pd.read_csv(messages_filepath)
    # read categories file
    categories = pd.read_csv(categories_filepath)
    # merge dataframes
    df = messages.set_index('id').join(categories.set_index('id'), on=['id'])
    return df

def clean_data(df):
    '''
    Function to clean the dataframe as follows:
        - split the 'categories' column into multiple columns witb their 
            corresponding meaning
        - rename the newly created columns
        - convert the information from string to int
        - drop the old 'categories' column
        - join new columns to df
    
    Args:
        df (dataframe): raw dataframe
    Returns:
        df (dataframe): cleaned dataframe
    '''
    # convert categories column into multiple categorized columns 
    categories = df['categories'].str.split(';', expand=True)
    # rename the newly created columns
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    # convert category number to just 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the initial column
    df = df.drop(columns=['categories'])
    # join new columns
    df = df.join(categories)
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()