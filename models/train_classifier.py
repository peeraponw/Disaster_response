import sys
import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import pickle
from joblib import dump, load
import warnings; warnings.simplefilter('ignore')


def load_data(database_filepath):
    '''
    Function to load the cleaned data from ETL pipeline
    
    Args:
        database_filepath (string): database name
    Returns:
        X (dataframe): messages as features of the dataset
        Y (dataframe): numbers to specify the category of the dataset
        category_names (list of strings): column names
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster', con=engine)
    df = df.dropna(how='any')
    X = df['message']
    Y = df.drop(columns=['message', 'original', 'genre'])
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenize the input text by lowering and lemmatization
    
    Args:
        text (string): unprocessed text
    Returns:
        text (string): text which are lowered and lemmatized
    '''
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text.lower())
    return text


def build_model():
    '''
    Build model by creating a pipeline first. Then apply GridSearchCV concept
    
    Args:
        None
    Returns:
        model (estimator object): an estimator that is able to make a prediction
    '''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier())),
        ])
    parameters = {
    'clf__estimator__n_estimators': [2,20,50],
    'clf__estimator__max_depth': [5,10,30],
    'clf__estimator__min_samples_split': [5,10,20]    
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model by comparing the prediction on test data and the test results
    
    Args:
        model (estimator): estimator to make a prediction
        X_test (dataframe): features of test data
        Y_test (dataframe): results of test data
        category_names (list of strings): column names
        
    Returns:
        None
    '''
    y_pred = model.predict(X_test)
    for idx in range(y_pred.shape[1]):
        report = classification_report(Y_test.as_matrix()[:,idx], y_pred[:,idx],
                                       output_dict=True)
        print('Category', category_names[idx])
        for key in report.keys():
            print('Label {}:\t Precision={:.3}, Recall={:.3}, F1-score={:.3}'\
                  ''.format(key, report[key]['precision'], report[key]['recall'],
                            report[key]['f1-score']))

def save_model(model, model_filepath):
    '''
    Save the model into pickle file
    
    Args:
        model (estimator): trained model
        model_filepath (string): path to save the model
    Returns:
        None
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()