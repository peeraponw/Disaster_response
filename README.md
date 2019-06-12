# Disaster Response Pipeline Project
This project is a part of Udacity's Data Science Nanodegree. The concept of this project is to classify incoming texts during disaster. The classified results can lead to better understanding and better preparation of corresponding parties during the fatal situations.
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Summary of the project
This project gives a better understanding of the overall process in data science pipeline. It composes of three important parts:
1. **ETL Pipeline** - The (E)xtract, (T)ransform, (L)oad are common in data science workflow. Combining them into a pipeline makes the workflow much easier to handle the upcoming parts.
2. **ML Pipeline** - In machine learning process, there might be more than just putting data into training and making prediction. This project handles text data which needs multiple transformation processes before feeding into a prediction model. It includes `CountVectorizer`, `TfidfTransformer` and `RandomForestClassifier` as the final estimator. 
3. **Visualization** - This project uses Flask and Plotly to deploy the visualization into a web application.