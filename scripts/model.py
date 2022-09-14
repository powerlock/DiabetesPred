import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
import joblib
from xgboost import XGBClassifier
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score as accuracy,
    recall_score as recall,
    precision_score as precision,
    f1_score
)
from data_clean import *
import shap


# Load pre-cleaned data
""" train_date = '2021-04-14'
test_date = '2021-05-05'
X_train = pd.read_csv(f"../data/X_{train_date}.csv")
y_train = pd.read_csv(f"../data/y_{train_date}.csv")
X_test = pd.read_csv(f"../data/X_{test_date}.csv")
y_test = pd.read_csv(f"../data/y_{test_date}.csv")

# Do a clean up for the data to ensure 1 column
y_train = y_train.iloc[:,1:]

y_test = y_test.iloc[:,1:] """
""""
train the model,
predict based on the model and return the metrics.
"""
#set basline folder path for model
BASE_DIR = Path(__file__).resolve(strict=True).parent

def extractdata(dateChoice, users_ds_df, users_info_df):
    X, y = create_data_files(dateChoice, users_ds_df, users_info_df)
    return X,y
# create a function to get X, y and do the traing, test split
def datafeed(dateChoice, users_ds_df, users_info_df):
    X, y = create_data_files(dateChoice, users_ds_df, users_info_df)
    #print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    #print(X_train.shape, y_train.shape, X_test.shape,y_test.shape)
    return X_train, X_test, y_train, y_test 
# feed the cleaned data from upstream
def train(X_train, y_train):

    model = XGBClassifier(random_state=42,learning_rate=0.1, max_depth=7, min_child_weight=18, n_estimators=100, n_jobs=1, subsample=0.4, verbosity=0)

    model.fit(X_train, y_train)

    joblib.dump(model, Path(BASE_DIR).joinpath(f"{'XGBC'}.joblib"))
    return model
def predict(X_test):
    model_file = Path(BASE_DIR).joinpath(f"{'XGBC'}.joblib")
    if not model_file.exists():
        print("model doesn't exist")
        return False

    model = joblib.load(model_file)

    prediction = model.predict(X_test)

    return prediction

def metrics(y_test,auto_pred):

    output = {}
    output["accuracy"] =  str(round(accuracy(y_test, auto_pred),4))
    output["precision"] = str(round(precision(y_test, auto_pred,average = 'macro'),4))
    output["recall"] = str(round(recall(y_test, auto_pred,average = 'macro'),4))
    output["f1 score"] = str(round(f1_score(y_test, auto_pred,average = 'macro'),4))
    output["confusion matrix"] = str(confusion_matrix(y_test, auto_pred))

    return output

# convert prediction list to a dictionary with string
def convert(prediction_list):

    output = {}
    

    for d in range(len(prediction_list)):
        val = prediction_list[d]
        output['usr'+str(d)] = str(val.astype(bool))

    return output
#Use the following and pre-cleaned data to train the model, model parameters are saved in XGBC.joblib file.
#if __name__ == "__main__":
    
    #dateChoice = '2021-04-14'
    #test_date = '2021-06-07'
    #users_ds_df, users_info_df = load_data()
    #X_train, X_test, y_train, y_test = datafeed(dateChoice, users_ds_df, users_info_df)
    #model = train(X_train, y_train)
    #X_test,y_test = extractdata(test_date, users_ds_df, users_info_df)
    
    #print(X_test.iloc[1,:])
    #print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    #prediction_list = predict(X_test).model
    # print(X_test.columns)
    # print(y_test.head(3))
    #output = convert(prediction_list)
    #print(output)
    # Use shap to explain visulizae importance rank
    #model_file = Path(BASE_DIR).joinpath(f"{'XGBC'}.joblib")
    #model = joblib.load(model_file)
    #gbm_shap_values = shap.KernelExplainer(model.predict,X_train)

    #shap.summary_plot(gbm_shap_values, X_train)
    # XN = pd.DataFrame(X_train)


    #explainer = shap.TreeExplainer(model)
    #shap_val = explainer.shap_values(X_test)
    # #shap.summary_plot(shap_val, XN.values,plot_type='bar')
    #shap.plots.beeswarm(shap_val, max_display=20)
    #shap.summary_plot(shap_val, X_test)