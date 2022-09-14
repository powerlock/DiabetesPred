"""
This file loads and cleans the data. The clean X and y are saved to csv files.
"""

import pandas as pd
# import matplotlib.pyplot as plt
# plt.rcParams.update({'axes.labelsize':16, 'axes.titlesize':20})
# import seaborn as sns
# sns.set_style('whitegrid')
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime, timedelta

# In the load data funcion, the default data filename is users_info.csv, and users_daily_stats.csv
def load_data(data1='users_info',data2='users_daily_stats'):
    ##
    # Load the data
    users_info_df = pd.read_csv('../data/' +data1+'.csv')
    users_ds_df = pd.read_csv('../data/'+data2+'.csv')

    ##
    # Generate additional variables from user info file
    users_info_df['APP_LENGTH'] = users_info_df['CURRENT_AGE'] - \
                                    users_info_df['AGE_WHEN_REGISTERED']
    users_info_df['HAS_OTHER_COUNT'] = users_info_df['EVENT_COUNT'].notnull()
    # One hot encode the APPLICATION_NAME variable
    users_info_df = pd.concat(
        [users_info_df,
        users_info_df['APPLICATION_NAME'].str.get_dummies(sep=",")
        ], axis=1)
    # One hot encode the OS variable
    users_info_df = pd.concat(
        [users_info_df,
        users_info_df['OS'].str.get_dummies(sep=",")
        ], axis=1)
    users_info_var_select = ['GENDER','CURRENT_AGE','HAS_PUMP','HAS_METER',
                            'SYNC_COUNT','APP_LENGTH','HAS_OTHER_COUNT',
                            'c4c','kiosk','logbook','patient_uploader',
                            'server','android','browser','ios']

    ##
    # Adjust and generate variables from daily stats file
    # Change the units of the reading counts variables from counts to % of time
    read_cnt_vars = ['ABOVE_180','ABOVE_250','ABOVE_400',
                    'BELOW_50','BELOW_54','BELOW_60','BELOW_70']
    for i,var in enumerate(read_cnt_vars):
        #users_ds_df[var] = users_ds_df[var] / users_ds_df['TIME_CGM_ACTIVE']
        users_ds_df[var] = round(100 * users_ds_df[var] /
                                users_ds_df['READING_COUNT'],2)
    # Create variable for Glycemic Risk Index (GRI)
    # GRI = (3.0 × VLow) + (2.4 × Low) + (1.6 × VHigh) + (0.8 × High)
    # or alternatively: GRI = (3.0 × HypoComp) + (1.6 × HyperComp)
    # where Hypoglycemia Component = VLow + (0.8 × Low)
    #   and Hyperglycemia Component = VHigh + (0.5 × High)
    # and where VLow = # time < 54, Low = # time from 54 to < 70
    #          VHigh = # time > 250, High = # time from 250 to > 180
    GRI_temp = pd.DataFrame()
    GRI_temp['VLow'] = users_ds_df['BELOW_54']
    GRI_temp['Low'] = users_ds_df['BELOW_70'] - users_ds_df['BELOW_54']
    GRI_temp['VHigh'] = users_ds_df['ABOVE_250']
    GRI_temp['High'] = users_ds_df['ABOVE_180'] - users_ds_df['ABOVE_250']
    # Create HypoComp, HyperComp, and GRI for each user and day in the daily stats dataframe
    users_ds_df['GRI_HYPO'] = GRI_temp['VLow'] + 0.8 * GRI_temp['Low']
    users_ds_df['GRI_HYPER'] = GRI_temp['VHigh'] + 0.5 * GRI_temp['High']
    users_ds_df['GRI'] = 3 * users_ds_df['GRI_HYPO'] + 1.6 * users_ds_df['GRI_HYPER']
    # sort values by USER_ID and DATE
    users_ds_df.sort_values(by=['USER_ID','DATE'], inplace=True)
    # Also put the date variable into datetime format
    users_ds_df['DATE'] = users_ds_df['DATE'].apply(lambda x: datetime.strptime(x[2:], '%y-%m-%d'))

    ##
    # Create the prediction and related variables
    users_ds_df['GRI_PAST_2WK'] = users_ds_df['GRI'].rolling(
                                    14).mean()
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=7)
    users_ds_df['GRI_FUTURE_1WK'] = users_ds_df['GRI'].rolling(
                        window=indexer, min_periods=1).mean()
    indexer2 = pd.api.indexers.FixedForwardWindowIndexer(window_size=14)
    users_ds_df['GRI_FUTURE_2WK'] = users_ds_df['GRI'].rolling(
                        window=indexer2, min_periods=1).mean()

    return users_ds_df, users_info_df
##
# Create a function that will write out X and y into files, given a date
def create_data_files(dateChoice, users_ds_df, users_info_df):
    # We select variables for a certain day, and then reset the index to the user_id
    # Note: the index is the user_id because the data was ordered by user_id (so we don't need to set it explicitly)
    X = users_ds_df[['TOTAL_INSULIN', 'TOTAL_BASAL', 'TOTAL_BOLUS',
                    'HAS_REMOTE_SMBG_DATA', 'HAS_REMOTE_CGM_DATA',
                    'HAS_REMOTE_INSULIN_DATA', 'HAS_IN_CLINIC_SYNC',
                    'GRI_FUTURE_2WK'
                    ]][users_ds_df['DATE'] == dateChoice] \
                    .reset_index().drop(columns='index')
    # Define a variable of user info data with only the columns we want
    users_info_temp = users_info_df.drop(columns=[
        "DIABETES_TYPE","DATE_OF_BIRTH","AGE_WHEN_REGISTERED","APPLICATION_NAME","OS",
        "EVENT_COUNT","FOOD_COUNT","EXERCISE_COUNT","MEDICATION_COUNT","Unnamed: 0"
    ]).set_index("ID")
    # Now, join that dataframe in
    X = X.join(users_info_temp)
    # Convert daily variables into a structured (tabular) data format
    # These variables we will make into multiple variables for data from the past 2 weeks:
    var_proc_2wk = ['TIME_CGM_ACTIVE','AVERAGE_VALUE','ABOVE_250','BELOW_54',
                   'LOWEST_VALUE','HIGHEST_VALUE','GRI']
    # We make 14 variables containing all the users, for the last 14 days
    dates = []
    for i in range(14):
        dates.append(datetime.strptime(dateChoice[2:], '%y-%m-%d') - timedelta(days=i))
    for j in range(len(dates)):
        globals()[f"data_{j}"] = users_ds_df[var_proc_2wk] \
            [users_ds_df['DATE'] == dates[j]].reset_index()
    # We will create 14 values for each variable, one each for each day in the past 2 weeks
    for var in var_proc_2wk:
        for j in range(len(dates)):
            X[f"{var}_{j}"] = globals()[f"data_{j}"][var]
    # That results in a fragmented dataframe, so let's copy it over to fix that
    X = X.copy()
    # Now we create an additional value for each variable
    # which is the average over the last 2 week period
    for var in var_proc_2wk:
        # create a list of column names:
        names = []
        for j in range(len(dates)):
            names.append(f"{var}_{j}")
        # Now take the average of these values and save it as a new variable
        X[f"{var}_avg2wk"] = X[names].mean(axis=1)
    # We remove insulin variables for now, since some are NaNs
    has_insulin = (X['TOTAL_INSULIN'].isna()).astype(int)
    X.drop(columns=['TOTAL_INSULIN','TOTAL_BASAL','TOTAL_BOLUS'],
          inplace=True)
    X['HAS_INSULIN_DATA'] = has_insulin
    # Create the prediction variable
    # We will predict whether the average GRI over the next 2 weeks will be > 40
    # Values above 40 are classified as Zone C
    y = (X.GRI_FUTURE_2WK > 40).astype(int)
    # We save the future data in a separate variable for later use
    future_data = X['GRI_FUTURE_2WK']
    # And then remove the future data from X
    X = X.drop(columns='GRI_FUTURE_2WK')
    # There are a couple NaNs in 'HAS_REMOTE_CGM_DATA'
    # We will fill the NaNs with False values
    X.fillna(False, inplace=True)
    # And change the booleans to 0's and 1's
    var_convert = ['HAS_REMOTE_SMBG_DATA', 'HAS_REMOTE_CGM_DATA',
                   'HAS_REMOTE_INSULIN_DATA', 'HAS_IN_CLINIC_SYNC',
                   'HAS_CGM','HAS_PUMP','HAS_METER','HAS_OTHER_COUNT']
    for var in var_convert:
        X[var] = (X[var]).astype(int)
    # And change Gender so that Male = 1 and Female = 0
    X["GENDER"].replace(["male","female"], [1,0], inplace=True)

    ##
    # Save the training data as a csv file
    X.to_csv(f"../data/X_{dateChoice}.csv")
    y.to_csv(f"../data/y_{dateChoice}.csv")

    return X, y

""" ##
# Write out files for the dates of interest
dateChoice = '2021-04-14'
create_data_files(dateChoice, users_ds_df, users_info_df)
dateChoice = '2021-05-05'
create_data_files(dateChoice, users_ds_df, users_info_df)
 """
##
# Define some useful functions for looking at the results
# Function to print the accuracy and classification report
def print_acc_and_CR(y_test, model_predictions):
    # Calculate and display the model accuracy
    print('Accuracy: {:.2f}%'.format(accuracy_score(y_test, model_predictions) * 100))
    # Calculate and display the classification report for the model
    print('Classification report: \n', classification_report(y_test, model_predictions))
# Function to display the confusion matrix
def display_confusion_matrix(y_test, model_predictions):
    # Calculate and display the confusion matrix
    model_confusionMatrix = confusion_matrix(y_test, model_predictions)
    # Create variables that will be displayed as text on the plot
    strings2 = np.asarray([['True Negatives \n', 'False Positives \n'], ['False Negatives \n', 'True Positives \n']])
    labels2 = (np.asarray(["{0} {1:g}".format(string, value)
                          for string, value in zip(strings2.flatten(),model_confusionMatrix.flatten())])
             ).reshape(2, 2)
    # Use a heat map plot to display the results
    sns.heatmap(model_confusionMatrix, annot=labels2, fmt='', vmin=0, annot_kws={"fontsize":17})
    plt.xlabel('Predicted value');
    plt.ylabel('Actual value');
