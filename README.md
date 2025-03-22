# GlycemicRiskPred

In this repo, we predict glycemic risk from diabetes data as part of a Capstone Project. Continuous glucose monitor (CGM) and user demographic data were used for the prediction and were extracted from the Glooko database. We are not offering a license for use of this work. This work is covered by an NDA and distribution of it is subject to written approval from an authorized Glooko representative.


# Project motivation and value
The project goal was to develop a machine learning model to predict high glycemic risk patients. Healthcare providers (e.g., doctors) are seeking ways to improve the quality and cost-efficiency of diabetes care. Accurate prediction of high-risk patients allows for attention and intervention that may prevent future hospital visits and associated complications. This is a problem worth solving because high glycemic risk is associated with worsening health outcomes and increased healthcare costs. Intervention with high-risk patients could reduce patient harm from disease. Additionally, intervention could also reduce healthcare costs, allowing positive business value to be derived from the model.

# Data description
`data are hidden due to NDA`
Data was downloaded from a Glooko database stored in Snowflake and accessed via SQL queries. Two data files were used for this project. One file (users_info.csv) contains user information, including variables such as age, gender, equipment used, and the number of syncs. The second file (users_daily_stats.csv) contains daily CGM and insulin statistics, such as the number of readings, average value, data sync types, and the number of counts above or below certain clinically significant values (e.g., 54, 70, 180, and 250). Users were down-selected from the database such that the dataset was limited to users who are 18 or older, live in the United States, have type 1 diabetes, have consented to data sharing/analysis, and have recorded glucose values every day over the months of April through June 2021.

During the exploratory data analysis step of this project, several observations were made. These are summarized in the list below.

- The dataset contains 4130 users, with --% female and a mean age of -- years
- -- of users have insulin pumps and --% of users have single measurement blood glucose meters
- -- of users have recorded special activities, such as exercise, food, and medication events
- For all of the users and all of the days of glucose data, the mean and mode is -- and -- mg/dL
- For all of the users and all of the days of glucose data, the mean time in range (between 70 and 180 mg/dL) is --%, with --% mean time in hyperglycemia and -% mean time in hypoglycemia
- For all of the users and all of the days of glucose data, the mean lowest value is 75 mg/dL and the mean highest value is 267 mg/dL
- Glucose values and trends can fluctuate widely from hour to hour and day to day

The data file with daily CGM summary statistics contains multiple months of data for each user. For this reason, the data can be classified as time-series data. In contrast, the user demographic data is tabular, with one row of data per user. The time-series data was cast into a tabular format by pulling in the data from the last 14 days as different individual variables, based on an input date, and attaching these variables to a user. This resulted in over 130 variables. Feature reduction was accomplished by removing highly correlated (>0.8) features. The input date is the date that the prediction is made, where past data is used to predict future glycemic risk. Note that the glycemic risk index (GRI) is used as a corrolary for identifying high glycemic risk patients. The GRI is an arithmetic calculation based on the portion of CGM readings in certain high and low ranges.

# Model explanation

Several models were developed to predict an average GRI above 40 for a future 2-week period. These included random forest, logistic regression, and gradient-boosted decision trees. Additionally, two AutoML libraries were used to search for the highest-performing model, autoML and TPOT. A gradient-boosted decision tree (GBDT) model was selected as the final model architecture due to its superior performance. A GBDT model works by sequentially stacking decision tree models such that later models attempt to predict the error left over by the previous model. On test data, the final model typically performs with approximately 86% f1 score, accuracy, precision, and recall.

The SHapeley Additive Parameters (SHAP) algorithm was used to assist in explaining the importance of different features when the model makes predictions. Several outputs from the SHAP algorithm are shown below. The most important feature for prediction is the average GRI over the two week period prior to the prediction date.

- SHAP summary plot

![SHAP SUMMARY PLOT](/exported/shap_summary.png)

- Feature importance plot

![FEATURE IMPORTANCE](https://github.com/powerlock/DiabetesPred/blob/main/exported/feature_importance.png)


- Example decision tree plot

![TREE PLOT](https://github.com/powerlock/DiabetesPred/blob/main/exported/tree_plot.png)


# Pipeline description
## Flow of the pipeline
Data clean ---> data extraction with a mathematical model ---> data export as X, y -----> model training ----> prediction -----> performance evaluation (metrics) and explanation
## Functions explanation. In this project, we create our own pipeline with several functions. Below are details of these functions:
1. data_clean.load_data() -- Raw data clean (remove high cor. feature, encoding data, remove NAN data, statistically rolling data), calculation of GRI with modeled equation.
    default data: data1='users_info',data2='users_daily_stats'
2. data_clean.create_data_files() -- concat two different datasets into one, clean data, label data with date.
    dateChoice: str. e.g. '2021-04-14'
3. model.extractdata() -- extract data to return X, y. This function applies to both training and test data.
4. model.datafeed() -- feed X, y data to obtain training and test dataset.
5. model.train() -- train the data, use the joblib to save the trained model parameters, and finally return the model
6. model.predict() -- predict the outcomes with any test data, and return prediction list.
7. model.metrics() -- return a series of metrics values, including accuracy, precision, recall,  f1 score, and confusion matrix.
8. model.convert() -- convert the prediction results into a dictionary with a string property, so the FastApi can call and use.

9. Examples using model.py on the local machine. The following code can be copied into the model.py end of the file. Then run the command `python model.py` to execute the python file.
```
if __name__ == "__main__":
    dateChoice = '2021-04-14'
    test_date = '2021-06-07'
    users_ds_df, users_info_df = load_data()
    X_train, X_test, y_train, y_test = datafeed(dateChoice, users_ds_df, users_info_df)
    model = train(X_train, y_train)
    X_test,y_test = extractdata(test_date, users_ds_df, users_info_df)
    
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    prediction_list = predict(X_test).model
    metrics = metrics(y_test, prediction_list)
    print(metrics)
    output = convert(prediction_list)
    print(output)
    #Use shap to explain visulizae importance rank
    model_file = Path(BASE_DIR).joinpath(f"{'XGBC'}.joblib")
    model = joblib.load(model_file)

    shap.summary_plot(gbm_shap_values, X_train)
    explainer = shap.TreeExplainer(model)
    shap_val = explainer.shap_values(X_test)
    shap.summary_plot(shap_val, X_test)
```
# How to deploy it on your local machine and run it with FastAPI?
1. Install requirements 
`pip install -r requirements.txt`
2. run the uvicorn command:
`uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8000`
3. Navigate to localhost in your browser.
`http://localhost:8000/docs`
4. Click Try it out on the `/predict` function and copy the following text into the response body.
```
request URL:
http://localhost:8000/predict
```
```
Request Body:
{
  "test_date": "2021-06-07"
}

```
```
<response header: >

 {content-length: 71708 
 content-type: application/json 
 date: Sat,13 Aug 2022 22:38:08 GMT 
 server: uvicorn }

```

<Response body:>

<[https://github.com/j-philtron/GlycemicRiskPred/blob/main/exported/response_example.json]>




# How to deploy it on EC2 with docker?
## File inspection:
  Inspect each of the files and make sure you understand what they are doing.
   - `model.py` contains the code to perform GRI value greater than 40 predictions.
   - `main.py` contains the code to handle the request and return the result.
   - `requirements.txt` contains the dependencies of your ML App.
   - `Dockerfile` contains all the commands to assemble a docker image.
   - `events/event.json` contains the event that triggers the GRI prediction.
   - `template.yaml` defines the application's AWS resources and is required when using SAM CLI.
      The application uses several AWS resources, including Lambda functions and an API Gateway API. These resources are defined in the `template.yaml`. 

## Build and Test Locally via SAM CLI
1. To build and deploy your application for the first time, run the following in your shell. 

   ```bash
   GRI% sam build
    ```
   It can take a few minutes to complete. The SAM CLI builds a docker image from a Dockerfile and then installs dependencies defined in `requirements.txt` inside the docker image. The processed template file is saved in the `.aws-sam/build` folder. 

   <details>
   <summary>View the long build output from SAM CLI </summary>
2. Test a single function locally. 

   Invoke it directly with a test event. An event is a JSON document that represents the input that the function receives from the event source. One sample `event.json` is provided under `events` directory.

   Run functions locally and invoke them with the `sam local invoke` command.

   ```bash
   GRI% sam local invoke GRIFunction --event events/event.json
   ```

   <details>
   <summary> Click here to see a sample output.</summary>

## Deploy on AWS:
1. Generating user credentials
2.    - **Setting the Credentials** - The next step is set the credentials as environment variables. On your open terminal, run the following commands:

   ```bash
      export AWS_ACCESS_KEY_ID=your_access_key_id
      export AWS_SECRET_ACCESS_KEY=your_secret_access_key
   ```
   while replacing your_access_key_id and your_secret_access_key with the values you find in the csv file you downloaded earlier. If you're using our AWS resources, you will use the keys we provided you.
3. Deployment using SAM CLI

   The command will package and deploy your application to AWS, with a series of prompts:

   ```bash
   sam deploy --guided
   ```
4. Test your API directly by starting with clicking on "Let's try out".

![Try it out button](images/try_it_out.png)

and run a query. For example, change the provided request body to



