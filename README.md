# GlycemicRiskPred
FourthBrain Capstone Project to predict glycemic risk from diabetes data

CGM data and user demographics/etc. will be used to predict the Glycemic Risk. This project mainly focuses on Time in Range and the risk correlation.

We are not offering a license for use of this work. This work is covered by an NDA and distribution of it is subject to written approval from an authorized Glooko representative.
# Motivation of the project, technical and business values.
# Explanation of the data. 
- shap image: `/exported/SHAP_SUMMARY_PLOT.png`
- tree plot: `/exported/Tree_plot.png`
- feature importance `/exported/Feature_importance.png`
# Explanation of the model

# Explanation of the pipeline
## Flow of the pipeline
Data clean ---> data extraction with a mathmatical model ---> data export as X, y -----> model training ----> prediction -----> performance evaluation (metrics) and explanation
## Functions explanation. In this project, we create our own pipeline with several functions. Below are details of these functions:
1. data_clean.load_data() -- Raw data clean (remove high cor. feature, encoding data, remove NAN data, statistically rolling data), calculation of GRI with modeled equation.
    default data: data1='users_info',data2='users_daily_stats'
2. data_clean.create_data_files() -- concat two different dataset into one, clean data, label data with date.
    dateChoice: str. e.g. '2021-04-14'
3. model.extractdata() -- extract data to return X, y. This function applies to both training and test data.
4. model.datafeed() -- feed X, y data to obtain training and test dataset.
5. model.train() -- train the data, use the joblib to save the trained model parameters, and finally return the model
6. model.predict() -- predict the outcomes with any test data, and return prediction list.
7. model.metrics() -- return a series of metrics values, including accuracy, precision, recall,  f1 score, and confusion matrix.
8. model.convert() -- convert the prediciton results into a dictionary with string property, so the FastApi can call and use.

9. Examples using model.py on the local machine. The following code can be copied into the model.py end of the file. Then run the command `python model.py` to execute the python file.
<!-- #if __name__ == "__main__":
    #dateChoice = '2021-04-14'
    #test_date = '2021-06-07'
    #users_ds_df, users_info_df = load_data()
    #X_train, X_test, y_train, y_test = datafeed(dateChoice, users_ds_df, users_info_df)
    #model = train(X_train, y_train)
    #X_test,y_test = extractdata(test_date, users_ds_df, users_info_df)
    
    #print(X_train.shape, y_train.shape)
    #print(X_test.shape, y_test.shape)

    #prediction_list = predict(X_test).model
    #metrics = metrics(y_test, prediction_list)
    #print(metrics)
    #output = convert(prediction_list)
    #print(output)
    ##Use shap to explain visulizae importance rank
    #model_file = Path(BASE_DIR).joinpath(f"{'XGBC'}.joblib")
    #model = joblib.load(model_file)

    #shap.summary_plot(gbm_shap_values, X_train)
    #explainer = shap.TreeExplainer(model)
    #shap_val = explainer.shap_values(X_test)
    #shap.summary_plot(shap_val, X_test) -->

# How to deploy it on your local machine and run it with FastAPI?
1. Install requirements 
`pip install -r requirements.txt`
2. run the uvicorn command:
`uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8000`
3. Navigate to localhost in your browser.
`http://localhost:8000/docs`
4. test the code with respective endpoint.
```
{
  "test_date": "2021-06-07"
}
```
```
response header
 {content-length: 71708 
 content-type: application/json 
 date: Sat,13 Aug 2022 22:38:08 GMT 
 server: uvicorn }
```

response example: `../exported/response_example.json`

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



