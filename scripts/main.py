from fastapi import FastAPI, Query, HTTPException,File, UploadFile
from pydantic import BaseModel
from model import predict, convert,extractdata, metrics
from typing import List
import uvicorn
from data_clean import *
app = FastAPI()

# load multiple datafiles, this function need to be fixed, so the following can use the uploaded data. Right now, it is loading from the repo.

@app.post("/upload")
def upload(files: List[UploadFile] = File(...)):
    for file in files:
        try:
            contents = file.file.read()
            with open(file.filename, 'wb') as f:
                f.write(contents)

        except Exception:
            return {"message": "There was an error uploading the file(s)"}
        finally:
            
            file.file.close()

    return {"message": f"Successfuly uploaded {[file.filename for file in files]}"}    


# pydantic models
class GlyIn(BaseModel):
    
    test_date: str
    def dataclean(test_date):
        users_ds_df, users_info_df = load_data()
        
        X_test,y_test = extractdata(test_date, users_ds_df, users_info_df)

        return X_test, y_test
    
# response model
class GlyOut(GlyIn):
    metrics: dict
    forecast: dict
    

@app.post("/predict", response_model=GlyOut, status_code=200)
def get_prediction(payload: GlyIn):
    test_date = payload.test_date
    X,y = GlyIn.dataclean(test_date)
    
    prediction_list = predict(X)
   
    if not prediction_list.any():
        raise HTTPException(status_code=400, detail="Model not found.")
    response_object = {
        "test_date":test_date,
        "metrics":metrics(y,prediction_list),
        "forecast": convert(prediction_list)}

    return response_object

#if __name__ == "__main__":
    #GlyIn.test_date = "2021-06-07"
    #dataclean(data)
    #r = get_prediction(GlyIn)
    
    #uvicorn.run(app, host="0.0.0.0", port=8000)
    #uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8000
# url = 'http://127.0.0.1:8000/upload'
# files = [('files', open('images/1.png', 'rb')), ('files', open('images/2.png', 'rb'))]
# resp = requests.post(url=url, files=files) 
# print(resp.json())