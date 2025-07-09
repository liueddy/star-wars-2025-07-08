import pandas as pd
from processor import PProcessor
import joblib
import json
from pydantic import BaseModel


class Validator:
    """The following class serves as a validator
    to confirm whether unsafe user inputs are
    properly formatted and converts the input
    features into format which a given model
    can use."""
    def __init__(self,proc_pkl,model_pkl):
        self.pproc = PProcessor().load(proc_pkl)
        self.model = joblib.load(str(model_pkl))
    
    def _uvalidate(self,input:dict,bmodel:BaseModel=None) -> pd.DataFrame:
        return input

    def run(self, uinput) -> dict:
        v_df = self._uvalidate(uinput)
        t_df = self.pproc.transform(v_df)
        return {"prediction":self.model.predict(t_df)}

if __name__ == "__main__":
    # create mock model that can return 1
    class MockModel:
        def __init__(self,x):
            self.x=x
        def predict(self,uinput):
            return self.x
        
    mm = MockModel(1)
    joblib.dump(mm,"pkl/mockmodel.pkl")
    
    v = Validator("pkl/processor.pkl","pkl/mockmodel.pkl")
    print(v.model.predict(199))
