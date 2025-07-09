import numpy as np
import pandas as pd
from server.pprocessor import PProcessor
import joblib
import warnings
from pydantic import BaseModel


class Validator:
    """The following class serves as a validator to confirm whether unsafe user inputs are
    properly formatted and converts the input features into format which a given model
    can use.
    
    This class must take in a valid ptransformer.pkl file and a valid model.pkl file.
    A pydantic BaseModel is optional but highly recommended.
    """
    def __init__(self,proc_pkl,model_pkl,bmodel:BaseModel=None) -> None:
        self.pproc = PProcessor().load(proc_pkl)
        self.model = joblib.load(str(model_pkl))
        self.bmodel = bmodel
    
    def _uvalidate(self,uinput:dict) -> pd.DataFrame:
        """validates the given user input based on the given BaseModel"""
        if self.bmodel:
            # given base model to validate input
            midata = self.bmodel(**uinput)
            return pd.DataFrame.from_dict(midata.model_dump(),orient='index').transpose().convert_dtypes()
        else:
            # ignore data validation and try to jump to predictions
            # this is not recommended but the model will always throw 
            # an error given bad data.
            warnings.warn("this is NOT recommended, to suppress warning add a valid BaseModel")
            return pd.DataFrame.from_dict(uinput,orient='index').transpose().convert_dtypes()

    def run(self, uinput) -> np.array:
        v_df = self._uvalidate(uinput)
        t_df = self.pproc.transform(v_df)
        return self.model.predict(t_df)

if __name__ == "__main__":
    # create mock model that can return 1
    # class MockModel:
    #     def __init__(self,x):
    #         self.x=x
    #     def predict(self,uinput):
    #         return self.x
        
    # mm = MockModel(1)
    # joblib.dump(mm,"pkl/mockmodel.pkl")
    
    v = Validator("pkl/ptransformer.pkl","pkl/mockmodel.pkl")
    # print(v.model.predict(199))
