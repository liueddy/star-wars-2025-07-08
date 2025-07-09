import numpy as np
import pandas as pd
from pprocessor import PProcessor
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
        self.pproc = PProcessor()
        self.pproc.load(proc_pkl)
        self.model = joblib.load(str(model_pkl))
        self.bmodel = bmodel
        # self.y_col = str(y_col)
    
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
        # print("\nv_df:\n",v_df)
        t_df = self.pproc.transform(v_df)
        # print("\nt_df:\n",t_df)
        return self.model.predict(t_df).item()
    
    def getFeatures(self) -> dict:
        if hasattr(self.model,"feature_importances_"):
            return dict(zip(
                self.model.feature_names_in_,
                self.model.feature_importances_
            ))
        elif hasattr(self.model,"coef_"):
            """Explains by showing weights of model"""
            def add_expl(s,w):
                expl_dict[s] = w

            # save feature names
            X = self.model.feature_names_in_

            # add to dict:
            expl_dict = {
                "Intercept":self.model.intercept_.item()
            }
            [add_expl(X[i],self.model.coef_.flatten()[i]) for i in range(len(X))]
            return expl_dict


if __name__ == "__main__":
    import swdantic    
    v = Validator("pkl/ptransformer.pkl",
                  "pkl/model.pkl",
                  swdantic.SWDantic,
                #   y_col="empire_or_resistance_resistance"
                  )
    print(v.run({
        "unit_type":"resistance_soldier",
        "homeworld":"Stewjon",
    }))
