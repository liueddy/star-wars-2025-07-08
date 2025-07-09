import sklearn
import sklearn.preprocessing
import numpy as np
import pandas as pd
import joblib

class PProcessor():
    """The following class is a processor that handles data preparation both for training
    AND for non-training data.
    - The Processor class handles both numeric and categorical data.
    - The Processor class can also save/load an inner PTransformer class which can save
      the requisite transformer to a pkl file.
    
    The PTransformer class takes in exactly two TRAINED preprocessors and calls transform 
    to transform a given new value from an unprocessed state to a fully processed state. 

    Methods:
    - fit
    - transform
    - fit_transform
    - save(filepath:str)
    - load(filepath:str)
    - load_transform(filepath:str,df:pd.DataFrame)
    """
    def __init__(self,):
        # change this is you need to...
        self.pnum = sklearn.preprocessing.StandardScaler().set_output(transform="pandas")
        self.pcat = sklearn.preprocessing.OneHotEncoder(drop='first',sparse_output=False).set_output(transform="pandas")
        self.hasFitted = False
        self.pt = None
    
    class PTransformer():
        """This is a Processor().Transformer() which can tranform numeric or categorical
        data.  This class has one function, transform(data:pd.DataFrame) which will, 
        when called transform the dataframe based on the pre-fit sklearn preprocessors.

        fields:
        - pnum - a pre-trained sklearn numeric preprocessor
        - pcat - a pre-trained sklearn categoric preprocessor
        methods:
        - transform(data:pd.DataFrame)

        Note: This class does NOT allow re-fitting.
        """
        def __init__(self,pnum,pcat):
            """takes in a TRAINED preprocessor"""
            self.pnum = pnum
            self.pcat = pcat

        def transform(self,df:pd.DataFrame) -> pd.DataFrame:
            """Transforms DataFrame based on pre-fit preprocessors"""
            x1 = self.pnum.transform(df.select_dtypes(include="number"))
            x2 = self.pcat.transform(df.select_dtypes(exclude="number")).astype(int)
            return pd.merge(left=x1,right=x2,how="outer",on=x1.index).drop("key_0",axis=1)
        
        def __repr__(self):
            """Overwrite repr"""
            return f"PTransformer(pnum={self.pnum}, pcat={self.pcat})" 

    def _fit_num(self,df:pd.DataFrame):
        """returns a trained processor object for numeric data"""
        return self.pnum.fit(df.select_dtypes(include="number"))
    
    def _fit_cat(self,df:pd.DataFrame):
        """returns a trained processor object for categoric data"""
        return self.pcat.fit(df.select_dtypes(exclude="number"))
    
    def fit(self,df:pd.DataFrame) -> bool:
        """Tries fitting, returns whether successfully fit or not"""
        try:
            self._fit_num(df)
            self._fit_cat(df)
            self.pt = self.PTransformer(self.pnum,
                                        self.pcat)
            self.hasFitted = True
        finally:
            return self.hasFitted
        
    def transform(self,df:pd.DataFrame) -> pd.DataFrame:
        """uses fit to transform DataFrame"""
        assert self.hasFitted
        return self.pt.transform(df)
    
    def fit_transform(self,df:pd.DataFrame):
        """calls fit then transform on same data"""
        self.fit(df)
        return self.transform(df)

    def save(self,filepath="pkl/ptransformer.pkl"):
        """save to pkl file"""
        assert self.hasFitted
        joblib.dump(self.pt,str(filepath))
    
    def load(self,filepath):
        """load from pkl file
        @throws exception when obj not a PTransformer
        """
        obj = joblib.load(str(filepath))
        if "PTransformer" in obj.__repr__():
            self.hasFitted = True
            self.pt = obj
            return self.pt
        else:
            raise Exception(f"Error: File {filepath} not a PTransformer object")

if __name__ == "__main__":
    df = pd.read_csv("../troop_movements.csv",
                     usecols = ["unit_type",
                                "empire_or_resistance",
                                "location_x",
                                "location_y",
                                "destination_x",
                                "destination_y",
                                "homeworld"])
    p = PProcessor()
    p.fit(df)
    """WARNING: DO NOT SAVE / LOAD FROM MAIN"""
    # p.save("pkl/ptransformer.pkl")
    # pt = p.load("pkl/ptransformer.pkl")
    # x = pt.transform(df)
    # print(x.tail())