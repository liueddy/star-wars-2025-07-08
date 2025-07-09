import pandas as pd
from pprocessor import PProcessor
import sklearn.model_selection
import sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import joblib

# create dataframe to train data
df = pd.read_csv("../troop_movements.csv",
                 usecols = ["unit_type",
                            "empire_or_resistance",
                            "location_x",
                            "location_y",
                            "destination_x",
                            "destination_y",
                            "homeworld"])

# init y_col
y_col_name = "empire_or_resistance"

# perform fit_transform and save transformer
pproc = PProcessor()
X = pproc.fit_transform(df.loc[:,df.columns != str(y_col_name)])
y = pd.DataFrame(df[str(y_col_name)]=="resistance").rename({"empire_or_resistance":"is_rebel"})
pproc.save()

# init predict column
# y_col = "empire_or_resistance_resistance"

# split to 60-20-20 train-val-test
def tvt(X,y):
    X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X,      y,      stratify=y,      test_size=0.2)
    X_train,X_val,y_train,y_val   = sklearn.model_selection.train_test_split(X_train,y_train,stratify=y_train,test_size=0.25)
    return X_train,X_val,X_test,y_train,y_val,y_test

# perform split:
X_train,X_val,X_test,y_train,y_val,y_test = tvt(X,y)
print(X_train.columns)

# init model
model = DecisionTreeClassifier()
# model = LogisticRegression()

def train_score(model):
    """train then score the model"""
    model.fit(X_train,y_train)
    trscore = model.score(X_train,y_train)
    vscore =  model.score(  X_val,  y_val)
    tescore = model.score( X_test, y_test)

    print("train: ", trscore)
    print("val: ", vscore)
    print("test: ",tescore)
    return model

# train and score model
model = train_score(model)

if __name__ == "__main__":
    # save model to pkl file
    joblib.dump(model, 'pkl/model.pkl')