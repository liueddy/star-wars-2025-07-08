import pandas as pd
from pprocessor import PProcessor
import sklearn.model_selection
import sklearn.tree
from sklearn.tree import DecisionTreeClassifier
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

# perform fit_transform and save transformer
pproc = PProcessor()
t_df = pproc.fit_transform(df)
pproc.save()

# init predict column
y_col = "empire_or_resistance_resistance"

# split to 60-20-20 train-val-test
def tvt(X,y_col):
    train,test = sklearn.model_selection.train_test_split(X,stratify=X.loc[:,y_col],test_size=0.2)
    train,val  = sklearn.model_selection.train_test_split(train,stratify=train.loc[:,y_col],test_size=0.25)
    return train, val, test

# perform split:
train,val,test = tvt(t_df,y_col)
# print(train.columns)

# init model
model = DecisionTreeClassifier()

def train_score(model,y_col):
    """train then score the model"""
    model.fit(train.loc[:,t_df.columns != f"{y_col}"],train.loc[:,f"{y_col}"])
    trscore = model.score(train.loc[:,t_df.columns != f"{y_col}"],train.loc[:,f"{y_col}"])
    vscore =  model.score(  val.loc[:,t_df.columns != f"{y_col}"],  val.loc[:,f"{y_col}"])
    tescore = model.score( test.loc[:,t_df.columns != f"{y_col}"], test.loc[:,f"{y_col}"])

    print("train: ", trscore)
    print("val: ", vscore)
    print("test: ",tescore)
    return model

# train and score model 
train_score(model,y_col)

if __name__ == "__main__":
    # save model to pkl file
    joblib.dump(model, 'pkl/model.pkl')

# for i in range(len(model.feature_names_in_)):
#     print(
#         t_df.columns[i],model.feature_importances_[i]
#     )
# print("intercept\n",model.intercept_)
# print("coef\n",model.coef_)

# plt.figure(figsize=(20,15))
# sklearn.tree.plot_tree(model)
# plt.savefig("decision_tree.png")