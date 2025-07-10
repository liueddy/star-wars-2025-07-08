from pymongo import MongoClient
import pandas as pd
import numpy as np

cli = MongoClient()
db = cli.swapi2
# col = db["troop_movements"]
# print(
#     pd.DataFrame(list(col.find({}))).loc[:,["unit_type", 
#                                             "empire_or_resistance",
#                                             "location_x",
#                                             "location_y",
#                                             "destination_x",
#                                             "destination_y",
#                                             "homeworld"]]
# )

col = db["home_worlds"]
print(
    np.abs(pd.DataFrame(list(col.find({})))["rebel_likelihood"].to_numpy() - 0.5)
)
