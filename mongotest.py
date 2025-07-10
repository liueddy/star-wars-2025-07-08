from pymongo import MongoClient
import pandas as pd

cli = MongoClient()
db = cli.swapi2
col = db["troop_movements"]
print(
    pd.DataFrame(list(col.find({}))).loc[:,["unit_type", 
                                            "empire_or_resistance",
                                            "location_x",
                                            "location_y",
                                            "destination_x",
                                            "destination_y",
                                            "homeworld"]]
)