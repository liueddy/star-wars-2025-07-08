List of classes:
- app.py        - Runs the Flask server
- pprocessor.py - Preprocesses data, saves ptransformers to pkl.
- swdantic.py   - Star Wars-specific Pydantic type-checker
- validator.py  - Validates query parameters with given
                  PyDantic type-checker.  Then transforms 
                  data and predicts from given pkl files.

List of scripts:
- train-save-model.py - Processes data from mongo
                        Saves pprocessor to pkl
                        Splits data to tvt
                        Fits model, shows acc
                        Saves model to pkl