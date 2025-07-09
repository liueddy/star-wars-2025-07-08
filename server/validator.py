from processor import PProcessor
import joblib

class Validator:
    """The following class serves as a validator
    to confirm whether unsafe user inputs are
    properly formatted and converts the input
    features into format which a given model
    can use."""
    def __init__(self,proc_pkl,model_pkl):
        self.pproc = PProcessor().load(proc_pkl)
        self.model = joblib.load(str(model_pkl))

