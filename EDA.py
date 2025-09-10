from models import Models 
from plots import Plots
from data import Data
from clustering import ClusteringModels




class EDA:
    def __init__(self, df):
        self.df = df.copy()
        self.data = Data(df)
        self.plot = Plots()
        self.cluster = ClusteringModels()
        self.models = Models()

