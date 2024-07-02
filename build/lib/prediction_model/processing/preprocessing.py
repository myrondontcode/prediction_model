from prediction_model.config import config
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.mean_dict = {}
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.mean_dict[col], inplace=True)
        return X

class ModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.mode_dict = {}
        for col in self.variables:
            self.mode_dict[col] = X[col].mode()[0]  # Taking the first mode
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.mode_dict[col], inplace=True)
        return X

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.variables_to_drop)
        return X
    
class DomainProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, variable_to_modify=None, variable_to_add=None):
        self.variable_to_modify = variable_to_modify
        self.variable_to_add = variable_to_add
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        # Debugging: Print types and shapes
        print(f"Before conversion: type(X[self.variable_to_add]) = {type(X[self.variable_to_add])}")
        print(f"Shape of X[self.variable_to_add]: {X[self.variable_to_add].shape}")

        # Ensure the column is numeric
        if self.variable_to_add in X.columns:
            if isinstance(X[self.variable_to_add], pd.DataFrame):
                X[self.variable_to_add] = X[self.variable_to_add].iloc[:, 0]
            X[self.variable_to_add] = pd.to_numeric(X[self.variable_to_add], errors='coerce')
        else:
            raise ValueError(f"Column {self.variable_to_add} not found in input data")

        print(f"After conversion: type(X[self.variable_to_add]) = {type(X[self.variable_to_add])}")
        print(f"After conversion: {X[self.variable_to_add].head()}")

        if isinstance(self.variable_to_modify, list):
            for feature in self.variable_to_modify:
                if feature in X.columns:
                    X[feature] = pd.to_numeric(X[feature], errors='coerce')
                    X[feature] = X[feature] + X[self.variable_to_add]
                else:
                    raise ValueError(f"Column {feature} not found in input data")
        else:
            if self.variable_to_modify in X.columns:
                X[self.variable_to_modify] = pd.to_numeric(X[self.variable_to_modify], errors='coerce')
                X[self.variable_to_modify] = X[self.variable_to_modify] + X[self.variable_to_add]
            else:
                raise ValueError(f"Column {self.variable_to_modify} not found in input data")
        
        return X

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.label_dict = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index 
            self.label_dict[var] = {k: i for i, k in enumerate(t, 0)}
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X

class LogTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col] + 1)  # Adding 1 to avoid log(0)
        return X
