#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas
import datetime
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


# Criterion for filtering float-like columns from x
def is_float_convertable(x):
    
    try: 
        x.astype(float)
    except ValueError:
        return False
    
    return True

# Criterion for filtering date-like columns from x
def is_datelike(x):
    
    try:
        pandas.to_datetime(x)
        
        if not is_float_convertable(x):
            return True
    except:
        return False
    
    return False


class CustomPreprocessing(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        self.ohe = OneHotEncoder(sparse=False)
        
        self.numeric = None
        self.dates = None
        self.strings = None
    
    def _process(self, X, method):
        
        # Group by contract, code and counter
        grouper = ["contractId", "code", "counter"]
        select = [column for column in X if not column in grouper]
        
        aggregated = list(group[select] for context, group 
                          in X.groupby(grouper))
        
        # Decompose past readings from readings for assessment
        past = []
        assess = []
        for x in aggregated:
            
            # Last item of the row
            assess_ = x.iloc[-1]
            
            # Append n-1 rows from group
            past_ = x.iloc[0:-1]
            
            # Hmm ... unfortunately if have to drop some values
            # that have not been available @ decision making time
            validity_changed_after_decision = \
                    (past_["validityChangedAt"] > assess_["readAt"])
            past_["valid"][validity_changed_after_decision] = np.nan
            # -> Problem with DB updates! ...
            
            past.append(past_)
            assess.append(assess_)
            
        X = [] # Features for predicting
        y = []
        
        select.remove("valid")
        for past_, assess_ in zip(past, assess):
            X.append(assess_[select].tolist() + \
                     past_[::-1].values.flatten().tolist())
            y.append(assess_["valid"])
            
        # Get matrix shape of X: padding of individual # of past items
        n_features = 3 * (len(select) + 1) + len(select)
        
        # Feature matrix needs to be 2D in this case. Since # of past readings 
        # varies, some data points need to be dropped, some other need to be 
        # padded (with na)
        Xout = []
        for Xi in X:
            
            n = len(Xi)
            
            if n > n_features:
                Xi = Xi[-n_features:]
            elif n < n_features:
                Xi = Xi + ([np.nan] * (n_features - n))
            
            Xout.append(Xi)
        
        # Feature matrix: Features characterizing the past reading history
        X = pandas.DataFrame(Xout) 
        # Target vector: Binary vector (1 -> valid, 0 -> invalid)
        y = np.array(y)
        
        # Obviously, we have multi-type data available. All types have to be 
        # converted into float. For converting categorical data, there are 
        # special encoding methodes available. 
        
        # Decompose data by type
        if "fit" in method:
            self.numerical = [column for column in X \
                              if is_float_convertable(X[column])]
            self.dates = [column for column in X \
                          if is_datelike(X[column])]
            self.strings = [column for column in X 
                            if not column in self.numerical + self.dates]
        
        #print(numerical)
        #print(dates)
        #print(strings)
        
        # Convert dates to float: Total seconds since millenium
        null_date = datetime.datetime(2000, 1, 1)
        for d in self.dates:
            X[d] = (pandas.to_datetime(X[d]) - null_date).dt.total_seconds()
            
        # Convert str columns: One-Hot-Encoding
        Xstr = X[self.strings].fillna("nan")
        Xstr = pandas.DataFrame(getattr(self.ohe, method)(Xstr))
        X.drop(columns=self.strings, inplace=True)
        X.columns = range(len(X.columns))
        string_columns = np.arange(max(X.columns) + 1,
                                   (max(X.columns) + Xstr.shape[1] + 1))
        X[string_columns] = Xstr
        
        return X
    
    def fit(self, X, y=None):
        
        self._process(X, "fit_transform")
        
        return self
    
    def transform(self, X):
        
        return self._process(X, "transform")
        