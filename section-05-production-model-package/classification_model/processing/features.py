from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class cabin():

    def get_first_cabin(row):
        try:
            return row.split()[0]
        except:
            return np.nan


class passenger():

   def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
