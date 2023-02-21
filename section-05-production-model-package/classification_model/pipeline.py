

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# to persist the model and the scaler
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer,
)

from feature_engine.encoding import (
    RareLabelEncoder,
    OrdinalEncoder,
)
from feature_engine.encoding import OneHotEncoder
titanic_pipe = Pipeline([

   ('categorical_imputation',
                 CategoricalImputer(variables=['sex', 'cabin', 'embarked',
                                               'title'])),
                ('missing_indicator',
                 AddMissingIndicator(variables=['age', 'fare'])),
                ('median_imputation',
                 MeanMedianImputer(variables=['age', 'fare'])),
                ('extract_letter',
                 ExtractLetterTransformer()),
                ('rare_label_encoder',
                 RareLabelEncoder(n_categories=1,
                                  variables=['sex', 'cabin', 'embarked',
                                             'title'])),
                ('categorical_encoder',
                 OneHotEncoder(drop_last=True,
                               variables=['sex', 'cabin', 'embarked',
                                          'title'])),
                ('scaler', StandardScaler()),
                ('Logit', LogisticRegression(C=0.0005, random_state=0))
])
