from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from src.features.build_features import FeatureEngineer

numerical_features = ['work_year', 'remote_ratio', 'Is_Lead', 'Is_manager', 'Is_director', 'Is_Principal']

ordinal_features = ['experience_level', 'company_size']

nominal_features = ['company_location', 'employee_residence', 'job_category', 'employment_type']


numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

nominal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[
        ['EN', 'MI', 'SE', 'EX'],
        ['S', 'M', 'L']
    ]))
])

preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_features)
    ("cat", nominal_transformer, nominal_features)
    ("ord", ordinal_transformer, ordinal_features)
])

def create_pipeline():
    model = Pipeline([
        ('Feature_engineering', FeatureEngineer()),
        ("Preprocessing", preprocessor())
        ("regressor", LinearRegression())
    ])
    return model

