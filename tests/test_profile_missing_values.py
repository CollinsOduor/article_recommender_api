import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, RobustScaler

import app


class ProfileMissingValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = pd.DataFrame(
            [
                {"age": 20.0, "gender": "female", "jobtitle": "teacher",
                 "subscription_status": "registered", "allow_marketing": True},
                {"age": 40.0, "gender": "male", "jobtitle": "writer",
                 "subscription_status": "subscribed", "allow_marketing": False},
            ]
        )
        frame["allow_marketing"] = frame["allow_marketing"].astype(object)
        cls.preprocessor = ColumnTransformer(
            [
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                ]), ["age"]),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
                ]), ["gender", "jobtitle"]),
                ("ord", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=10)),
                ]), ["subscription_status"]),
                ("bool", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=False)),
                    ("to_int", FunctionTransformer(
                        np.asarray, kw_args={"dtype": np.int64}, feature_names_out="one-to-one"
                    )),
                ]), ["allow_marketing"]),
            ]
        ).fit(frame)
        cls.resources = SimpleNamespace(
            preprocessor=cls.preprocessor,
            profile_cols=list(cls.preprocessor.get_feature_names_out()) + list(app.TIME_FEATURES),
        )
        cls.time = datetime(2026, 11, 20, 15, 51, tzinfo=timezone.utc)

    def test_all_null_profile_is_imputed(self):
        vector = app._profile_vector(self.resources, app.UserProfile(), self.time)
        expected = self.preprocessor.transform(pd.DataFrame([{
            "age": np.nan, "gender": np.nan, "jobtitle": np.nan,
            "subscription_status": np.nan, "allow_marketing": np.nan,
        }]))
        np.testing.assert_allclose(vector[0, :-4], expected[0])
        self.assertTrue(np.isfinite(vector).all())

    def test_partial_profile_preserves_false_and_other_values(self):
        profile = app.UserProfile(age=34, gender="female", allow_marketing=False)
        vector = app._profile_vector(self.resources, profile, self.time)
        expected = self.preprocessor.transform(pd.DataFrame([{
            "age": 34.0, "gender": "female", "jobtitle": np.nan,
            "subscription_status": np.nan, "allow_marketing": False,
        }]))
        np.testing.assert_allclose(vector[0, :-4], expected[0])


if __name__ == "__main__":
    unittest.main()
