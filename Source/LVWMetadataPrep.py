import random
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def preprocess_metadata(metadata_df):
    # Convert the runtime into int
    metadata_df["runtime"] = metadata_df["runtime"].apply(lambda x: x.split(" min")[0])
    metadata_df["runtime"] = metadata_df["runtime"].astype(float)

    # Features
    metadata_x = metadata_df.drop("goodforairplane", axis=1)
    # Target
    metadata_Y = metadata_df["goodforairplane"]

    # Same for test set
    # Features
    # metadata_test_x = metadata_test_df.drop("goodforairplane", axis=1)
    # Target
    # metadata_test_Y = metadata_test_df["goodforairplane"]

    # Map the age rating to the corresponding integer
    age_mapping = {"PG-13": 13., "R": 17., "TV-MA": 18., "G": 1., "PG": 10., "NOT RATED": 0., "APPROVED": 0.,
                   np.nan: 0.}
    metadata_x["rated"] = metadata_x["rated"].map(age_mapping)

    # Impute the missing values of training data
    values = metadata_x.iloc[:, list(range(3, 7))].values
    imputer = IterativeImputer(random_state=1)
    transformed_values = imputer.fit_transform(values)
    colnames = metadata_x.columns[3:7]
    metadata_x[colnames] = transformed_values

    return (metadata_x, metadata_Y)

def preprocess_test_metadata(metadata_df):
    # Map the age rating to the corresponding integer
    age_mapping = {"PG-13": 13., "R": 17., "TV-MA": 18., "G": 1., "PG": 10., "NOT RATED": 0., "APPROVED": 0.,
                   np.nan: 0.}
    metadata_df["rated"] = metadata_df["rated"].map(age_mapping)

    # Impute the missing values of training data
    values = metadata_df.iloc[:, list(range(2, 8))].values
    imputer = IterativeImputer(random_state=1)
    transformed_values = imputer.fit_transform(values)
    colnames = metadata_df.columns[2:8]
    metadata_df[colnames] = transformed_values
    return metadata_df


def main():
    ##### Preprocess the Testdata which is useable for ML
    metadata_testset = pd.read_csv("../CoE_dataset/Test_Set/metadata_test_plus_ratings_sorted.csv")
    metadata_testset = preprocess_test_metadata(metadata_testset)
    metadata_testset.to_csv("../Results/metadata_test_sorted.csv", index=False)


if __name__ == "__main__":
    main()
