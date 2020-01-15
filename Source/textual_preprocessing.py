import pandas as pd
import numpy as np

#Read Training data
textual_df = pd.read_csv("../CoE_dataset/Dev_Set/text_descriptors/tdf_idf_dev.csv", sep=",")
ground_truth = pd.read_excel("../CoE_dataset/Dev_Set/dev_set_groundtruth_and_trailers.xls")
# Preprocess the data
# Transpose and drop missing values
textual_df = textual_df.T.dropna()
# Sort Movies alphabetically
ground_truth = ground_truth.sort_values("movie")
# Reset index for merging
ground_truth = ground_truth.reset_index(drop=True)
# Do the same for textual df
textual_df = textual_df.reset_index(drop=True)
# Drop unnecessary columns
ground_truth = ground_truth.drop([ 'movie', 'filename','trailer' ], axis=1)
# Join everything
result = ground_truth.join(textual_df)

# Write out to results directory
result.to_csv("../Results/textual.csv",index=False)
