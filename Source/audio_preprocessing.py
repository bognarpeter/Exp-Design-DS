import pandas as pd
import numpy as np
from pathlib import Path

#process audio descriptors
pathlist = Path("../CoE_dataset/Dev_Set/audio_descriptors/").glob('**/*.csv')
pathlist = [str(pl) for pl in pathlist]
pathlist.sort()

audio_df = pd.DataFrame(columns=range(0,13))

for path in pathlist:
     _df = pd.read_csv(path, sep=",", header=None)
     _df = _df.T.dropna().mean()
     features = _df.tolist()
     audio_df = audio_df.append(pd.Series(features, index=audio_df.columns ), ignore_index=True)

audio_df = audio_df.reset_index(drop=True)

#process ground truth
ground_truth = pd.read_excel("../CoE_dataset/Dev_Set/dev_set_groundtruth_and_trailers.xls")
ground_truth = ground_truth.sort_values("movie")
ground_truth = ground_truth.reset_index(drop=True)
ground_truth = ground_truth.drop([ 'movie', 'filename','trailer' ], axis=1)

result = ground_truth.join(audio_df)

# Write out to results directory
result.to_csv("../Results/audio.csv",index=False)
