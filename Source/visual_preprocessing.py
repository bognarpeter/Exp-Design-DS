import pathlib
import numpy as np
import pandas as pd


## VISUAL
# define the path
result_path = pathlib.Path.cwd() / '..' / 'Results'
visual_data_path = pathlib.Path.cwd() / '..' / 'CoE_dataset' / 'Test_Set' / 'vis_descriptors'

# define the pattern
currentPattern = "*.csv"
all_visual_files = []
for currentFile in visual_data_path.glob(currentPattern):
    all_visual_files.append(currentFile.parts[-1])

df = pd.read_csv(str(visual_data_path) + "/" + all_visual_files[0], header=None)

c = 1
for i in all_visual_files:
    if c==1:
        df_vis = pd.read_csv(str(visual_data_path) + "/" + i, header=None)
        all_features = []
        for index, row in df_vis.iterrows():
            for j in row.values:
                all_features.append(j)
        df_vis_concat = pd.DataFrame([all_features], columns=list(range(0, 1652)))
        name = i[:-4]
        df_vis_concat.insert(0, "filename", name)
        first_row = df_vis_concat.values.tolist()
        c=43
    else:
        df_vis = pd.read_csv(str(visual_data_path) + "/"  + i, header=None)
        all_features = []
        for index, row in df_vis.iterrows():
            for j in row.values:
                all_features.append(j)
        df_temp = pd.DataFrame([all_features], columns=list(range(0, 1652)))
        name = i[:-4]
        df_temp.insert(0, "filename", name)
        first_row.append(df_temp.values.tolist()[0])
df_vis_fin = pd.DataFrame(first_row)

z = list(range(0,1652))
z.insert(0, "filename")
colnames = z
df_vis_fin.columns = colnames

df_vis_fin.to_csv(str(result_path) + "/" + "visual_test.csv",index=False)
