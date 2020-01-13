import pathlib
import numpy as np
import pandas as pd
import xml.etree.ElementTree as et

# define the path
result_path = pathlib.Path.cwd() / '..' / 'Results'
xml_data_path = pathlib.Path.cwd() / '..' / 'CoE_dataset' / 'Dev_Set' / 'XML'

# define the pattern
currentPattern = "*.xml"
all_xml_files = []
for currentFile in xml_data_path.glob(currentPattern):
    #all_xml_files.append(currentFile.parts[-1])
    all_xml_files.append(currentFile)

def parse_XML(xml_file, df_cols):
    global out_df

    xtree = et.parse(str(xml_file))
    xroot = xtree.getroot()
    fin = []
    res = []
    res.append(xml_file.parts[-1].split(".")[0])

    for node in xroot:
        for i in node_cols:
            res.append(node.attrib.get(i))

    fin.append(res)
    out_df = pd.DataFrame(data=fin, columns =meta_cols)
    return out_df

def parse_XML1(xml_file, df_cols, df):
            global out_df
            xtree = et.parse(str(xml_file))
            xroot = xtree.getroot()
            fin = []
            res = []
            name = xml_file.parts[-1].split(".")[0]
            if name.split(".")[0] == "Mr":
                res.append("Mr._Go")
            else:
                res.append(name.split(".")[0])
            for node in xroot:
                for i in node_cols:
                    res.append(node.attrib.get(i))

            fin.append(res)
            df1 = pd.DataFrame(data=fin, columns =meta_cols)

            frames = [df, df1]

            out_df = pd.concat(frames)

            return out_df


meta_cols = ["filename","language", "year", "genre", "country",
            "runtime", "rated"]

node_cols = ["language", "year", "genre", "country",
            "runtime", "rated"]

user_rating_cols = ["metascore", "imdbRating", "tomatoRating", "tomatoUserRating"]
user_and_meta_cols = ["language", "year", "genre", "country",
            "runtime", "rated", "metascore", "imdbRating", "tomatoRating", "tomatoUserRating"]

user = False
meta = True

if user == True:
    #meta_cols=["filename","metascore", "imdbRating", "tomatoMeter", "tomatoUserMeter"]
    meta_cols=["filename","language", "year", "genre", "country",
            "runtime", "rated"]
    node_cols=user_rating_cols
if user == True and meta == True:
    meta_cols=["filename","language", "year", "genre", "country",
            "runtime", "rated", "metascore", "imdbRating", "tomatoRating", "tomatoUserRating"]
    node_cols=user_and_meta_cols

c = 1
for i in all_xml_files:
    if c==1:
        parse_XML(i,  meta_cols)
        c=43
    else:
        parse_XML1(i, meta_cols, out_df)

trailers_data_path = pathlib.Path.cwd() / '..' / 'CoE_dataset' / 'Dev_Set' / 'dev_set_groundtruth_and_trailers.xls'
df_ground = pd.read_excel(str(trailers_data_path),index_col=False)

a = df_ground["filename"].sort_values()
b = out_df["filename"].sort_values()

for i in a.values:
    if i in b.values:
        continue
    else:
        print(i)

df_join = pd.merge(df_ground, out_df, on="filename", how="inner")

if user == False or meta == True:
    count=0
    for i in df_join["genre"].values:
        df_join["genre"].values[count] = i.replace(" ", "")
        count+=1

    count=0
    for i in df_join["language"].values:
        df_join["language"].values[count] = i.replace(" ", "")
        count+=1

    count=0
    for i in df_join["country"].values:
        df_join["country"].values[count] = i.replace(" ", "")
        count+=1

    genre = df_join.genre.str.split(",", expand=True).stack()
    language = df_join.language.str.split(",", expand=True).stack()
    country = df_join.country.str.split(",", expand=True).stack()

    df_join = pd.concat([df_join,
                pd.get_dummies(genre, prefix="g").groupby(level=0).sum()],axis=1) \
                .drop(["genre"],axis=1)

    df_join = pd.concat([df_join,
                    pd.get_dummies(language, prefix="l").groupby(level=0).sum()],axis=1) \
                    .drop(["language"],axis=1)

    df_join = pd.concat([df_join,
                    pd.get_dummies(country, prefix="c").groupby(level=0).sum()],axis=1) \
                    .drop(["country"],axis=1)



if user == True:
    df_join.replace("N/A", np.nan, inplace=True)
df_join.columns
df_join = df_join.astype("category")

df_join.to_csv(str(result_path) + "/" + "meta_plus_ratings.csv",index=False)


## VISUAL
# define the path
visual_data_path = pathlib.Path.cwd() / '..' / 'CoE_dataset' / 'Dev_Set' / 'vis_descriptors'

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

df_join1 = pd.merge(df_ground, df_vis_fin, on="filename", how="inner")
df_join1 = df_join1.drop(["movie", "trailer"], axis=1)

df_join = pd.merge(df_join, df_join1, on="filename", how="inner")
df_join  = df_join.drop(["filename", "movie", "trailer"], axis=1)

print("RESULTS:")
print(df_join)

df_join.to_csv(str(result_path) + "/" + "meta+visual.csv",index=False)
