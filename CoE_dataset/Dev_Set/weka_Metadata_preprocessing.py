import pathlib
import pandas as pd
import xml.etree.ElementTree as et

# define the path
path="D:/TU/ED/EX2/CoE_dataset/Dev_Set/XML"
currentDirectory = pathlib.Path(path)

# define the pattern
currentPattern = "*.xml"
all_xml_files = []
for currentFile in currentDirectory.glob(currentPattern):
    all_xml_files.append(currentFile.parts[-1])
    
#print(all_xml_files)

def parse_XML(xml_file, name, df_cols):
            global out_df

            xtree = et.parse(xml_file)
            xroot = xtree.getroot()
            fin = []
            res = []
            res.append(name.split(".x")[0])
            
            for node in xroot:
                for i in node_cols:
                    res.append(node.attrib.get(i))

            fin.append(res)
            print(fin)
            out_df = pd.DataFrame(data=fin, columns =meta_cols)
            return out_df
        
def parse_XML1(xml_file, name, df_cols, df):
            global out_df
            xtree = et.parse(xml_file)
            xroot = xtree.getroot()
            fin = []
            res = []
            res.append(name.split(".x")[0])
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

c = 1
for i in all_xml_files:
    if c==1:
        parse_XML("XML/" + i, i,  meta_cols)
        c=43
    else:
        parse_XML1("XML/"+i, i, meta_cols, out_df)
		
df_ground = pd.read_excel("dev_set_groundtruth_and_trailers.xls",index_col=False)


a = df_ground["filename"].sort_values()
b = out_df["filename"].sort_values()

for i in a.values:
    if i in b.values:
        continue
    else:
        print(i)
		
		
df_join = pd.merge(df_ground, out_df, on="filename", how="inner")

df_join = df_join.drop(["movie", "filename", "trailer"], axis=1)

import numpy as np

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
				
				
df_join.to_csv("metadata.csv",index=False)

#WEKA 10 fold CV, 
#                TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area
#Weighted Avg.    0,558    0,519    0,557      0,558    0,468      0,067    0,522     0,526     