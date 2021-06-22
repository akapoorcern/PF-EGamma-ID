import pandas as pd
import uproot3 as uproot
from dask import delayed
import dask.dataframe as dd
import gc

def clean(df,xsecwt=1,sel='',EleType=99):
    
    df.query(sel,inplace = True)
    df["EleType"]=EleType
    df["NewWt"]=1.0
    df["rwt"]=1.0
    df["xsecwt"]=xsecwt
    
    df['ele_fbrem']=df['ele_fbrem'].astype(float)
    df['ele_convDist']=df['ele_convDist'].astype(float)
    df['ele_convDcot']=df['ele_convDcot'].astype(float)
    df.loc[df["EleMVACats"] == 4, "EleMVACats"] = 3
    print('Selected for EleType ' +str(EleType)+ ' =' + str(df['NewWt'].sum()))
    return df

def daskframe_from_rootfiles(processes, treepath,branches):
    def get_df(file, xsecwt, selection, EleType, treepath=None,branches=['ele*']):
        tree = uproot.open(file)[treepath]
        print(file)
        #return clean(tree.arrays(branches,library="pd"),xsec=xsec,EleType=EleType)
        return clean(tree.pandas.df(branches=branches),xsecwt=xsecwt,sel=selection,EleType=EleType)
        #return tree.pandas.df()

    dfs=[]
    for process in processes:
        dfs.append(delayed(get_df)(process['path'],process['xsecwt'],process['selection'] +" & " + process['CommonSelection'],process['EleType'],treepath, branches))
    daskframe = dd.from_delayed(dfs)
    return daskframe

