import uproot
import glob
import pandas as pd

file_list = glob.glob("egmPrachu/egmNtuple*.root")    # This will create a list of all your files in the directory
Tree = "egmNtuplizer/EventTree"    # Tree location in the root file
branches=["pho*"]    # Branches you want to select (Skimming)
cut="phoPt>10"    # Cuts (Trimming)
nameofdf="df.csv.gzip"    # Name of final file

#----------------------------------------
df=pd.DataFrame()
for file in file_list:
    dfa=uproot.open(file)[Tree].pandas.df(branches=branches,flatten=True).reset_index(drop=True)
    dfa.query(cut,inplace=True)
    df=pd.concat([df,dfa])
df.to_csv(nameofdf,compression='gzip')
#----------------------------------------
