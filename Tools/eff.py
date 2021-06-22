import pandas as pd
import matplotlib.pyplot as plt
def eff(group_df,var,EleType,Wt):
    signalpass=group_df.loc[(group_df[var] == 1) & (group_df["EleType"] == EleType),Wt].sum()
    signalrej=group_df.loc[(group_df[var] == 0) & (group_df["EleType"] == EleType),Wt].sum()
    signaleff=signalpass/(signalpass+signalrej)
    return signaleff

def EffTrend(cat='',var='',Wt='',groupbyvar='',label='',title='',plotname='',df=pd.DataFrame(),plot_dir=''):
    figMVAComp, axesComp = plt.subplots(1,1, figsize=(5, 5))

    ax=axesComp

    DY_list=[]
    bcToE_list=[]
    QCDEME_list=[]
    Tau_list=[]
    GJet_list=[]
    ptbinsi=[]

    for i,group_df in df.groupby(groupbyvar):
        DY_list.append(eff(group_df,var,0,Wt))
        bcToE_list.append(eff(group_df,var,1,Wt))
        QCDEME_list.append(eff(group_df,var,2,Wt))
        Tau_list.append(eff(group_df,var,3,Wt))
        GJet_list.append(eff(group_df,var,4,Wt))
        ptbinsi.append(i)
    color=['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628']
    #"e_DY","efrom_QCDbcToE","efrom_QCDEMEnriched","efrom_TauGun","efrom_GJet"    
    ax.plot(ptbinsi,DY_list,markersize=4,marker='o',label='e_DY eff',color=color[0])
    ax.plot(ptbinsi,bcToE_list,markersize=4,marker='o',label='efrom_QCDbcToE eff',color=color[1])
    ax.plot(ptbinsi,QCDEME_list,markersize=4,marker='o',label='efrom_QCDEMEnriched eff',color=color[2])
    ax.plot(ptbinsi,Tau_list,markersize=4,marker='o',label='efrom_TauGun eff',color=color[3])
    ax.plot(ptbinsi,GJet_list,markersize=4,marker='o',label='efrom_GJet eff',color=color[4])
    ax.set_ylim(0,1.1)
    ax.set_xlabel(label)
    ax.set_title(title)
    ax.legend()
    figMVAComp.savefig(plot_dir+plotname)