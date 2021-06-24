import pandas as pd
import matplotlib.pyplot as plt
import math
from uncertainties import ufloat
from uncertainties.umath import *

def eff(group_df,var,EleType,Wt):
    signalpass=len(group_df.query('('+var+' == 1) & (EleType == '+str(EleType)+')'))
    signalpass=ufloat(signalpass,math.sqrt(signalpass))
    signaltotal=len(group_df.query('(EleType == '+str(EleType)+')'))
    signaltoal=ufloat(signaltotal,math.sqrt(signaltotal))
    signaleff=signalpass/(signaltotal)
    #print(str(signaleff))
    return [signaleff.n, signaleff.s]

def EffTrend(cat='',var='',Wt='',groupbyvar='',ptbins=[],label='',title='',plotname='',df=pd.DataFrame(),plot_dir=''):
    figMVAComp, axesComp = plt.subplots(1,1, figsize=(5, 5))

    ax=axesComp

    DY_list=[]
    bcToE_list=[]
    QCDEME_list=[]
    Tau_list=[]
    GJet_list=[]
    ptbinsi=[]
    
    DY_liste=[]
    bcToE_liste=[]
    QCDEME_liste=[]
    Tau_liste=[]
    GJet_liste=[]
    ptbinsmy=ptbins[:-1]
    for i,group_df in df.groupby(groupbyvar):
        DY_list.append(eff(group_df,var,0,Wt)[0])
        bcToE_list.append(eff(group_df,var,1,Wt)[0])
        QCDEME_list.append(eff(group_df,var,2,Wt)[0])
        Tau_list.append(eff(group_df,var,3,Wt)[0])
        GJet_list.append(eff(group_df,var,4,Wt)[0])
        
        DY_liste.append(eff(group_df,var,0,Wt)[1])
        bcToE_liste.append(eff(group_df,var,1,Wt)[1])
        QCDEME_liste.append(eff(group_df,var,2,Wt)[1])
        Tau_liste.append(eff(group_df,var,3,Wt)[1])
        GJet_liste.append(eff(group_df,var,4,Wt)[1])
        
        ptbinsi.append(i)
    color=['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628']
    #"e_DY","efrom_QCDbcToE","efrom_QCDEMEnriched","efrom_TauGun","efrom_GJet"    
    ax.errorbar(ptbinsmy,DY_list,yerr=DY_liste,markersize=1,marker='o',label='e_DY eff',color=color[0])
    ax.errorbar(ptbinsmy,bcToE_list,yerr=bcToE_liste,markersize=1,marker='o',label='efrom_QCDbcToE eff',color=color[1])
    ax.errorbar(ptbinsmy,QCDEME_list,yerr=QCDEME_liste,markersize=1,marker='o',label='efrom_QCDEMEnriched eff',color=color[2])
    ax.errorbar(ptbinsmy,Tau_list,yerr=Tau_liste, markersize=1,marker='o',label='efrom_TauGun eff',color=color[3])
    ax.errorbar(ptbinsmy,GJet_list,yerr=GJet_liste,markersize=1,marker='o',label='efrom_GJet eff',color=color[4])
    ax.set_ylim(0,1.1)
    ax.set_xlabel(label)
    #ax.set_xticklabels(ptbinsmy)
    #ax.set_xticklabels(str([ptbinsmy[i],ptbinsmy[i+1]]) for i in range(len(ptbinsmy)-1))
    ax.set_title(title)
    ax.legend()
    figMVAComp.savefig(plot_dir+plotname)