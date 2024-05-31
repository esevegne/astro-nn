# Libraries
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import speasy as spz
from speasy import amda
from datetime import datetime, timedelta
import time
import calendar

#Physical  units
e = 1.602176634e-19
m_i = 1.67262192369e-27
m_e = 9.1093837015e-31

def rename(products,data):
    '''
    Rename the columns of dataset based on hard-coded matching between prods name (product.xmlid) and convenient name e.g. 'vx_e' for x velocity of electrons.
    '''
    prd = ['edp-dce','brst_bgse','brst_jgse','brst_dis_n','brst_desm_n','brst_dis_vgse','brst_desm_v']
    names = [['ex','ey','ez'],['bx','by','bz'],['jx','jy','jz'],['i_density'],['e_density'],['vx_i','vy_i','vz_i'],['vx_e','vy_e','vz_e']]
    for k,product in enumerate(products):
        product = product.xmlid #return e.g. mms4_brst_dis_vgse 
        product = product[5:] #suppress the "mmsN_"
        i_name = prd.index(product)
        data[k].columns =names[i_name]
        
def re_index(df,index):
    '''
    Just change the index to simplier code later
    '''
    dg = df.copy()
    dg.index = index
    return dg

def downsample(df,index):
    '''
    Downsample the selected dataframe to the selected lenght, and select index as the new indexes for df.
    '''
    dlenght = len(index)
    dg = pd.concat([df.iloc[k*len(df)//dlenght:min(len(df),(k+1)*len(df)//dlenght),:].mean(axis=0) for k in range(dlenght)],axis=1).T
    dg.index = index
    return dg

def upsample(df,index):
    '''
    Downsample the selected dataframe to the selected lenght, and select index as the new indexes for df.
    '''
    dg = pd.DataFrame(index=index,columns=df.columns)
    for col in df.columns:
        dg[col]=np.interp(index, df.index, df[col].values)
    return dg

def keys(t1,t2):
    '''
    Return the correct keys for the dictionnary of timetables according to the months.
    '''
    return['MMS_Burst_Mode_'+dt_keys for dt_keys in pd.date_range(start=t1.strftime('%Y-%m-01') , end=t2.strftime('%Y-%m-'+str(calendar.monthrange(t2.year, t2.month)[1])), freq='ME',inclusive='both').strftime('%Y%B')]

def sat_group(sat=str):
    '''
    Return the corresponding products for a selected satelite
    '''
    if sat[:3] == 'mms':
        MMStree = spz.inventories.tree.amda.Parameters.MMS
        if sat[-1] == '1':
            products = [MMStree.MMS1.EDP.fast.mms1_edp_dce, #electric field
                MMStree.MMS1.FGM.mms1_fgm_brst.mms1_brst_bgse, #magnetic field
                MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_current.mms1_brst_jgse, #current
                #MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_dism.mms1_brst_dis_n, #ion density
                MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_dism.mms1_brst_dis_vgse, #ion velocity
                MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_desm.mms1_brst_desm_n, #electron density
                MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_desm.mms1_brst_desm_v, #electron velocity
                #MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_dism.mms1_brst_dispres, #ion pressure tensor
                #MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_desm.mms1_brst_desmpres # electron pressure tensor
                MMStree.MMS1.Ephemeris.mms1_orb_t89d.mms1_xyz_gse #position xyz
                ]
        if sat[-1] == '2':
            products = [MMStree.MMS2.EDP.fast.mms2_edp_dce, #electric field
                MMStree.MMS2.FGM.mms2_fgm_brst.mms2_brst_bgse, #magnetic field
                MMStree.MMS2.FPI.burst_mode.mms2_fpi_brst_current.mms2_brst_jgse, #current
                #MMStree.MMS2.FPI.burst_mode.mms2_fpi_brst_dism.mms2_brst_dis_n, #ion density
                MMStree.MMS2.FPI.burst_mode.mms2_fpi_brst_dism.mms2_brst_dis_vgse, #ion velocity
                MMStree.MMS2.FPI.burst_mode.mms2_fpi_brst_desm.mms2_brst_desm_n, #electron density
                MMStree.MMS2.FPI.burst_mode.mms2_fpi_brst_desm.mms2_brst_desm_v, #electron velocity
                #MMStree.MMS2.FPI.burst_mode.mms2_fpi_brst_dism.mms2_brst_dispres, #ion pressure tensor
                #MMStree.MMS2.FPI.burst_mode.mms2_fpi_brst_desm.mms2_brst_desmpres # electron pressure tensor
                MMStree.MMS2.Ephemeris.mms2_orb_t89d.mms2_xyz_gse, #position xyz
                ]
        if sat[-1] == '3':
            products = [MMStree.MMS3.EDP.fast.mms3_edp_dce, #electric field
                MMStree.MMS3.FGM.mms3_fgm_brst.mms3_brst_bgse, #magnetic field
                MMStree.MMS3.FPI.burst_mode.mms3_fpi_brst_current.mms3_brst_jgse, #current
                #MMStree.MMS3.FPI.burst_mode.mms3_fpi_brst_dism.mms3_brst_dis_n, #ion density
                MMStree.MMS3.FPI.burst_mode.mms3_fpi_brst_dism.mms3_brst_dis_vgse, #ion velocity
                MMStree.MMS3.FPI.burst_mode.mms3_fpi_brst_desm.mms3_brst_desm_n, #electron density
                MMStree.MMS3.FPI.burst_mode.mms3_fpi_brst_desm.mms3_brst_desm_v, #electron velocity
                #MMStree.MMS3.FPI.burst_mode.mms3_fpi_brst_dism.mms3_brst_dispres, #ion pressure tensor
                #MMStree.MMS3.FPI.burst_mode.mms3_fpi_brst_desm.mms3_brst_desmpres # electron pressure tensor
                MMStree.MMS3.Ephemeris.mms3_orb_t89d.mms3_xyz_gse, #position xyz
                ]
        if sat[-1] == '4':
            products = [MMStree.MMS4.EDP.fast.mms4_edp_dce, #electric field
                MMStree.MMS4.FGM.mms4_fgm_brst.mms4_brst_bgse, #magnetic field
                MMStree.MMS4.FPI.burst_mode.mms4_fpi_brst_current.mms4_brst_jgse, #current
                #MMStree.MMS4.FPI.burst_mode.mms4_fpi_brst_dism.mms4_brst_dis_n, #ion density
                MMStree.MMS4.FPI.burst_mode.mms4_fpi_brst_dism.mms4_brst_dis_vgse, #ion velocity
                MMStree.MMS4.FPI.burst_mode.mms4_fpi_brst_desm.mms4_brst_desm_n, #electron density
                MMStree.MMS4.FPI.burst_mode.mms4_fpi_brst_desm.mms4_brst_desm_v, #electron velocity
                #MMStree.MMS4.FPI.burst_mode.mms4_fpi_brst_dism.mms4_brst_dispres, #ion pressure tensor
                #MMStree.MMS4.FPI.burst_mode.mms4_fpi_brst_desm.mms4_brst_desmpres # electron pressure tensor
                MMStree.MMS4.Ephemeris.mms4_orb_t89d.mms4_xyz_gse, #position xyz
                ]
    return products

if __name__ == "__main__":
    import warnings
    from tables import NaturalNameWarning
    warnings.filterwarnings('ignore', category=NaturalNameWarning)

    ## Spacecraft
    sats = ['mms1','mms2','mms3','mms4']
    
    ## Time period to download
    times = [(datetime(2015,9,7,0,0,0),datetime(2016,1,1,0,0,0))]

    ## Stawarz & al (2021) - Comparative Analysis of the Various Generalized Ohm's Law Terms in Magnetosheath Turbulence as Observed by Magnetospheric Multiscale
    #times = [(datetime(2016,9,28,16,30,0),datetime(2016,9,28,17,30,0)),(datetime(2016,12,9,8,30,0),datetime(2016,12,9,9,30,0)),(datetime(2017,1,28,8,30,0),datetime(2017,1,28,9,30,0))]
    
    ##Data path (where to save data)
    #data_path = "./data_stawarz.hdf5"
    data_path = "./data_training.hdf5"

    for sat in sats:
        print(f"Spacecraft = {sat}")
        for t1,t2 in  times:
            print("Data interval:", t1, "to", t2)
            products = sat_group(sat=sat)
            kept_f = 'brst_desm_n'
            for k,product in enumerate(products):
                if product.xmlid[5:]==kept_f:
                    kept_k = k
                    break

            burst_timetables = spz.inventories.tree.amda.TimeTables.SharedTimeTables.MMS_BURST_MODE
            bursts = pd.concat([amda.get_timetable(burst_timetables.__dict__[k],disable_cache=True).to_dataframe() for k in keys(t1,t2)])
            bursts = bursts.where((str(t1) < bursts)&(bursts < str(t2))).dropna()
            data_times = pd.DataFrame(np.zeros((len(bursts),4)),columns=['t_import','data_size [GB]','t_downsample','t_saving'])

            for k,event in enumerate(bursts.values):
                try:
                    print("Loading",event[0], "to", event[1], "data with Speasy")
                    start = time.time()
                    data = spz.get_data(products,event[0],event[1],disable_cache=True)
                    end = time.time()
                    data_times.iloc[k,0] = end-start
                    print("Loading OK, time elapsed: ",data_times.iloc[k,0])
                    
                    data = [data[0]['e_gse'].to_dataframe()]+[k.to_dataframe() for k in data[1:]]+[data[0]['dce : qual'].to_dataframe()]
                    rename(products,data)
                    kept_idx = data[kept_k].index
                    print("Resampling everything to the choosen frequency")
                    start = time.time()
                    df = pd.concat([pd.concat([downsample(df=data[ind],index=kept_idx) for ind in np.argwhere(np.array([len(dat_) for dat_ in data]) > len(kept_idx))[:,0] ],axis=1),
                                    pd.concat([re_index(df=data[ind],index=kept_idx) for ind in np.argwhere(np.array([len(dat_) for dat_ in data]) ==len(kept_idx))[:,0] ],axis=1),
                                    pd.concat([upsample(df=data[ind],index=kept_idx) for ind in np.argwhere(np.array([len(dat_) for dat_ in data]) < len(kept_idx))[:,0] ],axis=1)],
                                axis=1)
                    
                    df = df.where(df['dce : qual']>=2).dropna().drop('dce : qual',axis='columns') #select only ok / good data

                    end = time.time()
                    data_times.iloc[k,2] = end-start

                    print("Downsampling OK, time elapsed: ",data_times.iloc[k,2])
                    print("Saving data to HDF5")
                    start = time.time()
                    df.to_hdf(data_path,key=f"{sat}/{datetime.strftime(event[0],format='%Y_%m_%dT%H_%M_%S')}",mode='a', dropna=True)
                    end = time.time()
                    data_times.iloc[k,3] = end-start
                    print("Saving OK, time elapsed: ",data_times.iloc[k,3])

                except Exception as e:
                    print('Data error, burst is skipped')
                    print(e)
                    continue