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
    prd = ['dce_gse_brst','brst_bgse','brst_jgse','brst_dis_n','brst_desm_n','brst_dis_vgse','brst_desm_v']
    names = [['ex','ey','ez'],['bx','by','bz'],['jx','jy','jz'],['i_density'],['e_density'],['vx_i','vy_i','vz_i'],['vx_e','vy_e','vz_e']]
    for k,product in enumerate(products):
        product = product.xmlid #return e.g. mms4_brst_dis_vgse 
        product = product[5:] #suppress the "mmsN_"
        i_name = prd.index(product)
        data[k].columns[:]=names[i_name]


def downsample(data):
    '''
    Downsample the data to the lowest frequency in order to make coherent dataset without NaN when concatenated.
    Keep the first lowest frequency for timestamps index.
    '''
    sizes = [len(data[k].to_dataframe()) for k in range(len(data))]
    fmin, imin = min(sizes), sizes.index(min(sizes))
    print(sizes)
    time = data[imin].to_dataframe().index
    df = pd.concat([pd.concat([dx.iloc[k*sizes[j]//fmin:min(sizes[j],(k+1)*sizes[j]//fmin),:].mean(axis=0) for k in range(fmin)],axis=1).T for j,dx in enumerate([dat.to_dataframe() for dat in data])],axis = 1)
    df = df.set_index(time)
    return df

def keys(t1,t2):
    '''
    Return the correct keys for the dictionnary of timetables according to the months.
    '''
    return['MMS_Burst_Mode_'+dt_keys for dt_keys in pd.date_range(start=t1.strftime('%Y-%m-01') , end=t2.strftime('%Y-%m-'+str(calendar.monthrange(t2.year, t2.month)[1])), freq='ME',inclusive='both').strftime('%Y%B')]

    
if __name__ == "__main__":

    ## Global interval - all the data will be inside this range
    sat = 'mms1'
    t1 = datetime(2015,9,1,0,0,0)
    t2 = datetime(2016,8,28,0,0,0)

    print("Data interval:", t1,"to", t2)

    MMStree = spz.inventories.tree.amda.Parameters.MMS
    products=[MMStree.MMS1.EDP.burst.mms1_dce_brst.mms1_dce_gse_brst, #electric field
            MMStree.MMS1.FGM.mms1_fgm_brst.mms1_brst_bgse, #magnetic field
            MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_current.mms1_brst_jgse, #current
            #MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_dism.mms1_brst_dis_n, #ion density
            MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_dism.mms1_brst_dis_vgse, #ion velocity
            MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_desm.mms1_brst_desm_n, #electron density
            MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_desm.mms1_brst_desm_v, #electron velocity
            #MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_dism.mms1_brst_dispres, #ion pressure tensor
            #MMStree.MMS1.FPI.burst_mode.mms1_fpi_brst_desm.mms1_brst_desmpres # electron pressure tensor
            ]
    
    burst_timetables = spz.inventories.tree.amda.TimeTables.SharedTimeTables.MMS_BURST_MODE
    bursts = pd.concat([amda.get_timetable(burst_timetables.__dict__[k],disable_cache=True).to_dataframe() for k in keys(t1,t2)])
    bursts = bursts.where((str(t1.date()) < bursts)&(bursts < str(t2.date()))).dropna()

    data_times = pd.DataFrame(np.zeros((len(bursts),4)),columns=['t_import','data_size [GB]','t_downsample','t_saving'])

    for k,event in enumerate(bursts.values):
        try:
            print("Loading",event[0], "to", event[1], "data with Speasy")
            start = time.time()
            data = spz.get_data(products,event[0],event[1],disable_cache=True)
            end = time.time()
            data_times.iloc[k,0] = end-start
            print("Loading OK, time elapsed: ",data_times.iloc[k,0])
            data_times.iloc[k,1] = np.sum([data[k].nbytes for k in range(len(data))])/1e6
            print("Data size: ",data_times.iloc[k,1]," MB")
            rename(products,data)
            print("Downsampling data to the lowest frequency")
            start = time.time()
            df = downsample(data) #no need to downsample by events because its already with bursts and events
            end = time.time()
            data_times.iloc[k,2] = end-start
            print("Downsampling OK, time elapsed: ",data_times.iloc[k,2])
            print("Saving data to HDF5")
            start = time.time()
            df.to_hdf('./data.hdf5',key=sat+"_"+datetime.strftime(event[0],format='%Y_%m_%dT%H_%M_%S'),mode='a', dropna=True)
            end = time.time()
            data_times.iloc[k,3] = end-start
            print("Saving OK, time elapsed: ",data_times.iloc[k,3])
        except Exception as e:
            print('Data error, burst is skipped')
            print(e)
            continue