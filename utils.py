import time
import pandas as pd
import numpy as np
import gc

def get_count(df, cols, cname, value):
    df_count = pd.DataFrame(df.groupby(cols)[value].count()).reset_index()
    df_count.columns = cols + [cname]
    df = df.merge(df_count, on=cols, how='left')
    del df_count
    gc.collect()
    return df
	
def get_sum(df, cols, cname, value):
    df_count = pd.DataFrame(df.groupby(cols)[value].sum()).reset_index()
    df_count.columns = cols + [cname]
    df = df.merge(df_count, on=cols, how='left')
    del df_count
    gc.collect()
    return df

def get_mean(df, cols, cname, value):
    df_mean = pd.DataFrame(df.groupby(cols)[value].mean()).reset_index()
    df_mean.columns = cols + [cname]
    df = df.merge(df_mean, on=cols, how='left')
    del df_mean
    gc.collect()
    return df

def get_std(df, cols, cname, value):
    df_std = pd.DataFrame(df.groupby(cols)[value].std()).reset_index()
    df_std.columns = cols + [cname]
    df = df.merge(df_std, on=cols, how='left')
    del df_std
    gc.collect()
    return df

def get_nunique(df, cols, cname, value):
    df_nunique = pd.DataFrame(df.groupby(cols)[value].nunique()).reset_index()
    df_nunique.columns = cols + [cname]
    df = df.merge(df_nunique, on=cols, how='left')
    del df_nunique
    gc.collect()
    return df
    
def get_cumcount(df, cols, cname):
    df[cname] = df.groupby(cols).cumcount() + 1
    return df

def get_hour(datetime):
    return datetime.hour
def get_day(datetime):
    return datetime.day
	
def get_rank(order_by,group_by):
    _ord = np.lexsort((order_by, group_by))
    _cs1 = np.zeros(group_by.size,dtype=np.int)
    _prev_grp = group_by[_ord[0]]
    for i in xrange(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            if order_by[i0]==order_by[_ord[i-1]]:
                _cs1[i] = _cs1[i - 1]
            else:
                _cs1[i] = _cs1[i - 1] + 1
        else:
            _cs1[i] = 0
            _prev_grp = group_by[i0]
    org_idx = np.zeros(group_by.size, dtype=np.int)
    org_idx[_ord] = np.asarray(xrange(group_by.size))
    return _cs1[org_idx]

def get_silde_sum(df,end_day,cols,cname):
    df=df.loc[(df.day>=end_day-4)&(df.day<end_day),cols+['is_trade']]
    tb=df.groupby(cols,as_index=False)['is_trade'].agg({cname:np.sum})
    return tb
	
def get_silde_cnt(df,end_day,cols,cname):
    df=df.loc[(df.day>=end_day-4)&(df.day<end_day),cols+['is_trade']]
    tb=df.groupby(cols,as_index=False)['is_trade'].agg({cname:'count'})
    return tb
