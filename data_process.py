import pandas as pd
import numpy as np
import gc
from utils import *
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()

def data_process0(df):
    print "encode cate id now"
    aa=df['item_category_list'].str.split(';',expand=True)
    aa=aa.rename(columns={0:'cate0',1:'cate1',2:'cate2'})
    df=df.join(aa)

    df['cate0']=df['cate0'].fillna('0')
    df['cate1']=df['cate1'].fillna('0')
    df['cate2']=df['cate2'].fillna('0')

    df['cate_id']=np.add(df['cate0'].values, df['cate1'].values)
    df['cate_id']=np.add(df['cate_id'].values, df['cate2'].values)

    del df['item_category_list']
    del aa
    gc.collect()
    
    return df
	
def data_process1(df):
    print "fill nan now"
    np.random.seed(2018)
    for col in ['item_brand_id','item_city_id','user_age_level','user_occupation_id','user_star_level','user_gender_id']:
        filter1=df[col]!=-1
        filter2=df[col]==-1
        a=df.loc[filter1,col]

        fill1=np.random.randint(0,a.shape[0]-1,df.loc[filter2,col].shape[0])
        df.loc[filter2,col]=a.iloc[fill1].values
        
    for col in ['shop_score_service','shop_score_delivery','shop_score_description']:
        df.loc[df[col]==-1,col]=None
        
    del fill1
    del a
    gc.collect()
    
    return df
	
def data_process2(df):
    print "encode now"
    df['item_id'] = lbl.fit_transform(df['item_id'])
    df['user_id'] = lbl.fit_transform(df['user_id'])
    df['shop_id'] = lbl.fit_transform(df['shop_id'])
    df['cate_id'] = lbl.fit_transform(df['cate_id'])
    df['item_city_id'] = lbl.fit_transform(df['item_city_id'])
    df['item_brand_id'] = lbl.fit_transform(df['item_brand_id'])
    return df
	
def data_process3(df):
    print "decode time"
    df.loc[:,'context_datetime'] = pd.to_datetime(df['context_timestamp']+28800,unit='s')
    df.loc[:,'day']=df['context_datetime'].apply(get_day)
    df.loc[:,'hour']=df['context_datetime'].apply(get_hour)
    df.loc[df.day==31,'day']=0
    del df['context_datetime']
    gc.collect()
    return df
	
def data_process4(df):
    print "cate_match"
    pre_cate_tb=df['predict_category_property'].str.split(';',expand=True)[[0,1,2,3,4]]
    pre_cate_tb=pre_cate_tb.applymap(lambda x:str(x).split(':')[0])
    pre_cate_tb=pre_cate_tb.rename(columns={0:'pre0',1:'pre1',2:'pre2',3:'pre3',4:'pre4'})

    df=df.join(pre_cate_tb)
    df.loc[:,'cate_match']=-1
    a=(df['cate0']==df['pre0'])
    b=(df['cate1']==df['pre0'])
    c=(df['cate2']==df['pre0'])
    d1=a|b|c
    df.loc[d1,'cate_match']=0

    a=(df['cate0']==df['pre1'])
    b=(df['cate1']==df['pre1'])
    c=(df['cate2']==df['pre1'])
    d2=(a|b|c)&(~d1)
    df.loc[d2,'cate_match']=1

    a=(df['cate0']==df['pre2'])
    b=(df['cate1']==df['pre2'])
    c=(df['cate2']==df['pre2'])
    d3=(a|b|c)&(~(d1|d2))
    df.loc[d3,'cate_match']=2

    d4=(~(d1|d2|d3))
    df.loc[d4,'cate_match']=3
    del pre_cate_tb
    gc.collect()
    
    return df
	
def data_process5(df):
    print "tongji"
    df = get_cumcount(df, cols=['user_id', 'shop_id', 'day'], cname='user_shop_cumcount')
    df = get_cumcount(df, cols=['user_id', 'item_id', 'day'], cname='user_item_cumcount')
    df = get_nunique(df, cols=['shop_id'], cname='shop_item_nunique', value='item_id')
    
    df['user_day']=np.add(df.user_id.astype(str).values,df.day.astype(str).values)
    df['user_cate']=np.add(df.user_id.astype(str).values,df.cate_id.astype(str).values)
    df['pre_cate_id'] = lbl.fit_transform(df['predict_category_property'])
    df['pre_cate_item']=np.add(df.pre_cate_id.astype(str).values , df.item_id.astype(str).values)
    
    
    df['user_price_rank']=get_rank(df.item_price_level.values , df.user_day.values)
    df['temp']=-df['item_sales_level']
    df['user_sale_rank']=get_rank(df.temp.values , df.user_day.values)
    df['temp']=-df['item_collected_level']
    df['user_collect_rank']=get_rank(df.temp.values , df.user_day.values)
    
    df['pre_cate_price_rank']=get_rank(df.item_price_level.values,df.pre_cate_id.values)
    df['temp']=-df['item_sales_level']
    df['pre_cate_sale_rank']=get_rank(df.temp.values,df.pre_cate_id.values)
    df['temp']=-df['item_collected_level']
    df['pre_cate_collect_rank']=get_rank(df.temp.values,df.pre_cate_id.values)
    
    del df['temp']
    return df

def data_process6(df):
    print "tongji2"
    df = get_cumcount(df, cols=['user_id', 'shop_id', 'day'], cname='user_shop_cumcount')
    df = get_cumcount(df, cols=['user_id', 'item_id', 'day'], cname='user_item_cumcount')
    df = get_nunique(df, cols=['shop_id'], cname='shop_item_nunique', value='item_id')
    
    df['user_day']=np.add(df.user_id.astype(str).values,df.day.astype(str).values)
    df['user_cate']=np.add(df.user_id.astype(str).values,df.cate_id.astype(str).values)
    df['pre_cate_id'] = lbl.fit_transform(df['predict_category_property'])
    df['pre_cate_item']=np.add(df.pre_cate_id.astype(str).values , df.item_id.astype(str).values)
    
    df['temp']=-df['item_price_level']
    df['user_price_rank']=get_rank(df.temp.values , df.user_day.values)
    df['user_sale_rank']=get_rank(df.item_sales_level.values , df.user_day.values)
    df['user_collect_rank']=get_rank(df.item_collected_level.values , df.user_day.values)
    
    df['temp']=-df['item_price_level']
    df['pre_cate_price_rank']=get_rank(df.temp.values,df.pre_cate_id.values)
    df['pre_cate_sale_rank']=get_rank(df.item_sales_level.values,df.pre_cate_id.values)
    df['pre_cate_collect_rank']=get_rank(df.item_collected_level.values,df.pre_cate_id.values)
    
    del df['temp']
    return df