#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from datetime import date
import holidays
from datetime import datetime
from sklearn.model_selection import train_test_split
import h5py
import hdfdict
import os
import ast
import hickle as hkl ###hdf5 version of pickle
import ast
import deepdish as dd
import pickle
import json

from tensorflow import keras
import tensorflow as tf


# In[6]:



# In[7]:


global intlist
global floatist
intlist = ['int32', 'int64']
floatist = ['float32', 'float64']


##apply functions
def holiday_fun(col, holiday):
    '''
    check if date is a holiday or not
    '''
    return col.date() in holiday
    
def get_holiday_name(col, holiday):
    '''
    gets the name of the holiday
    '''
    return holiday.get(col.date())


# In[10]:


def date_expansion(df, datecol, location, year=True, month=True, day_of_month=True, 
                day_of_week=True, is_holiday=True, hour=True, minutes=True, seconds=True, drop = True):
    
    '''
    df : data frame
    datecol : date column from dataframe to do the expansion on
    location : is a list that may contain the country and provonic and state if needed
    
    returns new columns (year, month, day, is_holiday...etc)
    '''
    df[datecol] = pd.to_datetime(df[datecol])
    if df[datecol].dtype != '<M8[ns]' or df[datecol].dtype != 'datetime64[ns]':
            raise Exception("only one of the columns can be True at a time")
    
    
    
    if year:
        df[f'{datecol}_year'] = df[datecol].dt.year.astype('category')
    if month: 
        df[f'{datecol}_month'] = df[datecol].dt.month.astype('category')
    if day_of_month:
        df[f'{datecol}_day_of_month'] = df[datecol].dt.day.astype('category')
    if day_of_week:
        df[f'{datecol}_day_of_week'] = df[datecol].dt.dayofweek.astype('category')
    if hour:
        df[f'{datecol}_hour']=df[datecol].dt.hour.astype('category')
    if minutes:
        df[f'{datecol}_minutes'] = df[datecol].dt.minute.astype('category')
    if seconds:
        df[f'{datecol}_seconds'] = df[datecol].dt.second.astype('category')
        
    if is_holiday:
        HD = holidays.CountryHoliday(location[0], prov=None, state=None)## HD is short for holiday
        df[f'{datecol}_is_holiday'] = df[datecol].apply(lambda x: holiday_fun(x, HD))
        df[f'{datecol}_is_holiday'] = df[f'{datecol}_is_holiday'].astype('O')## using holiday library to check if each date in column is holiday
    #df['name_of_holiday'] = df[datecol].apply(lambda x: get_holiday_name(x, HD))## if holidy, retrieve the name of holiday
    
    if drop:
        df.drop(datecol, axis=1, inplace=True)
    
    return df


# In[ ]:





# In[11]:


########## 
date_param = {'year': True, 'month' :True, 'day_of_month' :True
             ,'day_of_week' : True, 'is_holiday' :True,
             'hour' : True, 'minute' : True, 'seconds' : True }
location= ['US']




# In[196]:


#task6 
### creating the meta dataframe


def create_meta_df(df):
    listofcols = df.columns.to_list()
    meta_df = pd.DataFrame({'col_name' : listofcols})
    meta_df['dtype'] = df.dtypes.to_list()
    meta_df['to_normalize'] = meta_df['dtype'].apply(lambda x: True if x in intlist or x in floatist else False)
    meta_df['one_hot_encode'] =False
    meta_df['label_encode'] =  meta_df['dtype'].apply(lambda x: True if ((x == 'object') or (str(x) == 'category')) else False)
    meta_df['to_drop'] = False
    meta_df['NA_policy'] = 'mean'
    place_holder = (df.apply(lambda x: len(x.unique()))).to_list()
    meta_df['unique'] = place_holder
    meta_df['cat_embed'] = False
    meta_df = meta_df[meta_df.col_name != 'year'] ## we will drop the row where col_name = year
    meta_df.to_csv('meta_datafrma.csv')
    return meta_df

# In[198]:




# In[183]:


def cat_onehot(train_df, test_df, valid_df, col):
    '''
    takes categorical columns 
    returns OneHotEncoded array of the Column, sklearn object
    
    '''
    encoder = OneHotEncoder(handle_unknown='ignore')
    train = train_df[col].values.astype(str).reshape(-1, 1)
    test = test_df[col].values.astype(str).reshape(-1, 1)
    valid = valid_df[col].values.astype(str).reshape(-1, 1)
    
    if train_df[col].dtype == 'object' or train_df[col].dtype == 'category':
        train = encoder.fit_transform(train)
        test = encoder.transform(test)
        test = test.reshape(-1, 1)
        
        valid = encoder.transform(valid)
        
        return encoder, train, test, valid




def norm_num_col(train_df, test_df, valid_df, col):
    '''
    df: dataframe
    numcol: name of columns to be normalized(numerical columns)
    returns: normalized array, scaler (sklearn) object
    '''
    train = train_df[col].values.reshape(-1, 1)
    test = test_df[col].values.reshape(-1, 1)
    valid = valid_df[col].values.reshape(-1, 1)
    scaler = StandardScaler()
    if train_df[col].dtype in intlist or train_df[col].dtype in floatist:
        train = scaler.fit_transform(train)

        
        test  = scaler.transform(test)

        
        valid  = scaler.transform(valid)
 
        
        return scaler, train, test, valid
    else: 
        raise Exception("Non numeric column type was passed: " + col)

    
def norm_num_col_minmax(train_df, test_df, valid_df, col):
    '''
    df: dataframe
    numcol: name of columns to be normalized(numerical columns)
    returns: normalized array, scaler (sklearn) object
    '''
    train = train_df[col].values.reshape(-1, 1)
    test = test_df[col].values.reshape(-1, 1)
    valid = valid_df[col].values.reshape(-1, 1)    
    scaler = MinMaxScaler()
    if train_df[col].dtype == 'int' or train_df[col].dtype == 'float':

 
        train = scaler.fit_transform(train)

        
        test  = scaler.transform(test)

        
        valid  = scaler.transform(valid)

        
        return scaler, train, test, valid
    else: 
        raise Exception("Non numeric column type was passed: " + col)


# In[209]:


def label_cat_encoding(train_df, test_df, valid_df, col):
    '''
    train_df: dataframe
    catcol: name of catergorical column which will be label encoded
    
    returns label encoded array, sklearn object
    '''
    encoder = LabelEncoder()
    train = train_df[col].values.astype(str)
    test = test_df[col].values.astype(str)
    valid = valid_df[col].values.astype(str)
    
    if (train_df[col].dtype == 'object') or (train_df[col].dtype == 'category'):
        train = encoder.fit_transform(train)
        
        test = encoder.transform(test)
        
        valid = encoder.transform(valid)
        
        return encoder, train, test, valid

#task5 

def missing_value(df, col, strat, value):
    '''
    datafram 
    col: column name
    
    start: is the name of the strategy to fill the nan values
        if strategy == "value"
        user will pass value to the the parameter value which will fill the Nan values

    incase the column is catigorical column, the messing value will be filled with the most frequent value automatically
    
    returns updated datafram
    
    '''

    if df[col].dtype == 'int64' or df[col].dtype == 'float64':
    
        if strat == 'drop_na':
            df[col].dropna(inplcae = True)
            return df
        if strat == 'mean':
            df[col].fillna(df[col].mean(), inplace = True)
            return df
        if strat == 'median':
            df[col].fillna(df[col].median(), inplace = True)
            return df
        if strat == 'value':
            df[col].fillna(value, inplace = True)
            return df

    else:
        if strat == 'drop_na':
            df[col].dropna(inplcae = True)
            return df
        else: 
            df[col].fillna(value=df[col].value_counts().idxmax())
            return df


# In[171]:


def drop_column(df, col):
    '''
    drops column from data frame
    returns updated dataframe
    '''
    df.drop([col], axis = 1, inplace = True)
    
    return df


# In[172]:


def train_test_valid_split(df, train_r, test_r, validation_r):
    '''
    dic: a dictionary that contains the transformed values of the columns
    train_r: train ratio split
    test_r: test ratio split 
    validation_r: validation ratio split
    
    returns: 3 different dics each one has column name and values
    '''
    dic = df.to_dict('series')
    
    train_dic = {}
    test_dic = {}
    valid_dic = {}
    Trans_dic = {}
    
    for col in dic.keys():
        L = len(dic[col]) ## the length of values from each transformed column, will be used for the split
        tr_r = int(L*train_r) ## the int value of the split ration to the trainig set
        ts_r = int(L*test_r)
        v_r = int(L*validation_r)
        
        train, test, validate =  dic[col][0:tr_r] , dic[col][tr_r:ts_r+tr_r], dic[col][ts_r+tr_r:v_r+ts_r+tr_r]
        
        
        train_dic.update({col : train})
        test_dic.update({col : test})
        valid_dic.update({col : validate})
        
    train_dic.update({'type' : 'training set'})
    test_dic.update({'type' : 'testing set'})
    valid_dic.update({'type' : 'validation set'})
    
    
    return train_dic, test_dic, valid_dic


# In[292]:


def transformation_fun(train_dict, test_dict, valid_dict, meta, cols):
    '''
    df: dataframe
    meta: dataframe contains information about the transformations 
    
    the function will do transformation on the dataset splits
    
    returns transformer and tranform object 
    '''
    columns = list(train_dict.keys())
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df_valid = pd.DataFrame()
    for col in columns:
        df_train[col] = train_dict[col]
        df_test[col] = test_dict[col]
        df_valid[col] = valid_dict[col]
           
    
    small_dic = {}
    big_dic = {}
    
    try:
        meta.set_index('col_name', inplace = True)
    except: 
        pass

    for i in range(len(meta)):
        
        if meta['to_normalize'][i]:
            if meta ['one_hot_encode'][i] or meta['label_encode'][i]:
                  raise Exception("only one of the columns can be True at a time")
        if meta ['one_hot_encode'][i]:
            if meta ['to_normalize'][i] or meta['label_encode'][i]:
                    raise Exception("only one of the columns can be True at a time")
        if meta['label_encode'][i]:
            if meta['to_normalize'][i] or meta ['one_hot_encode'][i]:
                    raise Exception("only one of the columns can be True at a time")
                    
    for col in cols:
        print(col)
        ###filling NAN will run first, it return DF that will be used for transornations 
        ###looping for each on of the rows for each col_name to perform transformations
       
        if meta.loc[col,'to_drop']: ###drop column from origianl dataframe, performed first of operations
            df_train = drop_column(df_train, col)
            df_test = drop_column(df_train, col)
            df_test = drop_column(df_test, col)
    
        NAN_free_df_train = missing_value(df_train, col, meta.loc[col,'NA_policy'], 0)
        NAN_free_df_test = missing_value(df_test, col, meta.loc[col,'NA_policy'], 0)
        NAN_free_df_valid = missing_value(df_valid, col, meta.loc[col,'NA_policy'], 0)
        
        if meta.loc[col,'to_normalize']:
            obj_, data_train, data_test, data_valid = norm_num_col(NAN_free_df_train,NAN_free_df_test, NAN_free_df_valid , col)
            small_dic = {col: {'train' : data_train, 'test' : data_test, 'valid' : data_valid, 'transformers' : obj_}}
            big_dic.update(small_dic)
            
        if meta.loc[col,'one_hot_encode']:
            obj_, data_train, data_test, data_valid = cat_onehot(NAN_free_df_train,NAN_free_df_test, NAN_free_df_valid , col)
            small_dic = {col: {'train' : data_train, 'test' : data_test, 'valid' : data_valid, 'transformers' : obj_}}
            big_dic.update(small_dic)
        
        if meta.loc[col,'label_encode']:
            obj_, data_train, data_test, data_valid = label_cat_encoding(NAN_free_df_train,NAN_free_df_test, NAN_free_df_valid , col)
            small_dic = {col: {'train' : data_train, 'test' : data_test, 'valid' : data_valid, 'transformers' : obj_}}
            big_dic.update(small_dic)
            
            
    try:
        del big_dic['type'] 
    except:
        pass
    
    
    print(cols)
    print(big_dic.keys())
    trans_dic = {}
    for col in (cols):
        trans_dic.update({col : big_dic[col]['transformers']})
    for col in list(big_dic.keys()):
        del big_dic[col]['transformers']
    
    return big_dic, trans_dic


# In[293]:


def train_test_split_target(df,mode, target ,train_r, test_r, validation_r):
    '''
    df: pandas dataframe
    mode: boolean if True the column will be label encoded
    
    train_r, test_r, validation_r: float variables indicates the percentage of each set
    
    returns 3 dicts (train, test, valid)
    '''
    
    train_dic = {}
    test_dic = {}
    valid_dic = {}

    
    L = len(df[target]) ## the length of values from each transformed column, will be used for the split
    tr_r = int(L*train_r) ## the int value of the split ration to the trainig set
    ts_r = int(L*test_r)
    v_r = int(L*validation_r)
    
    if mode:
        encoder = LabelEncoder()
        data = df[target].astype(str)
        data = encoder.fit_transform()
        df[target] = data

         
    train, test, validate =  df[target][0:tr_r] , df[target][tr_r:ts_r+tr_r], df[target][ts_r+tr_r:v_r+ts_r+tr_r]
            
    train_dic.update({target : np.array(train).reshape(-1,1)})

    test_dic.update({target : np.array(test).reshape(-1,1)})
    
    valid_dic.update({target : np.array(validate).reshape(-1,1)})
    
    if mode:
        return train_dic, test_dic, valid_dic, encoder
    
    return train_dic, test_dic, valid_dic


# In[294]:


def hdf5(data_dict, trans_dict ,meta_df, name, path , target_train, target_test, target_val):
    '''
    this function will store meta data about the 3 dict using h5py
    
    to store the meta data into an h5py format "deepdish" libraary is used,
    it stores the data in h5py format in a separet file.
    it pickles the transormers of the data and store it
    
    returns the same dicts
    

    '''
    ###############seprarating the dicts 
    train_dic = {}
    test_dic = {}
    valid_dic = {}
    for col in list(data_dict.keys()):
        train_dic.update({col:data_dict[col]['train']})
        test_dic.update({col:data_dict[col]['test']})
        valid_dic.update({col:data_dict[col]['valid']})
    
    ############### reindexing the meta dataframe   
    ############### a dict with column names as keys and datatype as values

    try:
        meta_df.reset_index(level=0, inplace=True)
    except:
        pass 

    metadata_dict = {}
    for col, typ in zip(meta_df['col_name'], meta_df['dtype']):
            metadata_dict.update({col : typ})
    
    
    try:
        del train_dic['type'] 
        del test_dic['type'] 
        del valid_dic['type'] 
    except:
        pass
    

    ################ droping the transformer object from training dict and creating a dict of transformer objs
    ################ creating folders
    fullname = os.path.join(path, name)  
    fullname2 = os.path.join(path, 'transformers')
    
    if not os.path.exists(fullname):
        os.makedirs(fullname)
        
    if not os.path.exists(fullname2):
        os.makedirs(fullname2)
    ################ craeting h5py file
   
    for k in train_dic.keys():
        try:
            del train_dic[k][1]
            del test_dic[k][1]
            del valid_dic[k][1]
        except:
            pass
    #with h5py.File(fullname+'/metaDataset.h5', 'w') as hf:
    #   dd.io.save(fullname+'/metaDataset.h5' ,metadata_dict)###storing metadata information 
       
    with h5py.File(fullname+'/dataset_h5.h5', 'w') as f:
        
        train_grp=f.create_group('train')
        test_grp=f.create_group('test')
        valid_grp=f.create_group('validation')
        meta_grp=f.create_group('meta')
        
        for k,v in train_dic.items():
        
            train_grp.create_dataset(k,data=v)###creating the dataset for train test valid
                
        for k,v in test_dic.items():
        
            test_grp.create_dataset(k,data=v)   
        
        for k,v in valid_dic.items():
      
            valid_grp.create_dataset(k,data=v)  
        
        
        for k,v in metadata_dict.items():
            meta_grp.create_dataset(k,data=str(v))
            
        #############writing the target    
        target_train_grp=f.create_group('target_train')
        target_test_grp=f.create_group('target_test')
        target_valid_grp=f.create_group('target_valid')
        
        for k,v in target_train.items():
            target_train_grp.create_dataset(k,data=v)###creating the dataset for train test valid
                
        for k,v in target_test.items():
            target_test_grp.create_dataset(k,data=v)   
        
        for k,v in target_val.items():
            target_valid_grp.create_dataset(k,data=v)  
            

        f.close()
    
    
    print("to access the H5 file ['meta', 'target_test', 'target_train', 'target_valid', 'test', 'train', 'validation'] are the keys")  
    ############### saving the transformers as a pickle format
    
    with open(fullname2 + '/trnasformer.pickle', 'wb') as handle:
        pickle.dump(trans_dict, handle)


    return train_dic, test_dic, valid_dic, trans_dict


###task8 
def keras_dataloader_all_old(num_cols, cat_cols,target_col, name, path, batch_size, shuffle):
    '''
    df: original dataframe
    name: name of the H5PY file 
    path: patht of the H5PY file
    num_cols : list of numerical column names
    cat_cols : list of categorical column names
    
    this function will read from the H5 file that contain the data, create the X, Y splits,
    batch the data and return the dataloader
    '''
    k = ['test', 'train', 'validation']
    k2 = ['target_test', 'target_train', 'target_valid']# list of keys to access the h5 dataset
    input_dic_train = {}
    input_dic_test = {}
    input_dic_valid = {}
    
    train_array = []
    test_array = []
    valid_array = []
    
    train_array_cat = []
    test_array_cat = []
    valid_array_cat = []
    
    
    target_train_dict= {}
    target_test_dict= {}
    target_val_dict= {}
    
    
    with h5py.File(path+name+'/dataset_h5.h5', "r") as f:
       
        for col in num_cols:
      
                    train_array.append(f[k[1]][col][()])                                   
                    train_array_f = np.hstack(train_array)
                    
                    test_array.append(f[k[0]][col][()])
                    test_array_f = np.hstack(test_array)
                    
                    valid_array.append(f[k[2]][col][()]) 
                    valid_array_f = np.hstack(valid_array)

        
        for col in cat_cols:

                    train_array_cat.append(f[k[1]][col][()])
                    input_dic_train.update({col : np.array(f[k[1]][col][()].reshape(-1,1))})

                    test_array_cat.append(f[k[0]][col][()]) 
                    input_dic_test.update({col : np.array(f[k[0]][col][()].reshape(-1,1))})

                    valid_array_cat.append(f[k[2]][col][()])   
                    input_dic_valid.update({col : np.array(f[k[2]][col][()].reshape(-1,1))})
                    
        if target_col:
            data = f[k2[1]][target_col][()]
            target_train_dict.update({target_col : np.array(data)})
            
            data = f[k2[0]][target_col][()]
            target_test_dict.update({target_col : np.array(data)})
            
            data = f[k2[2]][target_col][()]
            target_val_dict.update({target_col : np.array(data)})
        
  
    
    input_dic_train.update({'numeric_cols' : train_array_f})
    input_dic_test.update({"numeric_cols" : test_array_f})
    input_dic_valid.update({"numeric_cols" : valid_array_F})

    
    big_dict = {"training_data" : input_dic_train, "testing_data" : input_dic_test, "validation_data" : input_dic_valid}
    labels = {'training_label' :target_train_dict , 'testing_labels' : target_test_dict, 'validation_labels' : target_val_dict}

    
    ## creating the dataloader
    train_dataset= tf.data.Dataset.from_tensor_slices(input_dic_train)
    label_dataset= tf.data.Dataset.from_tensor_slices(target_train_dict)
    dataset = tf.data.Dataset.zip((train_dataset, label_dataset)).batch(batch_size).shuffle(shuffle)
 
    test_dataset= tf.data.Dataset.from_tensor_slices(input_dic_test)
    valid_dataset=  tf.data.Dataset.from_tensor_slices(input_dic_valid)
    
    return dataset, test_dataset, valid_dataset,big_dict,labels


def get_data_from_h5(num_cols, cat_cols,target_col, name, path):
    '''
    df: original dataframe
    name: name of the H5PY file 
    path: patht of the H5PY file
    num_cols : list of numerical column names
    cat_cols : list of categorical column names
    
    this function will read from the H5 file that contain the data, create the X, Y splits,
    batch the data and return the dataloader
    '''
    k = ['test', 'train', 'validation']
    k2 = ['target_test', 'target_train', 'target_valid']# list of keys to access the h5 dataset
    input_dic_train = {}
    input_dic_test = {}
    input_dic_valid = {}
    
    train_array = []
    test_array = []
    valid_array = []
    
    train_array_cat = []
    test_array_cat = []
    valid_array_cat = []
    
    
    target_train_dict= {}
    target_test_dict= {}
    target_val_dict= {}
    
    
    with h5py.File(path+name+'/dataset_h5.h5', "r") as f:
        for col in num_cols:
            train_array.append(f[k[1]][col][()])                                   
            train_array_f = np.hstack(train_array)

            test_array.append(f[k[0]][col][()])
            test_array_f = np.hstack(test_array)

            valid_array.append(f[k[2]][col][()]) 
            valid_array_f = np.hstack(valid_array) 

        
        for col in cat_cols:
            train_array_cat.append(f[k[1]][col][()])
            input_dic_train.update({col : np.array(f[k[1]][col][()].reshape(-1,1))})

            test_array_cat.append(f[k[0]][col][()]) 
            input_dic_test.update({col : np.array(f[k[0]][col][()].reshape(-1,1))})

            valid_array_cat.append(f[k[2]][col][()])   
            input_dic_valid.update({col : np.array(f[k[2]][col][()].reshape(-1,1))})
                    
        if target_col:
            data = f[k2[1]][target_col][()]
            target_train_dict.update({target_col : np.array(data)})
            
            data = f[k2[0]][target_col][()]
            target_test_dict.update({target_col : np.array(data)})
            
            data = f[k2[2]][target_col][()]
            target_val_dict.update({target_col : np.array(data)})
        
  
                    

    
    input_dic_train.update({'numeric_cols' : train_array_f})
    input_dic_test.update({"numeric_cols" : test_array_f})
    input_dic_valid.update({"numeric_cols" : valid_array_f})
    
    big_dict = {"training_data" : input_dic_train, "testing_data" : input_dic_test, "validation_data" : input_dic_valid}
    labels = {'training_label' :target_train_dict , 'testing_labels' : target_test_dict, 'validation_labels' : target_val_dict}
    
    return big_dict, labels


def create_keras_dataloader_tabular(input_dic_train, target_train_dict, batch_size, window_size, prefetch_size, shuffle, 
                                    numeric_cols_key_name='numeric_cols'):
    train_dataset= tf.data.Dataset.from_tensor_slices(input_dic_train)
    label_dataset= tf.data.Dataset.from_tensor_slices(target_train_dict)
    dataset = tf.data.Dataset.zip((train_dataset, label_dataset)).batch(batch_size).shuffle(shuffle)
    
    return dataset


def create_tabular_dl_from_h5(num_cols, cat_cols,target_col, name, path, batch_size, window_size, prefetch_size, shuffle):
    bd, ls = get_data_from_h5(num_cols, cat_cols,target_col, name, path)
    
    train_data_dict = bd['training_data']
    train_y_dict = ls['training_label']
    
    val_data_dict   = bd['validation_data']
    val_y_dict = ls['training_label']
    
    test_data_dict  = bd['testing_data']
    test_y_dict = ls['training_label']
    
    train_dl = create_keras_dataloader_tabular(train_data_dict, train_y_dict, batch_size,0, 0 ,shuffle)
    val_dl = create_keras_dataloader_tabular(val_data_dict, val_y_dict, batch_size,0, 0 , shuffle)
    test_dl = create_keras_dataloader_tabular(test_data_dict, test_y_dict, batch_size,0, 0 , shuffle)
    
    return train_dl, val_dl, test_dl

def create_keras_dataloader_tseries(input_dic_train, target_train_dict, batch_size, window_size, prefetch_size, shuffle, 
                                    numeric_cols_key_name='numeric_cols'):
    train_datasets_dict= {k: tf.data.Dataset.from_tensor_slices(v) for k, v in input_dic_train.items() }
    train_datasets_dict= {k: v.window(window_size, shift=1, drop_remainder=True) for k, v in train_datasets_dict.items() }
    train_datasets_dict= {k: v.flat_map(lambda w: w.batch(window_size) ) for k, v in train_datasets_dict.items() }
    
    label_dataset= tf.data.Dataset.from_tensor_slices(target_train_dict)
    label_dataset = label_dataset.window(window_size, shift=1, drop_remainder=True)
    label_dataset = label_dataset.flat_map(lambda w: w.batch(window_size) )
    
    dataset = tf.data.Dataset.zip((train_datasets_dict, label_dataset))
    dataset = dataset.batch(batch_size, drop_remainder=True).shuffle(shuffle).prefetch(prefetch_size)
    
    return dataset

def create_tseries_dl_from_h5(num_cols, cat_cols,target_col, name, path, batch_size, window_size, prefetch_size, shuffle):
    bd, ls = get_data_from_h5(num_cols, cat_cols,target_col, name, path)
    
    print(bd)
    
    train_data_dict = bd['training_data']
    train_y_dict = ls['training_label']
    
    val_data_dict   = bd['validation_data']
    val_y_dict = ls['training_label']
    
    test_data_dict  = bd['testing_data']
    test_y_dict = ls['training_label']
    
    train_dl = create_keras_dataloader_tseries(train_data_dict, 
                                               train_y_dict[target_col], 
                                               batch_size, 
                                               window_size, 
                                               prefetch_size, 
                                               shuffle)
    
    val_dl = create_keras_dataloader_tseries(val_data_dict, 
                                             val_y_dict[target_col], 
                                             batch_size, window_size, 
                                             prefetch_size, 
                                             shuffle)
    
    test_dl = create_keras_dataloader_tseries(test_data_dict, 
                                              test_y_dict[target_col], 
                                              batch_size, 
                                              window_size, 
                                              prefetch_size, 
                                              shuffle)
    
    return train_dl, val_dl, test_dl











