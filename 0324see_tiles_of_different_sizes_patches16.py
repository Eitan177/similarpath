import re
import h5py
import streamlit as st
from streamlit import caching
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from glob import glob
import urllib.request
import tempfile
from PIL import Image
import pdb
import tables

import pickle as pkl

import functools

import os
import gc
import altair as alt
import pickle
import base64
from tensorflow.keras.models import model_from_json
import types
from keras_pickle_wrapper import KerasPickleWrapper
from io import BytesIO



encoder_input_shape = (224, 224, 3)


tile_encoder = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', pooling='max', input_shape=encoder_input_shape)
tile_encoded_shape = tuple(tile_encoder.outputs[0].shape[1:])


def make_keras_picklable():

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            tf.keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
   
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            
            fd.write(state['model_str'])
            
            fd.flush()
             
            model = tf.keras.models.load_model(fd.name)
            
        self.__dict__ = model.__dict__


    cls = tf.keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

def load_saved_mod(modelpref):
    # load json and create model
    json_file = open(modelpref+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(modelpref+'.h5')
    print("Loaded model from disk")
    return loaded_model

@st.cache(allow_output_mutation=True)
def mk_classif(startover):
  classifier = []
  print('made classifer')
  return classifier

@st.cache(allow_output_mutation=True)
def get_tilelevel():
    print('tile run')
    return []
@st.cache(allow_output_mutation=True)
def get_tile():
    print('tile run')
    return []


@st.cache(allow_output_mutation=True)
def get_data():
    print('run')
    return []




@st.cache(allow_output_mutation=True,hash_funcs={h5py._hl.dataset.Dataset: id, h5py._hl.files.File: id})
def datain(tilesize,levels): #tcga_location_2240,file_2240,tcga_location_896,file_896):
    print('datain')
    
    location_use=[]
    file_use=[]
    maglevel=np.where(levels==tilesize)[0][0]

    url=['https://www.dl.dropboxusercontent.com/s/2gi9pvevxvxivet/location_12files_truncate_pkl40.hd5?dl=0',\
    'https://www.dl.dropboxusercontent.com/s/ocodemyp0se849c/location_12files_truncate_pkl41.hd5?dl=0',\
    'https://www.dl.dropboxusercontent.com/s/68924qw50zzrj67/location_12files_truncate_pkl42.hd5?dl=0',\
    'https://www.dl.dropboxusercontent.com/s/lkwjkfcpvj3l1rh/location_12files_truncate_pkl43.hd5?dl=0',\
    'https://www.dl.dropboxusercontent.com/s/dhur5yu33sc36t1/location_12files_truncate_pkl44.hd5?dl=0',\
    'https://www.dl.dropboxusercontent.com/s/cahn221jqczyycx/location_12files_truncate_pkl45.hd5?dl=0',\
    'https://www.dl.dropboxusercontent.com/s/m24ssdft4zfgnvb/location_12files_truncate_pkl46.hd5?dl=0',\
    'https://www.dl.dropboxusercontent.com/s/b9u0s8qmfvz5n02/location_12files_truncate_pkl47.hd5?dl=0',\
    'https://www.dl.dropboxusercontent.com/s/0yc5cyef9rdngcn/location_12files_truncate_pkl48.hd5?dl=0',\
    'https://www.dl.dropboxusercontent.com/s/l5gt8c8zq4q4mj6/location_12files_truncate_pkl49.hd5?dl=0',\
    'https://www.dl.dropboxusercontent.com/s/ufl54i6kak5qm2z/location_12files_truncate_pkl410.hd5?dl=0',\
    'https://www.dl.dropboxusercontent.com/s/tpgsjdtee6dksbm/location_12files_truncate_pkl411.hd5?dl=0']
    
    locf='location_12files_truncate_pkl4'+str(maglevel)+'.hd5'
   
    if len(glob(locf))==0:
        urllib.request.urlretrieve(url[maglevel],locf)
  
    #location_use=pd.read_hdf('locations_12files.hd5','tcga_location'+str(maglevel))
    location_use=pd.read_hdf('location_12files_truncate_pkl4'+str(maglevel)+'.hd5','tcga_location'+str(maglevel))
    
    
    cancers_tcga=['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA',\
       'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD',\
       'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC',\
       'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

    return(location_use,cancers_tcga)



@st.cache(allow_output_mutation=True, hash_funcs={h5py._hl.dataset.Dataset: id})
def get_():
    tables.file._open_files.close_all()

    #model_2240=load_saved_mod('1214model_2240_16max')
    #model_896=load_saved_mod('1204model_896_16max')


    levels=[896,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]#,1344,1792,2240,2688,3136,3584,4032,4480,4928,5376,6000]   
    levels=np.unique(np.sort(levels))

    model_level=load_saved_mod('0212mag_model_16max') #   load_saved_mod('0122model_22sizes_16max')  


    tcga=pd.read_csv('tcga_annot.tsv',sep='\t')


    
    model_diagnosis=load_saved_mod("0212diagnos_model_16max")
    #file_pred_d=h5py.File('diag_preds.h5','r+')
    #file_pred_d33=h5py.File('diag33pred.h5','r+')
    levels_onehot=tf.one_hot(np.arange(0,12),12).numpy()
    #file_pred=[]
  
    
    #file_predictions_ = h5py.File('location_tables_and_predictions.hd5','r+')


    gd_=pd.read_csv('gdc_manifest_svsname_to_filename.txt',sep='\t')

    #for mlevel in np.arange(0,12):  
    #    l_use=pd.read_hdf('location_12files_truncate_pkl4'+str(mlevel)+'.hd5','tcga_location'+str(mlevel))

    #    l_use.to_hdf('location_12files_truncate_pkl4.hd5',key='tcga_location'+str(mlevel))
         
    #pdb.set_trace()
        
        


    return tcga,model_level,model_diagnosis, levels, gd_


@st.cache(allow_output_mutation=True)
def rf_start_over(category_predicted_label,ind_rando_tile):
    for_rf_tiles=[]
    name_and_loc=[]
    for_rf_ind=[0]

    
    print('made rf files')

    return for_rf_tiles, name_and_loc,for_rf_ind
    
@st.cache(allow_output_mutation=True,hash_funcs={h5py._hl.dataset.Dataset: lambda _: None, h5py._hl.files.File: lambda _: None})
def euclidian(categories,rando_prediction):
    eucs=np.linalg.norm(np.array(categories)-rando_prediction,axis=1)    
    return eucs

@st.cache(allow_output_mutation=True,hash_funcs={h5py._hl.dataset.Dataset: lambda _: None, h5py._hl.files.File: lambda _: None})
def category_tiles_random(file_uploaded_tile,rando_prediction,tcga,location_use,categories_predict33,images_read_in,sortbydistance,gd_):
    shuf=np.arange(0,500)
    eucs=euclidian(categories_predict33,rando_prediction)
    if sortbydistance:   
        ordeucs=np.argsort(eucs)  
        eucs=eucs[ordeucs] 
        
        #pdb.set_trace()
        indm=ordeucs[images_read_in] 
        eucshow=eucs[images_read_in] 
    else:      
        indm=np.sort(np.random.default_rng().choice(np.arange(location_use.shape[0]),500,replace=False))
        eucshow=eucs[indm]
        np.random.shuffle(shuf)
    
    location_use_random=location_use.iloc[indm]
    svs_name_of_file_df_random =pd.DataFrame({'filename':location_use_random['wsi_filename'].apply(lambda x: os.path.split(x)[-1])}).merge(gd_,how='left')


    index_x=np.where(location_use_random.columns=='index_x')[0][0]
    colnames_loc=location_use_random.columns
    location_use_random=location_use_random.to_numpy()[shuf]
    #tile_use_random=tile_use_random[shuf]
    svs_name_of_file_df_random=svs_name_of_file_df_random.iloc[shuf] 

    category_idx_random=location_use_random[:,index_x]
    location_use_random=pd.DataFrame(location_use_random,columns=colnames_loc)
    category_predicted_label_random=tcga.loc[category_idx_random,'type']
    category_predicted_full_random = tcga.loc[category_idx_random, 'NCI.T.Label']

    return category_predicted_label_random,category_predicted_full_random,\
    location_use_random, svs_name_of_file_df_random,\
    eucshow   

@st.cache(allow_output_mutation=True,hash_funcs={h5py._hl.dataset.Dataset: lambda _: None, h5py._hl.files.File: lambda _: None})
def category_tiles(file_uploaded_tile,category,rando_prediction,location_use,tcga,tcgalabelmatchprediction,categories_predict,categories_predict33,images_read_in,sortbydistance,gd_):
   
    pre_category_ind=np.where(rando_prediction==np.sort(rando_prediction)[-category])[0]
    ##do not show cagetory nan, nan is 32 
    
    indm=np.array([])
    
    eucs=np.linalg.norm(np.array(categories_predict33)-rando_prediction,axis=1) 
    while indm.shape[0]==0:
        if pre_category_ind[0]==32:
            
            pre_category_ind[0]=np.where(rando_prediction==np.sort(rando_prediction)[-(1+category)])[0]

        if len(pre_category_ind)>1 and max(rando_prediction)==1:
            category_ind=pre_category_ind[-category]         
        else:
            category_ind=pre_category_ind[0]    
        if tcgalabelmatchprediction == 'Match':
            #pdb.set_trace()
            indm=np.where(np.logical_and(location_use['index_y']==category_ind,categories_predict[:]==category_ind))[0]
            indm2=np.where(np.logical_and(location_use['index_y']==category_ind,categories_predict[:]!=category_ind))[0]
            indm=np.hstack((indm,indm2))
        else:
            indm=np.where(np.logical_and(location_use['index_y']==category_ind,categories_predict[:]==category_ind))[0]
        pre_category_ind=1+pre_category_ind
        
    if sortbydistance:   
        ordeucs=np.argsort(eucs[indm],axis=0)
        
        indm=indm[ordeucs]
        
    eucshow=eucs[indm]
    print(eucshow)

    if len(indm)>500:
        indm=indm[images_read_in]
        if sortbydistance:
            eucshow=eucshow[images_read_in]

          
    location_use_correctlyclassified = location_use.iloc[indm]
    svs_name_of_file_df =pd.DataFrame({'filename':location_use_correctlyclassified['wsi_filename'].apply(lambda x: os.path.split(x)[-1])}).merge(gd_,how='left')
 
    rand_ind_c=np.random.choice(location_use_correctlyclassified.shape[0],1)[0]
    category_idx=location_use_correctlyclassified.iloc[rand_ind_c]['index_x']
    category_predicted_label=tcga.loc[category_idx,'type']
    category_predicted_full = tcga.loc[category_idx, 'NCI.T.Label']


    ht_c=tcga.loc[category_idx,'type']
    tcga_c=tcga.loc[category_idx,:]
    
    return tcga_c,category_predicted_label,category_predicted_full,\
    location_use_correctlyclassified, svs_name_of_file_df, \
    category_ind, ht_c,eucshow

@st.cache(suppress_st_warning=True,hash_funcs={tf.keras.Sequential: id,h5py._hl.dataset.Dataset: id, h5py._hl.files.File: id})
def process_upload(cropped_img):
    cropped_img=cropped_img[:,:,0:3]
    ## there should be no need to resize
    # cropped_img_resized_for_densenet=tf.expand_dims(tf.image.resize(np.array(cropped_img)/[255,255,255],(896,896)),axis=0) 
    if np.max(cropped_img)>1: 
        properrange=[255,255,255]
    else:
        properrange=[1,1,1]
    cropped_img_resized_for_densenet=tf.expand_dims(np.array(cropped_img)/properrange,axis=0)

    # break tiles into 4x4 patches, 16 patches for each 896x896 tile
    patched_cropped_img_resized_for_densenet=tf.image.extract_patches(cropped_img_resized_for_densenet,sizes=[1,224,224,1],strides=[1,224,224,1],rates=[1,1,1,1],padding='VALID')
    # reshape the patches to 3d, 16 patches in 16 rows per tile
    stacked_patches=tf.reshape(patched_cropped_img_resized_for_densenet,(cropped_img_resized_for_densenet.shape[0]* 16,224,224,3))
    # no resize of patches
    tiles_encoded=tile_encoder(stacked_patches)
    # perform feature sum for every 16 patches, returning to tile dimension output
    feature_sum_16_patches=tf.math.reduce_max(tf.reshape(tiles_encoded,(patched_cropped_img_resized_for_densenet.shape[0],16,tiles_encoded.shape[-1])),axis=1)       

    return feature_sum_16_patches


@st.cache(suppress_st_warning=True,hash_funcs={tf.keras.Sequential: id,h5py._hl.dataset.Dataset: id, h5py._hl.files.File: id})
def file_uploaded_to_tile_w_predlevel8962240(cropped_img,model_level):
    file_uploaded_tile = process_upload(cropped_img)

    file_upload_predictionlevel=np.round(model_level.predict(file_uploaded_tile),6)
    
    return file_upload_predictionlevel


@st.cache(suppress_st_warning=True,hash_funcs={tf.keras.Sequential: id,h5py._hl.dataset.Dataset: id, h5py._hl.files.File: id})
def file_uploaded_to_tile_w_predlevel(cropped_img,model_level):
    file_uploaded_tile = process_upload(cropped_img)

    file_upload_predictionlevel=np.round(model_level.predict(file_uploaded_tile),6)
    
    #file_upload_predictionlevel=np.reshape(file_upload_predictionlevel,np.max(file_upload_predictionlevel.shape))
    return file_upload_predictionlevel

@st.cache(allow_output_mutation=True,suppress_st_warning=True,hash_funcs={tf.keras.Sequential: id,h5py._hl.dataset.Dataset: id, h5py._hl.files.File: id})
def uploaded_to_vector(levels,tile_size,cropped_img,model_use):

    file_uploaded_tile = process_upload(cropped_img)  
    levels_onehot=tf.one_hot(np.arange(0,12),12).numpy()
    sizetouse=np.where(levels==tile_size)[0][0]
    
    file_uploaded_tile=np.hstack((np.squeeze(file_uploaded_tile),levels_onehot[sizetouse]))
    return file_uploaded_tile

@st.cache(allow_output_mutation=True,suppress_st_warning=True,hash_funcs={tf.keras.Sequential: id,h5py._hl.dataset.Dataset: id, h5py._hl.files.File: id})
def file_uploaded_to_tile_w_pred(levels,tile_size,cropped_img,model_use):
 
    file_uploaded_tile=uploaded_to_vector(levels,tile_size,cropped_img,model_use)

    file_upload_prediction=np.round(model_use.predict(np.expand_dims(file_uploaded_tile,axis=0)),6)
    file_upload_prediction=np.reshape(file_upload_prediction,np.max(file_upload_prediction.shape))[0:32]
   
    return file_upload_prediction




@st.cache(allow_output_mutation=True,suppress_st_warning=True,hash_funcs={tf.keras.Sequential: id,h5py._hl.dataset.Dataset: id, h5py._hl.files.File: id})
def find_tiles_probable_categories(file_uploaded_tile,tcgalabelmatchprediction,tilesize, tcga,location_use,rando_prediction,categories_predict,categories_predict33,images_read_in,cat_ind_four,sortbydistance,gd_):
    print('get 1st category')
    tcga1,category_1_predicted_label,category_1_predicted_full,location_use_1_correctlyclassified, svs_name_of_file1_df,category1_ind, ht1,eucshow1 = category_tiles(file_uploaded_tile,cat_ind_four,rando_prediction,location_use,tcga,tcgalabelmatchprediction,categories_predict,categories_predict33,images_read_in,sortbydistance,gd_)
    print('get 2nd category')
    tcga2,category_2_predicted_label,category_2_predicted_full,location_use_2_correctlyclassified, svs_name_of_file2_df, category2_ind, ht2,eucshow2  = category_tiles(file_uploaded_tile,cat_ind_four+1,rando_prediction,location_use,tcga,tcgalabelmatchprediction,categories_predict,categories_predict33,images_read_in,sortbydistance,gd_)
    print('get 3rd category')
    tcga3,category_3_predicted_label,category_3_predicted_full,location_use_3_correctlyclassified, svs_name_of_file3_df, category3_ind, ht3,eucshow3  = category_tiles(file_uploaded_tile,cat_ind_four+2,rando_prediction,location_use,tcga,tcgalabelmatchprediction,categories_predict,categories_predict33,images_read_in,sortbydistance,gd_)
    print('get 4th category')
    tcga4,category_4_predicted_label,category_4_predicted_full,location_use_4_correctlyclassified, svs_name_of_file4_df, category4_ind, ht4 ,eucshow4 = category_tiles(file_uploaded_tile,cat_ind_four+3,rando_prediction,location_use,tcga,tcgalabelmatchprediction,categories_predict,categories_predict33,images_read_in, sortbydistance,gd_)
    print('get random')
    category_predicted_label_random,category_predicted_full_random,location_use_random, svs_name_of_file_df_random, eucshow_rand  = category_tiles_random(file_uploaded_tile,rando_prediction,tcga,location_use,categories_predict33,images_read_in,sortbydistance,gd_)
    category_predicted_label={'c1':[category_1_predicted_label,category_1_predicted_full],
    'c2': [category_2_predicted_label,category_2_predicted_full],
    'c3': [category_3_predicted_label,category_3_predicted_full],
    'c4': [category_4_predicted_label,category_4_predicted_full]}
    return category_predicted_label, \
    location_use_1_correctlyclassified,\
    svs_name_of_file1_df,\
    category1_ind,\
    location_use_2_correctlyclassified,\
    svs_name_of_file2_df,\
    category2_ind,\
    location_use_3_correctlyclassified,\
    svs_name_of_file3_df, \
    category3_ind, \
    location_use_4_correctlyclassified,\
    svs_name_of_file4_df,\
    category4_ind,\
    ht1,ht2,ht3,ht4,category_predicted_label_random,category_predicted_full_random,location_use_random,svs_name_of_file_df_random,eucshow1,eucshow2,eucshow3,eucshow4,eucshow_rand, 


def sbar_altairlevel(list_of_predlevel_in,tiles_list,levels):
     
    
    list_of_pred=list_of_predlevel_in.copy()
    list_of_pred_np=np.array(list_of_pred).reshape((len(list_of_pred),-1))
    
    list_of_pred_np=np.unique(list_of_pred_np,axis=0)
    list_of_pred=list(list_of_pred_np)
    #while len(list_of_pred)> 1 and np.array_equal(list_of_pred[-2],list_of_pred[-1]):
    #    del list_of_pred[-1]
       
    sta=pd.DataFrame({'value':np.array([j for a in list_of_pred for j in a]),'indeximage':np.repeat(np.arange(0,len(list_of_pred)),1), 'magnification':np.tile("level",len(list_of_pred))})
    st.write(alt.Chart(sta).mark_bar().encode(
    x='magnification',
    y='sum(value)',
    color=alt.Color('indeximage', scale=alt.Scale(scheme='dark2'))
)) 

def sbar_altair(list_of_pred_in,tiles_list,tcga_cancers):
     
    tcga_cancers=tcga_cancers#[0:32]
    list_of_pred=list_of_pred_in.copy()
    list_of_pred_np=np.array(list_of_pred).reshape((len(list_of_pred),-1))
    
    list_of_pred_np=np.unique(list_of_pred_np,axis=0)
       
    list_of_pred=list(list_of_pred_np)
    #list_of_pred_np[0][0:32]
    #while len(list_of_pred)> 1 and np.array_equal(list_of_pred[-2],list_of_pred[-1]):
    #    del list_of_pred[-1]
    
    #sta=pd.DataFrame({'probability':np.array(list_of_pred),'indeximage':np.repeat(np.arange(0,len(list_of_pred)),len(tcga_cancers)), 'tcga':np.tile(tcga_cancers,len(list_of_pred))})
 
    sta=pd.DataFrame({'probability':np.array([j for a in list_of_pred for j in a]),'indeximage':np.repeat(np.arange(0,len(list_of_pred)),len(tcga_cancers)), 'tcga':np.tile(tcga_cancers,len(list_of_pred))})
    st.write(alt.Chart(sta).mark_bar().encode(
    x='tcga',
    y='sum(probability)',
    color=alt.Color('indeximage', scale=alt.Scale(scheme='dark2'))
))     

#@st.cache(allow_output_mutation=True,suppress_st_warning=True,hash_funcs={tf.tf.engine.sequential.Sequential: id})
def bars(category_predicted_label,ht1,ht2,ht3,ht4,rando_prediction,cat_ind_four):

    rando_show_four_top_categories=st.beta_columns(4)
    objects = (category_predicted_label['c1'][0],category_predicted_label['c2'][0],category_predicted_label['c3'][0],category_predicted_label['c4'][0])

    prob =np.sort(rando_prediction)[::-1][(cat_ind_four-1):(cat_ind_four+3)]

    bar= pd.DataFrame({
    'index': objects,
    'probabilities': prob,
    }).set_index('index')


    st.bar_chart(bar)

    rando_show_four_top_categories[0].write("category 1 tcga label {} predicted label {} prob from model {:.3f}".format(ht1,category_predicted_label['c1'][0],prob[0]))   
    rando_show_four_top_categories[1].write("category 2 tcga label {} predicted label {} prob from model {:.3f}".format(ht2,category_predicted_label['c2'][0],prob[1]))
    rando_show_four_top_categories[2].write("category 3 tcga label {} predicted label {} prob from model {:.3f}".format(ht3,category_predicted_label['c3'][0],prob[2]))   
    rando_show_four_top_categories[3].write("category 4 tcga label {} predicted label {} prob from model {:.3f}".format(ht4,category_predicted_label['c4'][0],prob[3]))   

@st.cache(allow_output_mutation=True,suppress_st_warning=True,hash_funcs={list: lambda _: None})
def which_category_show(for_rf_ind,for_rf_tiles,name_and_loc,\
    category_predicted_label,\
    location_use_1_correctlyclassified,\
    svs_name_of_file1_df,\
    category1_ind,\
        eucshow1,\
    location_use_2_correctlyclassified,\
    svs_name_of_file2_df,\
    category2_ind,\
        eucshow2,\
    location_use_3_correctlyclassified,\
    svs_name_of_file3_df,\
    category3_ind,\
        eucshow3,\
    location_use_4_correctlyclassified,\
    svs_name_of_file4_df,\
    category4_ind,\
        eucshow4,\
    tcga,ht1,ht2,ht3,ht4,\
    category_predicted_label_random,category_predicted_full_random,location_use_random,svs_name_of_file_df_random,euc_rand,\
    rownum,colnum,advance,tile_shape,more1234):

   
    start=advance*rownum*colnum
    last=start+(rownum*colnum)
    ind=np.arange(start,last)

    encoded=[]
    histos=[]
    wsi_tiles=[]
    wsi=[]
    eucshow_sub=[]
    more1234=re.sub(':.+','',more1234)
    if more1234==ht1:
        location_use_cont=location_use_1_correctlyclassified
        svs_name_of_file_df=svs_name_of_file1_df
        category_use = category1_ind
        htpred=category_predicted_label['c1'][0]
        eucshow_use=eucshow1 

    elif more1234==ht2:
        location_use_cont=location_use_2_correctlyclassified
        svs_name_of_file_df=svs_name_of_file2_df
        category_use = category2_ind
        htpred=category_predicted_label['c2'][0]
        eucshow_use=eucshow2

    elif more1234==ht3:
        location_use_cont=location_use_3_correctlyclassified
        svs_name_of_file_df=svs_name_of_file3_df
        category_use = category3_ind
        htpred=category_predicted_label['c3'][0]
        eucshow_use=eucshow3 
    elif more1234==ht4:
        location_use_cont=location_use_4_correctlyclassified
        svs_name_of_file_df=svs_name_of_file4_df
        category_use = category4_ind
        htpred=category_predicted_label['c4'][0]
        eucshow_use=eucshow4        
    else:
        location_use_cont=location_use_random
        svs_name_of_file_df=svs_name_of_file_df_random
        category_use= -1
        htpred = category_predicted_label_random
        eucshow_use=euc_rand 

    if not isinstance(htpred,str):
        htpred = htpred[ind[0]:(ind[-1]+1)]

    tcgause=tcga 
    show_image_num_extracted=st.beta_columns(len(ind))
    start_of_rf_ind = for_rf_ind[-1]   
    
    for c,s in enumerate(ind):
        print('ind {}'.format(s))
        
        file_idx=location_use_cont.iloc[s]['index_x']
        
        euc=eucshow_use[s]
        
        print(tcgause.loc[file_idx, 'wsi_filename'])
        wsi_slidename=tcgause.iloc[file_idx]['wsi_filename']
        wsi_slidenum=tcga.iloc[file_idx]
        xloc=location_use_cont.iloc[s]['x_loc']
        yloc=location_use_cont.iloc[s]['y_loc']
        ht=tcgause.loc[file_idx,'type']

     
        print(location_use_cont.iloc[s][['x_loc','y_loc']])
        print(ht)
        print(s)
        #
        #show_image_num_extracted[c].write('read image {}'.format(c+1))
        histos.append(ht)

        name_and_loc.append([wsi_slidename,wsi_slidenum,xloc,yloc,file_idx])
        eucshow_sub.append(euc)
    
    eucshow_array_sub=np.array(eucshow_sub)
    svs_for_output=svs_name_of_file_df.iloc[ind]
    #print('sim_or_not {} for_rf_in {}'.format(len(sim_or_not), len(for_rf_ind)))
    return name_and_loc,htpred, category_use,histos,eucshow_array_sub,svs_for_output





def show_similar( histos,rando_prediction,eucshow_array_sub,rownum,colnum,htpred,for_rf_ind,advance,svs_for_output,lenofpages ):
    bbb=0
    similar_check=[]
    dissimilar_check=[]
    cols=[None] * rownum
    
    #pdb.set_trace()
    with st.beta_expander('similar tiles'):
        for i in np.arange(0, rownum):
            print("i is {}".format(i))
            cols[i]=st.beta_columns(colnum)
            smartoptions_l=[]
            for h in np.arange(0, colnum):
               
                linkprint=os.path.join('https://portal.gdc.cancer.gov/files/',svs_for_output.iloc[bbb]['id'])
                histoshow=histos[bbb]
 
                if not isinstance(htpred,str):
                    htpredshow=[str(g) for g in htpred[bbb:(bbb+1)]][0]

                else:
                    htpredshow=htpred    
                
                if 1:#len(for_rf_ind)>len(sim_or_not):

                    if not isinstance(htpred,str):
                        #sim_or_not.append(cols[i][h].selectbox(label='',options=['no opinion','yes','no'],key=bbb))
                        #sim_or_not.append(cols[i][h].selectbox(label="{}) label from tcga {} Pred for category {}".format(h,histoshow,htpredshow),options=smartoptions,key=bbb))
                        cols[i][h].write("{}) label from tcga {} Pred for category {} euclidian {:.3f} {}".format(bbb,histoshow,htpredshow,eucshow_array_sub[bbb],linkprint))
                    else:
                        #sim_or_not.append(cols[i][h].selectbox(label='',options=['no opinion','yes','no'],key=bbb))
                        #sim_or_not.append(cols[i][h].selectbox(label="{}) label from tcga {} Pred for category {} {:.3f}".format(h,histoshow,htpredshow,predictions_similar[bbb]),options=smartoptions,key=bbb))
                        cols[i][h].write("{}) label from tcga {} Pred for category {} euclidian {:.3f} {}".format(bbb,histoshow,htpredshow,eucshow_array_sub[bbb],linkprint))
                
                bbb=bbb+1

           

    return

@st.cache(allow_output_mutation=True,hash_funcs={h5py._hl.dataset.Dataset: id})
def get_random(ran,location_use):
    r=np.random.choice(location_use.shape[0],1)
    #print(r)
    return r
 


def demonstrate_image():
    st.set_page_config(layout="wide")
    tcga,model_level,model_diagnosis,levels,gd_= get_()

    st.markdown('# Similar Image Search')

    
    st.sidebar.title('To use your own image as the query use the file upload')
    

    img_file = st.sidebar.file_uploader(label='Image', type=['png', 'jpg','jpeg'])
    
    if img_file is not None:
        saf=st.radio('Show explanations of function adjacent to buttons',[False,True])  
        print('just made image file')

        img = Image.open(img_file)
        p_for_cropped_img=img.crop((0,0,896,896))
        for_cropped_img=np.array(p_for_cropped_img)[:,:,0:3]
        with st.beta_expander("Image post-crop, Full Size"):
        #_ = cropped_img.thumbnail((150,150))
            
            if for_cropped_img.shape[0]<896:
                paddings = [[0, 896-tf.shape(for_cropped_img)[0]], [0,0],[0,0]]#[0, 896-tf.shape(nn)[1]],[0,0]]
                for_cropped_img = tf.pad(for_cropped_img, paddings, 'CONSTANT', constant_values=0) 
            if  for_cropped_img.shape[1]<896:
                paddings = [[0, 0],[0, 896-tf.shape(for_cropped_img)[1]],[0,0]] 
                for_cropped_img = tf.pad(for_cropped_img,paddings,'CONSTANT',constant_values=0)  
           
            
            
            st.image(for_cropped_img,width=np.array(for_cropped_img).shape[0])#,height=np.array(for_cropped_img).shape[1])

            get_data().append(for_cropped_img)
            #if len(get_data())>1:
            #    del get_data()[0]
            cropped_img=get_data()[-1] 

        with st.beta_expander('level predictor'):
            get_tilelevel().append(file_uploaded_to_tile_w_predlevel(cropped_img,model_level))
            
            stacked_altairlevel=sbar_altairlevel(get_tilelevel(),get_data(),levels)

            predicted_level = file_uploaded_to_tile_w_predlevel(cropped_img,model_level)[0][0]
            levelp=st.beta_columns(12)
            
            st.write('The magnification level prediction is: {:.2f}'.format(predicted_level))
            for ii in np.arange(0,12):
                levelp[ii].write(' {}'.format(levels[ii]))
            
            
            tilesize=st.selectbox('Tilesize to use,',levels,index=int(np.round(predicted_level)))     
          

    else:
        caching.clear_cache()
        st.stop()    
    if saf:    
        st.sidebar.write('Choose the size of images that will be returned as similar Note, \
            this might necessitate estimation of the input resolution, choosing the image resolution for similars most closely\
                matching the input is recommended')

    tile_shape = (tilesize, tilesize, 3)
    if saf:
        st.sidebar.write('The search will display the number of rows x num of columns that are selected, with a default of 9 images, 3x3')

    rownum=st.sidebar.selectbox(
        'Number of Rows',
        [2,3,4,5,6,7,8,9,10],index=1)
    colnum=st.sidebar.selectbox(
        'Number of Columns',
        [2,3,4,5,6,7,8,9,10],index=1)

    if saf:
            st.sidebar.write('read encodings of images in intervals of 500. Reading all image encodings is slow, reading 500 at a time\
                makes for speedier computation. The \'advance screen\' slider below the \'read images\' slider \
                    moves through the 500 images, in row num x colnum increments. Once the \'advance screen\' slider has moved as far right as it will go \
                        it is necessary to change the \'read image\' slider to the right to read in another 500 encodings') 
    advancesliderminimum=st.sidebar.select_slider('read images, each increment reads 500 new tiled images disk',['0-0.5K','0.5-1K','1-1.5K','1.5-2K','2-2.5K','2.5-3K','3-3.5K','3.5-4K','4-4.5K','4.5-5K','...any remaining'])
    image_indices_to_read = {'0-0.5K':np.arange(0,500),
    '0.5-1K':np.arange(500,1000),
    '1-1.5K':np.arange(1000,1500),
    '1.5-2K':np.arange(1500,2000),
    '2-2.5K':np.arange(2000,2500),
    '2.5-3K':np.arange(2500,3000),
    '3-3.5K':np.arange(3000,3500),
    '3.5-4K':np.arange(3500,4000),
    '4-4.5K':np.arange(4000,4500),
    '4.5-5K':np.arange(4500,5000),
    'remaining':np.arange(5000,20000)}
    images_read_in=image_indices_to_read[advancesliderminimum]
    
    lenofpages=int(np.round(len(images_read_in)/(colnum*rownum)))

    if saf:
        st.sidebar.write('Below slider moves through each 500 image chunk')
    advance = st.sidebar.slider('advance screen',
    min_value=0,max_value=lenofpages,value=0,step=1)
    


    location_use, cancers_tcga=datain(tilesize,levels)
    
    categories_predict=location_use.prediction_category
    categories_predict33=location_use[(np.arange(0,32))]
    model_use = model_diagnosis

    


    ind_rando_tile=-1           
    
    ## hack

    if saf:
        st.sidebar.write('below controls whether the similars are patches that are predicted correctly, \'Match\' or predicted incorrectly, \'Do Not Match\'. Choosing \'Do Not Match\' therefore displays patches that are predicted as one tumor and annotated as a different tumor')
    tcgalabelmatchprediction= st.radio('TCGA labels should match prediction labels',['Match','Do Not Match'])
    
    if saf:
        st.sidebar.write('The predictions directly above each similar image are from the premade TCGA-trained fully connected graph\
             or from the user-created classifier, with the similar or not similar designation assigned in interactive fashion')

    
    

    get_tile().append(file_uploaded_to_tile_w_pred(levels,tilesize,cropped_img,model_use))
    file_upload_prediction = get_tile()[-1]
    rando_prediction = file_upload_prediction
    for_alt_bar=get_tile()    
    file_uploaded_tile=uploaded_to_vector(levels,tilesize,cropped_img,model_use)
    with st.beta_expander('Distance metric'):
        sortbydistance=st.radio('Sort using distance metric (Euclidean',[True,False])
    if saf:
        st.sidebar.write('See query')
    thum=st.sidebar.radio('Thumbnail of query',['view','conceal'])
    if thum == 'view':
        if img_file is not None:
            _ = p_for_cropped_img.thumbnail((150,150))
            st.sidebar.image(p_for_cropped_img)
        else:
            wsi_tile_pil = Image.fromarray(wsi_tile)
            _ = wsi_tile_pil.thumbnail((150,150))
            st.sidebar.image(wsi_tile_pil)    
    
    cat_toshow=st.selectbox('categories predicted, shown in descending probability',['1-4','5-8','9-12','13-16','17-20','21-24','25-28','29-32'])
    cat_toshow_dic={'1-4':1,'5-8':5,'9-12':9,'13-16':13,'17-20':17,'21-24':21,'25-28':25,'29-32':29}
    cat_ind_four=cat_toshow_dic[cat_toshow]
    with st.beta_expander('Predicted'):
        category_predicted_label, location_use_1_correctlyclassified, svs_name_of_file1_df, \
        category1_ind,location_use_2_correctlyclassified, svs_name_of_file2_df, category2_ind,location_use_3_correctlyclassified,\
        svs_name_of_file3_df,category3_ind, \
        location_use_4_correctlyclassified,svs_name_of_file4_df,category4_ind,\
        ht1,ht2,ht3,ht4, category_predicted_label_random,category_predicted_full_random,location_use_random,svs_name_of_file_df_random,eucshow1, eucshow2, eucshow3, eucshow4, euc_rand  = \
        find_tiles_probable_categories(file_uploaded_tile,tcgalabelmatchprediction,tilesize, tcga,location_use,rando_prediction,categories_predict,categories_predict33,images_read_in,cat_ind_four,sortbydistance,gd_)
        #pdb.set_trace()
        for_more=[':'.join([str(i) for i in category_predicted_label['c1']]),
        ':'.join([str(i) for i in category_predicted_label['c2']]),
        ':'.join([str(i) for i in category_predicted_label['c3']]),
        ':'.join([str(i) for i in category_predicted_label['c4']]),
        'tiles at random']

        if saf:
            st.write('Choose the tumor images to display')
        more1234=st.radio("see more",for_more)
        if saf:
            st.sidebar.write('Sliding below starts the search over and removes previous results')
 
        for_rf_tiles, name_and_loc,for_rf_ind= rf_start_over(category_predicted_label,ind_rando_tile)

        bars(category_predicted_label,ht1,ht2,ht3,ht4,rando_prediction,cat_ind_four)
        if img_file:
            stacked_altair=sbar_altair(get_tile(),get_data(),cancers_tcga)
        name_and_loc,htpred, category_use,histos,eucshow_array_sub,svs_for_output = \
        which_category_show(for_rf_ind,for_rf_tiles,name_and_loc,category_predicted_label,\
        location_use_1_correctlyclassified,\
        svs_name_of_file1_df,\
        category1_ind,\
            eucshow1,\
        location_use_2_correctlyclassified,\
            svs_name_of_file2_df,\
        category2_ind,\
            eucshow2,\
        location_use_3_correctlyclassified,\
        svs_name_of_file3_df,\
        category3_ind,\
            eucshow3,\
        location_use_4_correctlyclassified,\
            svs_name_of_file4_df,\
        category4_ind,\
            eucshow4,\
        tcga,ht1,ht2,ht3,ht4,\
        category_predicted_label_random,category_predicted_full_random,location_use_random,svs_name_of_file_df_random, euc_rand,\
        rownum,colnum,advance,tile_shape,more1234)   
        

    show_similar( histos,rando_prediction, eucshow_array_sub,rownum,colnum,htpred,for_rf_ind,advance,svs_for_output,lenofpages )

if __name__ == '__main__':
    demonstrate_image()
