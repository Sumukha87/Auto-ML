import streamlit as st
import pandas as pd
import os
#import pandas_profiling
import ydata_profiling
from ydata_profiling import ProfileReport
from pycaret.datasets import get_data
from streamlit_pandas_profiling import st_profile_report
import pycaret
from pycaret.classification import setup,compare_models,pull,save_model
juice = get_data('juice')

with st.sidebar:
    st.image('https://www.wi6labs.com/wp-content/uploads/2019/12/Machine-learning-logo-1.png',width=200)
    st.title('AutoMachineLearning-By Sumukha')
    choise=st.radio('Navigation',['Upload','Profiling','ML','Download'])
    st.info('This a Python application which allows you to build an automated ML pipline using StreamLit.')


if os.path.exists('sourcedata.csv'):
    df=pd.read_csv('sourcedata.csv',index_col=None)

if choise=='Upload':
    st.title('Upload any of your Data for Modelling!!')
    file=st.file_uploader('Upload your Dataset here.')

    if file:
        df= pd.read_csv(file ,index_col=None)
        
        df.to_csv('sourcedata.csv',index=None)
        st.dataframe(df)


if choise=='Profiling':
    st.title('Automated Profiling of your data')
    profile = ProfileReport(df, title="Profiling Report")
    st_profile_report(profile)



if choise == 'ML':
    st.title('Machine Learning for your Data!!')
    target1=st.selectbox("Select your Target",df.columns)
    setup(data = juice,  target = 'Purchase')
    setup_df=pull()
    st.info('This is the ML Experiment Settings')
    st.dataframe(setup_df)
    best_model=compare_models()
    compare_df=pull()
    st.info('This is the Pycreat ML Model')
    st.dataframe(compare_df)
    save_model(best_model,'best_model')

if choise == 'Download':
    with open("best_model.pkl", 'rb') as f: 
        st.download_button("Download Model", f, "best_model_test.pkl")