# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:54:21 2021

@author: ShingFoon Lin
"""



import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm
from scipy.stats import norm, jarque_bera, kurtosis
import statsmodels.tsa.api as smt
seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(16, 6))
plt.rc("savefig", dpi=90)
plt.rc("font", family="sans-serif")
plt.rc("font", size=14)
st.set_option('deprecation.showPyplotGlobalUse', False)





## Basic setup and app layout
st.set_page_config(layout="wide")  # this needs to be the first Streamlit command called
st.title("QBUS6830 Time series Analysis for stocks")

st.markdown(
    "This app is to create a framework to modeling both mean and volatility part of a return series, under the framework delivered in QBUS6830 of USYD. "
)

st.sidebar.title("Control Panel")
left_col, right_col = st.beta_columns((2,1))

## User inputs on the control panel
st.sidebar.subheader("Choose the analysis target here")
symbol = st.sidebar.text_input("Input Yahoo Tickers", value = "TSLA")

def get_date():
    st.sidebar.subheader("Retrive the time series")
    today = datetime.date.today()
    start_date = st.sidebar.date_input("Selecting the Start date",datetime.date(2018,12,30))
    end_date = st.sidebar.date_input("Selecting the End date",today)
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must fall after start date.')
    return start_date, end_date


START_DATE, END_DATE = get_date()
##generate the stocks data

#def generate_data(tickers, START_DATE, END_DATE):
    
    #data = yf.download(tickers, start=START_DATE,end=END_DATE, adjusted=True)
    #return data

data = yf.download(symbol, start=START_DATE,end=END_DATE, adjusted=True)
data = data.reset_index()
data.index = data["Date"].dt.strftime('%Y-%m-%d').tolist()
#data = pd.read_csv(r"C:\Users\MSI_NB\Desktop\03-Streamlit-Forecast-WebApp-main\NSE-TATAGLOBAL11.csv")
#data = data.sort_index(ascending=False)











#df = data.set_index(["Date"])
R = np.log(data.iloc[:,4].pct_change().dropna()+1) * 100;

left_col.subheader("{} price series over {} to {}".format(symbol, START_DATE, END_DATE))
plt.figure(figsize=(20,6))
data.iloc[:,4].plot(label = "prices");#plt.legend();
plt.xlim(data.index.min(), data.index.max())
left_col.pyplot(plt)

left_col.subheader("{} return series".format(symbol))
plt.figure(figsize=(20,6))
R.plot(label = "return");#plt.legend();

left_col.pyplot(plt)

def stat(x) : 
    return pd.Series([x.count(),round(x.min(),4),x.idxmin(),round(x.quantile(.25),4),round(x.median(),4),
                      round(x.quantile(.75),4),round(x.mean(),4),round(x.max(),4),x.idxmax(),round(x.var(),4),
                      round(x.std(),4),round(x.skew(),4),round(x.kurt(),4)],index=['count','min','idxmin','25% quantile',
                    'median','75% quantile','mean','max','idxmax','var','std','skew','kurt'])

statr = pd.DataFrame(stat(R), columns=["{}'s Return".format(symbol)]).T                                                        
statr = statr.round({'min':4,'25% quantile':4,'median':4,'75% quantile':4,'mean':4,'max':4,'var':4,'std':4,'skew':4,'kurt':4}) 
                                                        
def autocorrelation_plot(y, lags=None, figsize=(12, 5), style='bmh',Title = None, simbol = None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    #with plt.style.context(style):    
    fig = plt.figure(figsize=figsize)
    layout = (1, 2)
    acf_ax = plt.subplot2grid(layout, (0, 0))
    acf2_ax = plt.subplot2grid(layout, (0, 1))

    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, title= "{}'s {} Autocorrelation Plot".format(simbol, Title))
    smt.graphics.plot_acf(y ** 2, lags=lags, ax=acf2_ax, title= "{}'s Squared {} Autocorrelation Plot".format(simbol,Title))
    plt.tight_layout()   

def PACF_plot(y, lags=None, figsize=(12, 5), style='bmh',Title = None, simbol = None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    #with plt.style.context(style):    
    fig = plt.figure(figsize=figsize)
    layout = (1, 2)
    acf_ax = plt.subplot2grid(layout, (0, 0))
    pacf_ax = plt.subplot2grid(layout, (0, 1))

    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, title= "{}'s {} Autocorrelation Plot".format(simbol, Title))
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, title= "{}'s {} Partial Autocorrelation Plot".format(simbol,Title))
    plt.tight_layout()  

                                                  

right_col.subheader("Histogram of {}'s return series vs. Nomal distribution".format(symbol))
plt.figure(figsize=(8.95,6))
_, bins, _ = plt.hist(R, 50, density=1, alpha=0.5, label="Frequency", rwidth = 20)
mu, sigma = norm.fit(R)
best_fit_line = norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line, label="Nomal");plt.legend()

right_col.pyplot(plt)


right_col.subheader("QQ-plot of {}'s return series".format(symbol))
fig, ax = plt.subplots(figsize=(9.5,6))
sm.qqplot(R, line ='45',ax = ax)
right_col.pyplot(plt)

st.markdown("✅    Return's ACF. vs. Squred Return's ACF")
PACF = st.checkbox("Return's ACF. vs. Return's PACF")

if PACF:
    fig = PACF_plot(R, lags = 50, figsize=(20,4),Title = "return", simbol = symbol)    
else:
    fig = autocorrelation_plot(R, lags = 50, figsize=(20,4),Title = "return", simbol = symbol)
st.pyplot(fig)


st.subheader("{}'s Return Statistics".format(symbol))
st.write(statr)



     

#options = data['Date'].dt.strftime('%Y-%m-%d').tolist()
#options = np.array(data['Date']).tolist()


#(start_time, end_time) = st.select_slider("Choose the time frame：",
#     options = options,
#     value= ('2014-06-27','2016-05-04'),
# )

#st.subheader('Chosed Time series')
#st.write("Starting time:",start_time)
#st.write("Ending time:",end_time)















