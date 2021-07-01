# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:54:21 2021

@author: ShingFoon Lin
"""

import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import yfinance as yf
import datetime
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

#st.markdown(
#    "*Check out the methodology walk-through "
#    "[here](https://www.crosstab.io/articles/staged-rollout-analysis) and Streamlit "
#    "app mechanics [here](https://www.crosstab.io/articles/streamlit-review).*"
#)

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

#data = pd.read_csv(r"C:\Users\MSI_NB\Desktop\03-Streamlit-Forecast-WebApp-main\NSE-TATAGLOBAL11.csv")
#data = data.sort_index(ascending=False)











#df = data.set_index(["Date"])
R = np.log(data.iloc[:,4].pct_change().dropna()+1) * 100;

plt.figure(figsize=(20,6))
data.iloc[:,4].plot(label = "prices");plt.legend()

left_col.pyplot(plt)

plt.figure(figsize=(20,6))
R.plot(label = "return");plt.legend();

left_col.pyplot(plt)

def stat(x) : 
    return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),
                      x.std(),x.skew(),x.kurt()],index=['count','min','idxmin','25% quantile',
                    'median','75% quantile','mean','max','idxmax','mad','var','std','skew','kurt'])
                                                        
def autocorrelation_plot(y, lags=None, figsize=(12, 5), style='bmh',Title = None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    #with plt.style.context(style):    
    fig = plt.figure(figsize=figsize)
    layout = (1, 2)
    acf_ax = plt.subplot2grid(layout, (0, 0))
    acf2_ax = plt.subplot2grid(layout, (0, 1))

    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, title= "{} Autocorrelation Plot".format(Title))
    smt.graphics.plot_acf(y ** 2, lags=lags, ax=acf2_ax, title= "Squared {} Autocorrelation Plot".format(Title))
    plt.tight_layout()                                                        

statr = pd.DataFrame(stat(R), columns=["Details of Return"])                                                        
                                                   

plt.figure(figsize=(8.95,6))
_, bins, _ = plt.hist(R, 50, density=1, alpha=0.5, label="Frequency", rwidth = 20)
mu, sigma = norm.fit(R)
best_fit_line = norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line, label="Nomal");plt.legend()

right_col.pyplot(plt)

fig, ax = plt.subplots(figsize=(9.5,6))
sm.qqplot(R, line ='45',ax = ax)
right_col.pyplot(plt)

fig = autocorrelation_plot(R, lags = 50, figsize=(20,4),Title = "Return")
st.pyplot(fig)


st.write(statr.T)



     

#options = data['Date'].dt.strftime('%Y-%m-%d').tolist()
#options = np.array(data['Date']).tolist()


#(start_time, end_time) = st.select_slider("Choose the time frame：",
#     options = options,
#     value= ('2014-06-27','2016-05-04'),
# )

#st.subheader('Chosed Time series')
#st.write("Starting time:",start_time)
#st.write("Ending time:",end_time)















