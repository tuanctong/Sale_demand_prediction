import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


st.title("Forecast Sales by SARIMA model")
option = st.selectbox('Which SKU would you like to forecast?',
('SKU1', 'SKU2', 'SKU3', 'SKU4', 'SKU5', 'SKU6', 'SKU7', 'SKU8', 'SKU9', 'SKU10',
'SKU11', 'SKU12', 'SKU13', 'SKU14', 'SKU15', 'SKU16', 'SKU17', 'SKU18', 'SKU19', 'SKU20',
'SKU21', 'SKU22', 'SKU23', 'SKU24', 'SKU25', 'SKU26', 'SKU27', 'SKU28', 'SKU29', 'SKU30',
'SKU31', 'SKU32', 'SKU33', 'SKU34', 'SKU35', 'SKU36', 'SKU37', 'SKU38', 'SKU39', 'SKU40',
'SKU41', 'SKU42', 'SKU43', 'SKU44'),index=10)
st.write('You selected:', option)
i = int(option[3:])
sku = pickle.load(open('sku.p', 'rb'))
train_data = pickle.load(open('train_data.p', 'rb'))
test_data = pickle.load(open('test_data.p', 'rb'))
predict_ser = pickle.load(open('predict_ser.p', 'rb'))
confint = pickle.load(open('confint.p', 'rb'))
st.header("Sale Prediction for 3 months look ahead")
n_pred_period = 4*3

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlabel('Week', loc='right')
ax.set_ylabel('Sale')
ax.plot(sku[i][:len(train_data[i])+1],color='chocolate',
            marker='o', markersize=3.5,label='Actual Sales (Training)')
ax.plot(sku[i][len(train_data[i]):len(train_data[i])+n_pred_period],color='skyblue',
        linestyle='dotted',marker='o',markersize=3.5,label='Actual Sales (Testing)')
ax.plot(predict_ser[i], linestyle='dashdot', 
            marker='o',markersize=3.5,label = 'Predicted Sale',color='red')
lower = confint[i][:, 0]
upper = confint[i][:, 1]
ax.fill_between(predict_ser[i].index, lower, upper, color='lightgrey', alpha=0.2)

fig.suptitle('SARIMA Regression Model Forecast for 3 Months Look Ahead - SKU {}'.format(i), fontsize=18)
ax.legend()
st.pyplot(fig)

def measure_metric(test_data, predict_ser,i):
  y = pd.DataFrame.to_numpy(test_data[i][:n_pred_period]).squeeze()
  yhat = pd.DataFrame.to_numpy(predict_ser[i]).squeeze()
  e = y-yhat
  mape=np.mean(e/y)*100
  return 'Mean Absolute Percentage Error: {:.2f} %'.format(mape)
mape = measure_metric(test_data, predict_ser,i)
st.header(mape)
