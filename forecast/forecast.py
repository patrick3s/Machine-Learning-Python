import pandas as pd
from pandas.core.frame import DataFrame
import pandas_datareader.data as web
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error , r2_score
from sklearn import datasets, linear_model
import yfinance as yf

yf.pdr_override()

def get_ibov_info() -> DataFrame:
    url = "http://bvmf.bmfbovespa.com.br/indices/ResumoCarteiraTeorica.aspx?Indice=IBOV&amp;idioma=pt-br"
    html = pd.read_html(url, decimal=",", thousands=".")[0][:-1]
    df = html.copy()[['Código', 'Ação', 'Tipo']]
    df.rename(columns={
        'Código': 'symbol',
        'Ação': 'name',
        'Tipo': 'type'
    }, inplace=True)
    df['yf_symbol'] = df['symbol'] + '.SA'
    return df

# make request action in yahoo finance to return data frame
def data_frame_data_action(action):
    df : DataFrame= web.DataReader(action,source='yahoo',start='2020-01-01',end='2021-11-11')
    df['Date'] = pd.to_datetime(df.index,format ='%Y-%m-%d')
    df.reset_index(drop=True,inplace=True)
    return df;

# creating a moving average based on days
def create_mmd_in_action_by_days_by_collum(data_frame : DataFrame,days: int =None,collum : str=None):
    assert isinstance(days,int) and days >= 1,'invalid day value'
    assert isinstance(collum,str) and collum != '','invalid collum value'
    assert isinstance(data_frame,DataFrame), 'invalid data_frame value'
    data_frame[f'mm{days}d'] = data_frame[collum].rolling(days).mean()
    

def pushing_the_stock_values_forward(data_frame:DataFrame,collum:str):
    assert isinstance(collum,str) and collum != '','invalid collum value'
    assert isinstance(data_frame,DataFrame), 'invalid data_frame value'
    data_frame[collum] = data_frame[collum].shift(-1)
    data_frame.dropna(inplace=True)

def values_to_test(data_frame:DataFrame):
    assert isinstance(data_frame,DataFrame), 'invalid data_frame value'
    qtd_rows = len(data_frame)
    percentage_training = 70
    percentage_test = 15
    qtd_rows_remove_training = (qtd_rows * percentage_training) // 100
    qtd_rows_training = qtd_rows - qtd_rows_remove_training
    qtd_rows_test = qtd_rows - percentage_test
    qtd_rows_validation = qtd_rows_training - qtd_rows_test
   
    features_to_forecast(data_frame,'Close',qtd_rows_training,qtd_rows_test,qtd_rows)

def features_to_forecast(data_frame:DataFrame, collum : str,qtd_row_training:int,qtd_row_test:int,qtd_rows):
    assert isinstance(collum,str) and collum != '','invalid collum value'
    assert isinstance(data_frame,DataFrame), 'invalid data_frame value'
    labels = data_frame[collum]
    features = data_frame.drop(['Open','Date','Close','mm21d'],1)
    scaler = MinMaxScaler().fit(features)
    features_scale = scaler.transform(features)
    
    x_train = features_scale[:qtd_row_training]
    y_train = labels[:qtd_row_training]
    lr = linear_model.LinearRegression()
    lr.fit(x_train,y_train)
    forecast = features_scale[qtd_row_test:qtd_rows]
    data_frame_date= data_frame['Date']
    date_action = data_frame_date[qtd_row_test:qtd_rows]

    res_full = data_frame['Close']
    res = res_full[qtd_row_test:qtd_rows]

    pred =lr.predict(forecast)
    df = pd.DataFrame({'data_pregao':date_action,'real':res,'previsao':pred})
    df['real'] = df['real'].shift(+1)
    df.set_index('data_pregao',inplace=True)
    print(df)

action = data_frame_data_action('MGLU3.SA')
create_mmd_in_action_by_days_by_collum(action,days=5,collum='Close')
create_mmd_in_action_by_days_by_collum(action,days=21,collum='Close')
pushing_the_stock_values_forward(action,'Close')
values_to_test(action)


