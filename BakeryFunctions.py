# -*- coding: utf-8 -*-
# @Author: Abi.Magnall
# @Date:   2022-10-07 

# import core ds libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error
import math  
import sklearn.metrics
import itertools
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Variables 

dark_magpie = '#112E3E'
medium_dark_magpie = '#054554'
medium_magpie = '#235087'
light_magpie = '#5489CB'
lighter_magpie = '#d1f5ff'
red_magpie = '#FF6459'
yellow_magpie = '#E9BA49'
pink_magpie = '#F8EEF9'
grey_magpie = '#AEAEAE'



# Raw Data Cleaning and Exploration Functions

def initial_eda(df):
    print(f'There are {df.shape[1]} columns and {df.shape[0]} rows \n')
    print('The first 5 rows are:')
    display(df.head())
    print('')
    print('The final 5 rows are: ')
    display(df.tail())
    print('')
    display(df.info())
    

def further_eda(df):
    print(f'There are {df.shape[1]} columns and {df.shape[0]} rows \n')
    display(df.info())
    print('\nThe number of missing values in each column are:')
    display(df.isna().sum())
    print('')
    print(f'There are {df.duplicated().sum()} duplicated rows of data in the dataset')
    print('')
    display(df.describe())
    
    
def clean_summary(df):
    print(f'There are {df.shape[1]} columns and {df.shape[0]} rows\n')
    print(f'The number of duplicated rows are: {df.duplicated().sum()}\n')
    print(f'The number of missing values are:\n{df.isna().sum()}\n')
    display(df.info())
    print('')
    display(df.describe())
    

def shop_clean(df):
    
    packet_products_dictionary ={21139258:2, 21139127:2, 21139133:2,10027913:4, 10027916:6, 21139182:2, 10029435:6,
                 10029439:8, 21139291:2, 10029409:6, 10029413:8, 23278184:3, 24359230:2, 24678085:2,
                 10025080:4, 10025703:6, 21139178:2}
    
    print('Dropping duplicate rows in progress\n')
    df.drop_duplicates(inplace=True)
    
    negative_quantity_rows = df[df['Quantity']<0]
    negative_discount_rows = df[df['DiscountAmount']<0]
    
    null_columns = ['DiscountAmount','DiscountReasonId','RefundReasonId']
    print('Filling the null values in progress\n')
    for column in null_columns:
        df[column].fillna(0, inplace=True)
    print(f'The number of missing values is: {df.isna().sum()}')
          
    print('\nSplitting the Date column and changing to DateTime in progress\n')
    df[['Date', 'Time']] = df.Date.str.split('T',expand=True)      
    df.drop(columns=['Time'],inplace=True)
    print('To validate the Time column has been dropped: ')    
    display(df.info())
    
    print('\nConverting the negative quantity values in progress\n')
    for row in negative_quantity_rows.index:
        df.loc[row,'Quantity'] = abs(df.loc[row,'Quantity'])
    print('To validate there are no remaining negative quantity values:\n')
    display(df[df['Quantity']<0])
    print('')
          
    print('\nConverting the negative discount amount values in progress\n')
    for row in negative_discount_rows.index:
        df.loc[row,'DiscountAmount'] = abs(df.loc[row,'DiscountAmount'])
    print('To validate there are no remaining negative discount amount values:\n')
    display(df[df['DiscountAmount']<0])
    print('')
          
    print('Adjusting quantity values for packet products in progress:\n')
    for key in packet_products_dictionary.keys():
        df.loc[df['ProductId']==key, 'Quantity']=packet_products_dictionary[key]
    print('Adjusting quantity values for packet products complete\n')
    
    print('Changing the Date column to be the correct data type and sorting the column\n')
    df['Date'] = df['Date'].astype('datetime64')
    df = df.sort_values('Date')
    return df


# Shop Data Processing 

def basic_eda(df):
    print(f'The number of duplicated rows are: {df.duplicated().sum()}\n')
    print(f'The number of missing values are: \n{df.isna().sum()}\n')

    display(df.head())
    
    display(df.info())
    
def convert_date(df, date_format='%Y-%m-%d'):
    df['Date']=  pd.to_datetime(df['Date'], format=date_format)
    display(df.info())
    
def revenue_calc(df):
    packet_products = [21139258, 21139127, 21139133,10027913, 10027916, 21139182, 10029435,
                 10029439, 21139291, 10029409, 10029413, 23278184, 24359230, 24678085,
                 10025080, 10025703, 21139178]
    
    for row in df.index:
        if df.loc[row,'RefundReasonId'] > 0:
            df.loc[row,'Revenue'] = 0
        elif df.loc[row, 'DiscountAmount'] > df.loc[row,'UnitPrice']*df.loc[row,'Quantity']:
            df.loc[row,'Revenue'] = 0
        elif df.loc[row,'DiscountAmount'] >0:
            if df.loc[row,'ProductId'] not in packet_products:
                df.loc[row,'Revenue'] = df.loc[row,'UnitPrice']*df.loc[row,'Quantity'] - df.loc[row,'DiscountAmount']
            else:
                df.loc[row,'Revenue'] = df.loc[row,'UnitPrice'] - df.loc[row,'DiscountAmount']
        else:
            if df.loc[row,'ProductId'] not in packet_products:
                df.loc[row,'Revenue'] = df.loc[row,'UnitPrice']*df.loc[row,'Quantity']
            else:
                df.loc[row,'Revenue'] = df.loc[row,'UnitPrice'] 
            
    display(df.head())
    
def total_revenue_calc(df):
    i=0
    for shop in df:
        shop_name = ['Aldeburgh', 'Southwold', 'Darsham', 'Norwich']
        print(f'The total revenue for {shop_name[i]} and products sold:')
        display(shop[['Revenue','Quantity']].sum())
        i+=1


def rename_col(df,col, rename):
    df.rename(columns={col:rename},inplace=True)
    return df


def add_missing_dates(df):
    # To validate there are no missing dates: 
    first_day = df.index.min()
    last_day=df.index.max()
    full_range = pd.date_range(start=first_day, end=last_day, freq="D")

    # to identify the missing dates in the dataframe 
    full_range.difference(df.index)
     
    # To add the missing dates
    df = df.reindex(full_range)
    return df


def interpolate_dates(df, col=''):
    if len(col) > 0:
        missing_dates = df[df[col].isna()].index
        df[col].interpolate(inplace=True)
    
        print(f'The number of null values are:\n{df.isna().sum()}\n')
        for date in missing_dates:
            display(df.loc[date])
        
    else:
        missing_dates = df[df['Croissant'].isna()].index
        df.interpolate(inplace=True)
        
    print(f'The number of null values are:\n{df.isna().sum()}')
    
    
    
def merge_dataset(df1, df2, column):
        df1 = pd.merge(df1,df2, on=column, how='left')
        return df1
    
    
def shop_product_clean(df):
        df = df.drop(columns=['Name', 'ProductId'])
        df = df[df['ProductName']!='Remove']
        df = df.groupby(['Date', 'ProductName']).sum()
        return df
    
    
    
def christmas_fill(df, col=''):
    christmas_rev = {'2020-12-25':0, '2021-12-25':0}

    for date in christmas_rev.keys():
        if len(col) > 0:
            df.loc[date, col]=christmas_rev[date]
        else:
            df.loc[date]=christmas_rev[date]
            
            
            
            
def school_holidays_add(df, school_hols):
    weekday_dictionary = {'Monday': 'Weekday', 'Tuesday': 'Weekday', 'Wednesday': 'Weekday',
                'Thursday': 'Weekday', 'Friday': 'Weekday', 'Saturday':'Weekend', 'Sunday':'Weekend'}
    
    df['Day'] = df.index.day_name()
    for date in df.index:
        for row in school_hols.index:
            if school_hols.loc[row,'Date'] == date:
                df.loc[date,'Day'] = school_hols.loc[row,'School Holiday'] 
                
    for day in weekday_dictionary.keys():
        df.loc[df['Day']==day, 'Day'] = weekday_dictionary[day]            

    return df



def fill_na(df):
    df.fillna(0,inplace=True)
    print('To validate the null values were filled correctly, the remaining null values are:\n')
    display(df.isna().sum())
    df.columns = [x[1] for x in df.T.index]
    print('\nRenaming the columns to remove Quantity complete\n')
    return df





def add_weather(shop_df,weather_df, index='no'):
    for date in shop_df.index:
        for row in weather_df.index:
            if index == 'yes':
                if row == date:
                    shop_df.loc[date, ['TAVG','PRCP']] = weather_df.loc[row,['TAVG','PRCP']] 
            else:
                if weather_df.loc[row,'Date'] == date:
                    shop_df.loc[date, ['TAVG','PRCP']] = weather_df.loc[row,['TAVG','PRCP']] 

    return shop_df




def add_cpi(shop_df, cpi_df):
    for date in shop_df.index:
        for row in cpi_df.index:
            if cpi_df.loc[row,'Date'].month == date.month and cpi_df.loc[row,'Date'].year == date.year:
                shop_df.loc[date,'CPI_Electricity'] = cpi_df.loc[row,'CPI_Electricity']
                shop_df.loc[date,'CPI_Food'] = cpi_df.loc[row,'CPI_Food']
                
                


# Bakery EDA

def top_bottom_products(df, shop):
    top_10 = df['Name'].value_counts().head(10)
    print(f'The top 10 selling products at {shop} are: \n{top_10}')
    bottom_10 = df['Name'].value_counts().tail(10)
    print(f'\nThe bottom 10 selling products at {shop} are: \n{bottom_10}')
    
    
    
def plot_rev_lines(df,col,title,yaxis):
    revenue = df['Total_Revenue']
    # Set up the Figure 
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    variable = df[col]
    # Create the graph data objects
    chart_data1 = go.Scatter(x=df.index,y=revenue, 
                             name='Revenue')
    chart_data2 = go.Scatter(x=df.index,y=variable, 
                             name=col)
    # Use add_trace()
    fig.add_trace(chart_data1)
    fig.add_trace(chart_data2, secondary_y=True)
    # Create the layout object and add it to the figure
    chart_layout = go.Layout(title=title,
                            xaxis_title='Date', yaxis_title='Revenue £')
    fig.update_yaxes(title_text=yaxis, secondary_y=True)
    fig.update_layout(chart_layout)
    fig.show()
    

    
def plot_regplot(df, y, x, title, ylable, xlabel):
    sns.regplot(data=df, x=x,y=y, color=light_magpie)
    plt.title(title, fontsize=20)
    plt.ylabel(ylable, fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    
    
    
# Model 1 Moving Average

def rev_moving_average(df, window, new_col):
    df[new_col] = df['Total_Revenue'].rolling(window, closed='left').mean().round(2)
    
    
    
def moving_average_subplot(df, col1, col2, col3, plot_label, df_slice=0, window_length = 0):
    plt.subplots(3,1, figsize= (15,15))
    if df_slice == 0:
        index = 0
    else: 
        index = (len(df)-df_slice)
    
    plt.subplot(3,1,1)
    plt.plot(df.index[index:], df['Total_Revenue'][index:], color = light_magpie, label = plot_label,
               linewidth=1)
    if window_length > 0:
        plt.plot(df.index[(len(df)-window_length):], df[col1][(len(df)-window_length):], color = red_magpie,
                 label = col1, linewidth=2, linestyle = 'dashed')
    else:
        plt.plot(df.index, df[col1], color = red_magpie, label = col1, linewidth=2)
    plt.ylabel('Revenue (£)', fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Date', fontsize=15)
    plt.xticks(fontsize=15)
    plt.legend(fontsize=10)
    plt.title(f'{col1} Revenue', fontsize=20)

    plt.subplot(3,1,2)
    plt.plot(df.index[index:], df['Total_Revenue'][index:], color = light_magpie, label = plot_label,
               linewidth=1)
    if window_length > 0:
        plt.plot(df.index[(len(df)-window_length):], df[col2][(len(df)-window_length):], color = red_magpie,
                 label = col2, linewidth=2, linestyle = 'dashed')
    else:
        plt.plot(df.index, df[col3], color = red_magpie, label = col3, linewidth=2)
    plt.ylabel('Revenue (£)', fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Date', fontsize=15)
    plt.xticks(fontsize=15)
    plt.legend(fontsize=10)
    plt.title(f'{col2} Revenue', fontsize=20)

    plt.subplot(3,1,3)
    plt.plot(df.index[index:], df['Total_Revenue'][index:], color = light_magpie, label = plot_label,
               linewidth=1)
    if window_length > 0:
        plt.plot(df.index[(len(df)-window_length):], df[col3][(len(df)-window_length):], color = red_magpie,
                 label = col3, linewidth=2, linestyle = 'dashed')
    else:
        plt.plot(df.index, df[col3], color = red_magpie, label = col3, linewidth=2)
    plt.ylabel('Revenue (£)', fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Date', fontsize=15)
    plt.xticks(fontsize=15)
    plt.legend(fontsize=10)
    plt.title(f'{col3} Revenue', fontsize=20)

    plt.tight_layout()
    plt.show()
    
    
def predict_rev_ma(train_test_df, ma_window, prediction_length, new_col, index_type):
    
    train_test_df[new_col]=0
    predict_date_index = len(train_test_df)-prediction_length
    predict_dates = train_test_df.index[predict_date_index:]
    counter_total_av = ma_window
    counter_pred_ma = 0
    
    for i in range(0,prediction_length):
        ma_revenue = 0
        current_row = train_test_df.index[predict_date_index+i]
        for j in range(ma_window,counter_pred_ma,-1):
            if index_type == 'Day':
                total_rev_col_dates = current_row - DateOffset(days=j)
            elif index_type=='Week':
                total_rev_col_dates = current_row - DateOffset(weeks=j)
            else:
                total_rev_col_dates = current_row - DateOffset(months=j)
            ma_revenue += train_test_df.loc[total_rev_col_dates, 'Total_Revenue']   
        for k in range(counter_pred_ma+1):
            if index_type == 'Day':
                total_rev_col_dates = current_row - DateOffset(days=k)
            elif index_type=='Week':
                total_rev_col_dates = current_row - DateOffset(weeks=k)
            else:
                total_rev_col_dates = current_row - DateOffset(months=k)
            ma_revenue += train_test_df.loc[total_rev_col_dates, new_col]
        train_test_df.loc[current_row, new_col] = round(ma_revenue/ma_window,2)
        
        if counter_total_av!=0:
            counter_total_av -= 1
        else: 
            counter_total_av=0
        if counter_pred_ma != ma_window:
            counter_pred_ma += 1
        else:
            counter_pred_ma = ma_window
        
    
    
def revenue_difference(df, ma_col, new_col):
    df[new_col] = df['Total_Revenue'] - df[ma_col]
    
def revenue_difference_plots(df, col1, col2, col3, width=0.8):
    plt.subplots(1,3, figsize=(30,10))
    plt.subplot(1,3,1)
    plt.bar(x=df.index,height=df[col1], color=dark_magpie, width=width)
    plt.title(f'Moving Average {col1}', fontsize=30, pad=10)
    plt.xticks(fontsize=25,rotation=25)
    plt.xlabel('Date', fontsize=25)
    plt.ylabel('Revenue Difference (£)', fontsize=25)
    plt.yticks(fontsize=25)

    plt.subplot(1,3,2)
    plt.bar(x=df.index,height=df[col2], color=medium_magpie, width=width)
    plt.title(f'Moving Average {col2}', fontsize=30, pad=10)
    plt.xticks(fontsize=25,rotation=25)
    plt.xlabel('Date', fontsize=25)
    plt.ylabel('Revenue Difference (£)', fontsize=25)
    plt.yticks(fontsize=25)

    plt.subplot(1,3,3)
    plt.bar(x=df.index,height=df[col3], color=light_magpie, width=width)
    plt.title(f'Moving Average {col3}', fontsize=30, pad=10)
    plt.xticks(fontsize=25,rotation=25)
    plt.ylabel('Revenue Difference (£)', fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('Date', fontsize=25)
    
    plt.tight_layout(w_pad=5)
    plt.show()
    

def mean_absolute_error(true_values, predicted_values, forecast_name):
    """
    Calculate the mean absolute error. 
    """
    # calculate MAE
    error = round(mae(true_values, predicted_values),2)
  
    print(f'The MAE between the actual and {forecast_name} revenue is {str(error)}')    
    
    
def mean_absolute_percentage_error_calc(true_values, predicted_values, forecast_name):
    """
    Calculate the mean absolute percentage error. 
    The prediction error is determined and divide by the true value, then averaged.
    """
    mape_score = mean_absolute_percentage_error(true_values, predicted_values)
    mape = round(mape_score*100,1)
    
    print(f'The MAPE between the actual and {forecast_name} revenue is {mape}%')
    
def rmse(actual, predicted, forecast_name):
    mse = sklearn.metrics.mean_squared_error(actual, predicted)  
    rmse = math.sqrt(mse)  

    print(f'The RMSE for the {forecast_name} is: {round(rmse,2)}')  
    

def full_accuracy_report(true_values, predicted_values, forecast_name):
    rmse(true_values, predicted_values, forecast_name)
    print('')
    mean_absolute_error(true_values, predicted_values, forecast_name)
    print('')
    mean_absolute_percentage_error_calc(true_values, predicted_values, forecast_name)



    


# linear regression model 

def full_lreg(X_train, y_train, X_test, train_df, test_df, title, y_test, forecast_label):
    print('Fitting linear regression model.\n')
    y = y_train
    # First the X and y values need to splitted: 
    X = X_train
    # Then the X_ constant needs to be added to give the intercept value
    X_withconstant = sm.add_constant(X)
    # The model is then instantiated using the sm.OLS method
    lm_revenue = sm.OLS(y,X_withconstant)
    # The model is fitted using the .fit() method
    lm_revenue_results = lm_revenue.fit()
    # The results are displayed using the .summary() method
    display(lm_revenue_results.summary())

    print('Making prediction with fitted model.\n')
    # Setting up X test with constant 
    X_test_withconstant = sm.add_constant(X_test)
    
    # Predictions
    prediction = lm_revenue_results.predict(X_test_withconstant)
    
    print('Plotting predicted results.\n')
    train_forecast_plot(train_df, test_df, prediction, title)
    
    print('The accuracy scores for the model are:\n')
    
    full_accuracy_report(y_test, prediction, forecast_label)


def add_results(results_df, model, r2, RMSE, AIC, MAPE, comments):
    """
    Function to add the rows entered into the model results dataframe and returns and displays the inputted results df.
    
    Inputs:
    results_df (dataframe): dataframe the new results will be appended to
    model (int): the model number that will be entered in the 'Model' column in the results df
    r2 (float) : the R2 value will be entered in the 'R2' column in the results df
    RMSE (float) : the RMSE value will be entered in the 'RMSE' column in the results df
    AIC (float): The AIC value of the model, will be added to 'AIC' column in resutls df 
    MAPE (string) : The MAPE score of the modek will be added to the 'MAPE (%)' columns in the results df
    comments (string): The list of columns / features used or changed in the model 
    
    Output:
    A new row in the results df containing the model, test score, hyperparameters, scaler and additional features used
    """
    results_dictionary = {'Model':model,
                    'R2': r2,
                    'RMSE':RMSE,
                         'AIC': AIC,
                         'MAPE (%)': MAPE,
                         'Comments':comments}
    results_df = results_df.append(results_dictionary, ignore_index=True)
    display(results_df)
    return results_df



def vif_func(X):
    """
    This function calculates the variance inflation factor of a given dataframe.

    Inputs:
    df (dataframe): dataframe containing the data for analysis
    
    """
    display(pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns))
    
    
    
# ARIMA 

# evaluate the SARIMAX model for a given order (p,d,q)
def evaluate_sarimax_model(X, arima_order, exog_data, param_seasonal, split_size):
    # prepare training and validation dataset
    train_size = int(len(X) * split_size)
    train, test = X[0:train_size], X[train_size:]

    # Test if there is exogenous data or not 
    if type(exog_data) == type(X):
        exog_size = int(len(exog_data) * split_size)
        ex_train, ex_test = exog_data[0:exog_size], exog_data[exog_size:]
            # fit model
        model = SARIMAX(train, exog=ex_train, order=arima_order, seasonal_order = param_seasonal, trend='c')
        model_fit = model.fit(disp=0)
        #predict
        predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, exog=ex_test)
        mape = mean_absolute_percentage_error(test, predictions)
        return mape
    
    else:    
        # To fit model
        model = SARIMAX(train, order=arima_order, seasonal_order = param_seasonal, trend='c')
        model_fit = model.fit(disp=0)
        #predict
        predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
        mape = mean_absolute_percentage_error(test, predictions)
        return mape
    


# To evaluate combinations of p, d and q values for the SARIMAX model
def evaluate_models(dataset, p_values, d_values, q_values, exog_data, s_range, s_order, split_size):
    dataset = dataset.astype('float32')
    best_mape, best_order, best_s_order = float('inf'), None, None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                seasonal_pdq = [(x[0], x[1], x[2], s_order) for x in list(itertools.product(s_range, s_range, s_range))]
                for param_seasonal in seasonal_pdq:
                    try:
                        mape = evaluate_sarimax_model(dataset, order, exog_data, param_seasonal, split_size)
                        if mape < best_mape:
                            best_mape, best_order, best_s_order = mape, order, param_seasonal
#                         print(f'SARIMAX{order} SEASONAL{param_seasonal} MAPE={round(mape,3)}')
                    except:
                        continue    
                    
    print(f'The Best SARIMAX {best_order} SEASONAL{best_s_order} MAPE={round(best_mape,3)}')
    
    
    
def train_forecast_plot(train_df, test_df, predictions, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Total_Revenue'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Total_Revenue'], mode='lines', name='Test', line={'color':red_magpie}))
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name='Predictions'))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        yaxis_title='Revenue (£)', 
        xaxis_title='Date',
        title=title
    )
    fig.show()    
    


    
# Market basket analysis
def transaction_encoder(df):
    df = df.groupby('TransactionId')['ProductName'].agg(['unique'])
    df = df['unique'].values.tolist()
    te = TransactionEncoder()
    df = te.fit_transform(df)
    df = pd.DataFrame(df, columns=te.columns_)
    return df