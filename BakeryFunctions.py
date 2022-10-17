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


# Variables 

dark_magpie = '#00313B'

light_magpie = '#5489CB'


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
    chart_layout = go.Layout(width=800,height=600, title=title,
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
    


