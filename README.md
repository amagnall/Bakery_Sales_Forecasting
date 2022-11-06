# Forecasting Bakery Sales 
This is my final capstone project submission for BrainStation's Data Science certificate program. The project is utilising various timeseries methods to forecast revenue of a bakery and developing a reporting dashboard summarising the best selling products and periodic trends. 
___

Hello!

My name is Abi Magnall and this is my capstone project submission for BrainStation's Data Science certificate program titled, “Forecasting Bakery Sales".


**Purpose of this project**: 

The aim of this project is to develop Moving Averages, Linear Regression, SARIMAX and Facebook Prophet models to forecast daily, weekly and monthly revenue for the amazing Artisan Bakery, [Two Magpies Bakery](https://twomagpiesbakery.co.uk/). Market Basket Analysis is also performed to provide recommendations and provide greater clarity to the business on the most popular, complimentary and subsitute products. Through various methods of data visualization and communication the findings and actionable insights are conveyed.


The notebooks in order are as follows with descriptions about their contents:


|Order |Notebook |  Description|
|:--| :- |  :-|
|1| 1_Bakery_Data_Gathering |  This notebook defines the problem statement and makes API calls to download the bakery data |               
|2| 2_Weather_Cleaning_EDA| This notebook explored and cleans the weather dataset |         
|3| 3_Bakery_Data_Cleaning | All the required cleaning steps for the raw bakery data| 
|4| 4_Bakery_Data_Preprocessing | The preprocessing steps required to merge and transform datasets into the correct format for timeseries |        
|5| 5_Bakery_Data_EDA | Exploratory Data Analysis into the clean and preprocessed bakery transaction data | 
|6| 6_Model_1_Moving_Average_Model| The steps taken for the baseline moving average model predictions | 
|7| 7_Model_2_Linear_Regression | The steps taken to develop the Linear Regression models |   
|8| 8_Model_3_SARIMAX | The steps taken to develop the SARIMAX models |        
|9| 9_Model_4_Facebook_Prophet| The steps taken to develop the Facebook Prophet models| 
|10| 10_Modelling_Summary| A notebook discussing all the models built, their results and a final test accuracy on new unseen data| 
|11| 11_Market_Basket_Analysis| Market Basket Analysis for each of the four bakeries | 


The folders in this file are required:

1. ‘models’ - this contains all .pkl files that are necessary to run the 10_Modelling_Summary notebook
2. requirement.txt - contains information on the bakery_environment required to run the notebooks
3. .ipynb files - all the notebooks 
