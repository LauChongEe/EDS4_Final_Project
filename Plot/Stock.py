from numpy.random import random
import numpy as np
import pandas as pd
import urllib.request
import os
import time
import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler


from bokeh.io import curdoc, output_file
from bokeh.layouts import column, row, widgetbox
from bokeh.plotting import ColumnDataSource, Figure
from bokeh.models.widgets import Select, TextInput, Slider, Button, Panel, Tabs, DataTable, TableColumn, NumberFormatter, RadioGroup
from bokeh.models import Div
import yahoo_quote_download as yqd

#analysis target
current_key_stats = datetime.datetime(2018, 2, 5, 0, 0)
#how_much_better = 20
#invest_unit_per_stock = 1

#analysis features
Analysis_Features = [
                     'Market Cap',
                     'Enterprise Value',
                     'Trailing P/E',
                     'Forward P/E',
                     'PEG Ratio',
                     'Price/Sales',
                     'Enterprise Value/Revenue',
                     'Profit Margin',
                     'Operating Margin', 
                     'Return on Equity',
                     'Revenue',
                     'Revenue Per Share',                         
                     'Net Income Avl to Common',
                     'Diluted EPS',
                     'Total Cash',
                     'Total Cash Per Share',
                     'Total Debt',
                     'Book Value Per Share',
                     'Operating Cash Flow',
                     'Beta',
                     '% Held by Insiders',
                     '% Held by Institutions',
                     'Shares Short', 
                     'Short Ratio'
                    ]  

#Stock Ranking					
#define function to rank stock based on differences between stock price changes & S&P 500 index changes
def Rank_Stock(row):  		
	how_much_better = int(sl_howmuch_gain.value)
	if row['Stock Price vs S&P 500 Index'] > how_much_better:
		#print(row['Ticker'], row['Stock Price vs S&P 500 Index'])
		return 1
	else:
		return 0

#Data Load
#define function to load historical data
def Load_Historical_Data():
    historical_data_path = os.path.join(os.path.join(os.getcwd(), 'Data'), "Historical.csv")
    df = pd.read_csv(historical_data_path, sep=',')
    return df

#define function to load forward data
def Load_Forward_Data(sel_date):
	sel_date = sel_date.strftime('%Y%m%d')
	forward_data_path = os.path.join(os.path.join(os.getcwd(), 'Data'), "Forward_" + sel_date + ".csv")
	df = pd.read_csv(forward_data_path, sep=',')
	return df

#Load Yahoo Finance stock price
def load_quote(ticker, startDate, endDate, search_method='F', count=0):
    count+=1        
    start = startDate.strftime('%Y%m%d')
    end = endDate.strftime('%Y%m%d')    
    data = yqd.load_yahoo_quote(ticker, start, end)   
	#print('===', ticker, start, end, search_method, count, data,'===')
    df = pd.DataFrame([sub.split(",") for sub in data if len(sub.split(",")[0])>0 and sub.split(",")[0] != 'Date'], columns=['Date','Open','High','Low','Close','Adj Close','Volume'])    
      
    if len(df) == 0 and count < 10:
        value = None
        if search_method=='F':
            startDate = startDate + timedelta(days=1)
            endDate = endDate + timedelta(days=1)
        else:
            startDate = startDate - timedelta(days=1)
            endDate = endDate - timedelta(days=1)
            
        return load_quote(ticker, startDate, endDate, search_method, count)
    else:        
        value = df.iloc[0]['Adj Close'] if len(df) > 0 else None
    return value
	
#Check for NaN value in data
def Check_Null_Data(df):
    #check for NaN data   
    #as there're a lot of N/A data, cautious approach will be implemented for stock analysis to follow rules of:
    #1) Avoid loss 
    #2) Do not forget rule#1
    #All N/A data found in key features used for analysis will be removed.
    return df.isnull().sum() 

#Data cleaning and null value handling
#define function to preprocess data
def Historical_Data_Preprocessing_All_Industry(df):     
    #drop row with any NA value based on Analysis Features + Stock Price 
    df = df.dropna(subset=['Stock Price']+Analysis_Features, how='any').reset_index().drop(['index'], axis=1)  
        
    #features data set
    X = df[Analysis_Features]  
    
    #label data set    
    df['Ranking'] = df.apply(Rank_Stock, axis=1)
    y = df['Ranking']
    
    return X,y

#define function to preprocess data
def Forward_Data_Preprocessing_All_Industry(df): 
    #drop row with any NA value based on Analysis Features + Stock Price 
    df = df.dropna(subset=['Ticker','Security','GICS Sector','GICS Sub Industry']+Analysis_Features, how='any').reset_index().drop(['index'], axis=1)
       
    #features data set
    X = df[Analysis_Features]
    
    #Listed company data set 
    z = df[['Ticker','Security','GICS Sector','GICS Sub Industry']]    
        
    return X,z
	
#Train and Test Dataset Split
def Train_Test_Split(X, y):
    #For the sake of testing our classifier output, we will split the data into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #X_columns = list(X_train)
    return X_train, X_test, y_train, y_test	

#Feature Scaling and Normalization
def Standard_Scale(X_train, X_test, X_ver):
    scaler = StandardScaler()
    #standard scale transform
    X_train_std_scale = scaler.fit_transform(X_train)
    X_test_std_scale = scaler.transform(X_test)
    X_ver_std_scale = scaler.transform(X_ver)
    return X_train_std_scale, X_test_std_scale, X_ver_std_scale

def MinMax_Scale(X_train, X_test, X_ver):
    scaler = MinMaxScaler()
    #MinMax scale transform
    X_train_mm_scale = scaler.fit_transform(X_train)
    X_test_mm_scale = scaler.transform(X_test)
    X_ver_mm_scale = scaler.transform(X_ver)
    return X_train_mm_scale, X_test_mm_scale, X_ver_mm_scale

#PCA - choosing number of components
def PCA_Component_Selection(n_components, X_train, X_test, X_ver):
    #choose 15 components for PCA to reduce features
    pca = PCA(n_components=n_components)

    X_train_PCA = pca.fit_transform(X_train)
    X_test_PCA = pca.transform(X_test)
    X_ver_PCA = pca.transform(X_ver)
    return X_train_PCA, X_test_PCA, X_ver_PCA

#Imbalance dataset handling
def Down_Sample_Dataset(X, y):  
    rus = RandomUnderSampler(return_indices=True, random_state=42)
    X_undersampling, y_undersampling, idx_resampled = rus.fit_sample(X, y)
    print('Y before down-sampling:', np.unique(y,return_counts=True))
    print('Y after down-sampling:', np.unique(y_undersampling,return_counts=True))
    return X_undersampling, y_undersampling

#Model Execution with Random Forest
def RF_Run(X_train, y_train, X_test, y_test, X_ver, z, calc_return=True):
    #Random Forest 
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=30, max_depth=5, min_samples_leaf=6).fit(X_train, y_train)
    
    # Random Forest Default accuracy 
    print('Random Forest Default (accuracy) Train Dataset', model.score(X_train, y_train))
    print('Random Forest Default (accuracy) Test Dataset', model.score(X_test, y_test))
    print()

    # Random Forest Default metric report
    from sklearn import metrics
    y_pred = model.predict(X_test)
    print('Random Forest Default Metric Report:')
    print(metrics.classification_report(y_pred, y_test))
    
    # Calculate AUROC
    from sklearn.metrics import roc_auc_score
    y_prob = model.predict_proba(X_test)
    y_prob = [p[1] for p in y_prob]    

    #test with latest stock key statistic
    import datetime
    print('Predict using latest Key Statistic @ {0}'.format(current_key_stats.strftime('%Y-%m-%d')))    
    y_pred = model.predict(X_ver)

	
    counter=0
    invest_list = []
    initial_price = []
    current_price = []
    investment_cost = []
    investment_return = []
    profit_loss = []
    for t in y_pred:   
         if t == 1:                
            invest_list.append(z.iloc[counter]['Security'])            
            if calc_return == True:        
                ini_price = load_quote(z.iloc[counter]['Ticker'],current_key_stats,current_key_stats, search_method='F')        
                cur_price = load_quote(z.iloc[counter]['Ticker'],datetime.date.today()-timedelta(days=1),datetime.date.today()-timedelta(days=1), search_method='B')  
                if ini_price != None:
                    ini_price = round(float(ini_price),2)
                    inv_cost = round(float(ini_price)*int(in_inv_unit.value),2)
                else:
                    inv_cost = None
                if cur_price != None:
                    cur_price = round(float(cur_price),2)
                    inv_return = round(float(cur_price)*int(in_inv_unit.value),2)
                else:
                    inv_return = None
                if inv_cost != None and inv_return != None:
                    pl = round(inv_return-inv_cost,2)
                else:
                    pl = None
                
                initial_price.append(ini_price)
                current_price.append(cur_price)
                investment_cost.append(inv_cost)
                investment_return.append(inv_return)
                profit_loss.append(pl)
         counter+=1
    
    print()
    if calc_return == True:
        df_rcmd = pd.DataFrame({
                                'Company':invest_list,
                                'Initial Price':initial_price,
                                'Current Price':current_price,
                                'Investment Cost':investment_cost,
                                'Investment Return':investment_return,
                                'Profit/Loss':profit_loss
                                }
                                , columns=['Company',
                                           'Initial Price',
                                           'Current Price',
                                           'Investment Cost',
                                           'Investment Return',
                                           'Profit/Loss'])
        print('Recommended companies for investment and total return as below:')
        print('Total earning from [{0}] till [{1}] is [${2}]'.format(current_key_stats.strftime('%Y-%m-%d'), datetime.datetime.today().strftime('%Y-%m-%d'), round(df_rcmd['Profit/Loss'].sum(),2)))
        print('Total investment cost so far is [${0}]'.format(round(df_rcmd['Investment Cost'].sum(),2)))
        print('Total investment return so far is [${0}]'.format(round(df_rcmd['Investment Return'].sum(),2)))
        print('Total investment return % gain so far is [{0}%]'.format(100*round(df_rcmd['Profit/Loss'].sum(),2)/round(df_rcmd['Investment Cost'].sum(),2)))
        print()
        df_top10 = df_rcmd.nlargest(10, 'Profit/Loss').reset_index().drop(['index'],axis=1)
        print('Top 10 companies by investment profit:')
        print('Top 10 Total earning from [{0}] till [{1}] is [${2}]'.format(current_key_stats.strftime('%Y-%m-%d'), datetime.datetime.today().strftime('%Y-%m-%d'), round(df_top10['Profit/Loss'].sum(),2)))
        print('Top 10 Total investment cost so far is [${0}]'.format(round(df_top10['Investment Cost'].sum(),2)))
        print('Top 10 Total investment return so far is [${0}]'.format(round(df_top10['Investment Return'].sum(),2)))
        print('Top 10 Total investment return % gain so far is [{0}%]'.format(100*round(df_top10['Profit/Loss'].sum(),2)/round(df_top10['Investment Cost'].sum(),2)))        
        print(df_top10)
    else:
        print('Recommended companies for investment as below:')
        df_rcmd = pd.DataFrame(invest_list, columns=['Company'])   
    
    return df_rcmd 

def get_KeyStat_Date():	
	return None

def get_data():	
	return None

def Analysis():
	try:	       
		#load historical & current Key Financial Statistic data
		df_historical = Load_Historical_Data()
		df_forward = Load_Forward_Data(current_key_stats)

		#Historical & Forward null value handling	
		X,y = Historical_Data_Preprocessing_All_Industry(df_historical)
		X_ver,z = Forward_Data_Preprocessing_All_Industry(df_forward)

		#split historical data into Train & Test set
		X_train, X_test, y_train, y_test = Train_Test_Split(X, y)

		#Feature Scaling and Normalization for Train, Test & Verification dataset
		X_train_mm_scale, X_test_mm_scale, X_ver_mm_scale = MinMax_Scale(X_train, X_test, X_ver)

		#check if there's any need for PCA to reduce number of features		
		X_train_mm_scale_PCA, X_test_mm_scale_PCA, X_ver_mm_scale_PCA = PCA_Component_Selection(20, X_train_mm_scale, X_test_mm_scale, X_ver_mm_scale)

		#check if there's imbalance dataset?
		print('Y train:',np.unique(y_train,return_counts=True))
		print('Y test:',np.unique(y_test,return_counts=True))

		#handling imbalance dataset by down-sampling the train set data
		X_train_mm_scale_PCA_reduce, y_train_reduce = Down_Sample_Dataset(X_train_mm_scale_PCA, y_train)

		if rd_load_price.active ==0:
			calc_return = False
		else:	
			calc_return = True
			
		#model execution with GridSearch Random Forest
		df_result = RF_Run(X_train_mm_scale_PCA_reduce, y_train_reduce, X_test_mm_scale_PCA, y_test, X_ver_mm_scale_PCA, z, calc_return=calc_return)
		#df_result.to_csv('GridSearch_RF_mm_Scale_PCA_Down_Sampling.csv', index=False)
		#df_result
		
		if calc_return==False:
			data_companies.data = {
				'Company': df_result['Company']
			}
		else:
			df_result.sort_values(by='Profit/Loss', ascending=False)
			data_companies.data = {
				'Company': df_result['Company'],
				'Initial Price': df_result['Initial Price'],
				'Current Price': df_result['Current Price'],
				'Investment Cost': df_result['Investment Cost'],
				'Investment Return': df_result['Investment Return'],
				'Profit/Loss': df_result['Profit/Loss']
			}
	except Exception as e:
		print(str(e))  
	return None

#Data for companies
data_companies = ColumnDataSource(data=dict())
columns_companies = [
    TableColumn(field="Company", title="Company"),
    TableColumn(field="Initial Price", title="Initial Price", formatter=NumberFormatter(format="$0,0.00")),
    TableColumn(field="Current Price", title="Current Price", formatter=NumberFormatter(format="$0,0.00")),
	TableColumn(field="Investment Cost", title="Investment Cost", formatter=NumberFormatter(format="$0,0.00")),
	TableColumn(field="Investment Return", title="Investment Return", formatter=NumberFormatter(format="$0,0.00")),
	TableColumn(field="Profit/Loss", title="Profit/Loss", formatter=NumberFormatter(format="$0,0.00"))
]
data_table_companies = DataTable(source=data_companies, columns=columns_companies, width=800)

#controls
#How much gain compare to S&P 500 index price
sl_howmuch_gain = Slider(start=0, end=20, step=1, value=5, title="How much better compared to S&P 500 index?")
#Investment Unit
in_inv_unit = TextInput(title="Investment Unit", value="1")
#load yahoo finance price
rd_load_price = RadioGroup(labels=["Do not load stock price", "To load stock price"], active=0)
#Prediction Go
bt_predict = Button(label="Go")

#add callback to widget
def Go():
	Analysis()	
bt_predict.on_click(Go)


controls = widgetbox(Div(text="<h1>Good Stock Recommendation</h1>"), sl_howmuch_gain, in_inv_unit, rd_load_price, bt_predict)
table_companies = widgetbox(data_table_companies)

layout = row(column(controls), 
			 column(
					row(Div(text="<h2>List of S&P 500 Stocks</h2>")), row(table_companies,width=900)
					)
			)

curdoc().add_root(layout)