"""
@author:
"""

# libraries
import pandas as pd
import numpy as np
from datetime import datetime
import pyodbc
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.ensemble import IsolationForest
import shap
import warnings 
warnings.filterwarnings('ignore')
import gc
gc.collect()

# functions
def establish_con(server,database, user, pwd, env):
    '''function to establish connection with prodcution database:EU'''
    cnxn = pyodbc.connect('Driver={SQL Server};'
    #'Server=AGNConnectusazrsqlp1011.database.windows.net;'
    f'Server={server};'
    f'Database={database};'
    f'UID={user};'
    f'PWD={pwd};'
    )
    print(f'connection established to db:{database}, EU:{env}')

    return cnxn


def close_connection(cur, cnxn):
    cur.close()
    cnxn.close()
    print('SQL connection closed')

def prepare_data(df_main, obs_end_date):
    '''
    1. renaming columns
    2. Aggregating SALES AT INVOICE LEVEL
    '''
    
    print(df_main.shape)
    # 3. Segregating Data (in case if we are not using LAST BILL DATE as OBS_END_DATE)
    df_raw = df_main[df_main['BILL_DATE_YYYY_MM_DD']<=obs_end_date]
      
    # 2.Renaming column names
    df_raw.rename(columns={'SAP_NET_SALES_AMOUNT_USD':'NET_SALES_AMOUNT',
                           'SAP_NET_SALES_C_UNITS':'NET_SALES_UNITS'}, inplace=True)
    
    # 4. Agrregating Sales
    df_raw = df_raw.sort_values(by=['BILL_NUMBER','BILL_DATE_YYYY_MM_DD'])
    df_raw = df_raw.groupby(['COUNTRY','CUSTOMER_ID','BILL_NUMBER']).agg({'BILL_DATE_YYYY_MM_DD':[lambda x: x.unique()[-1], 
                                                                                       lambda x: x.nunique()],
                                                                          'NET_SALES_AMOUNT':['sum'],
                                                                          'NET_SALES_UNITS' :['sum'],
                                                                         }).reset_index()
    # Note: If one bill number has more than one Bill date,  
    # picking up the latest bill date as per mariana suggestion
    df_raw.columns = ['COUNTRY','CUSTOMER_ID','BILL_NUMBER','BILL_DATE_YYYY_MM_DD','unique_BILL_DATE','NET_SALES_AMOUNT','NET_SALES_UNITS']
    print(f"Dataframe shape after aggregating at Invoice Level: {df_raw.shape}")
    print('dropping Negative and Zero Sales after aggregating Sales Amount at Invoice(BILL_NUMBER) level')
    df_raw_filtered = df_raw[df_raw['NET_SALES_AMOUNT']>0]
    
    # after dropping negative and Zero Sales
    # summing up Sales at Bill date level
    df_raw_filtered2 = df_raw_filtered.groupby(['COUNTRY','CUSTOMER_ID','BILL_DATE_YYYY_MM_DD']).agg({'NET_SALES_AMOUNT':'sum',
                                                                                          'NET_SALES_UNITS':['sum'],
                                                                                          }).reset_index()
    
    df_raw_filtered2.columns = ['COUNTRY','CUSTOMER_ID','BILL_DATE_YYYY_MM_DD','NET_SALES_AMOUNT','NET_SALES_UNITS']
    
    print(f'Dataframe Shape Before dropping Negative and Zero Sales:{df_raw.shape}')
    print(f'Dataframe Shape After dropping Negative and Zero Sales:{df_raw_filtered.shape}')  
    print(f'Total Invoices(BillNumbers) dropped with Negative and Zero Sale Amount: {df_raw.shape[0]-df_raw_filtered.shape[0]}')
    
    print(f"Dataframe Shape aggregating Sales at BILL_DATE level:{df_raw_filtered2.shape}")
    print(f"Min Bill Date: {df_raw_filtered2['BILL_DATE_YYYY_MM_DD'].min()}")
    print(f"Max Bill Date: {df_raw_filtered2['BILL_DATE_YYYY_MM_DD'].max()}")
          
    return df_raw_filtered2

def remove_outlier(abc):
    
    
    try:
        
        global counter
        counter+=1
      
        cid   = abc['CUSTOMER_ID'].unique()[0]
        clf   = IsolationForest(random_state = 42, contamination=0.01)
        preds = clf.fit_predict(abc[['INTER_PURCHASE_DAYS']].dropna())
        v  = list(abc['INTER_PURCHASE_DAYS'].dropna())
        p  = list(preds)
        df_abc = pd.DataFrame(zip(v,p),columns=['v','p'])
        outlier_list = list(df_abc['v'][df_abc['p']==-1])
        print(f'{counter}.{cid} - {outlier_list}')
        abc = abc.loc[~abc['INTER_PURCHASE_DAYS'].isin(outlier_list)]
        return abc
        
    except:
        print(f'{counter}.{cid}: Empty Dataframe')
    
               
   
def calculate_percentile(df, p):
    ''' 
    Calculating 90th Percentile of INTER_PURCHASE_DAYS
    with respect to each customer
    '''
    
    # dict_pp = {}
    cid_list = []
    pp_list  = []
    print(f'calculating {p}th percentile of Inter purchase days')
    
    for idx, cid in enumerate(df['CUSTOMER_ID'].unique()):
        
        ecdf = ECDF(df[df['CUSTOMER_ID']==cid]['INTER_PURCHASE_DAYS'].dropna())
        pp   = np.percentile(ecdf.x, p)
        cid_list.append(cid)
        pp_list.append(pp)
        
    df_pp = pd.DataFrame(zip(cid_list,pp_list))
    df_pp.columns = ['CUSTOMER_ID',f'PERCENTILE_{p}']
    
    # merge
    df = df.merge(df_pp, on='CUSTOMER_ID', how='left')
    
    return df

def freq_exceed(df,col):
    p = col[-2:]
    df_temp = df[df['INTER_PURCHASE_DAYS']>float(df[col].unique())]
    print(df_temp['CUSTOMER_ID'].unique())
    df[f'FREQ_AFTER_P{p}'] = df_temp['INTER_PURCHASE_DAYS'].nunique()
    return df

# Labeling Target
# def label_class(df,p):
#        df['STATUS'] = np.nan
#        for idx, cid in enumerate(df['CUSTOMER_ID']):
#            print(idx, cid)
#            lss  = df['LAST_SEEN_SINCE_IN_DAYS'].loc[idx]
#            p90  = df[f'PERCENTILE_{p}'].loc[idx]
#            p100 = df['PERCENTILE_100'].loc[idx]
            
#            if lss < p90:
#                df['STATUS'].loc[idx] = 'NON CHURN'  
#            elif lss > p100:
#                df['STATUS'].loc[idx] = 'CHURNED'     
#            elif p90 <= lss <= p100:
#                df['STATUS'].loc[idx] = 'RISK OF CHURN'
#        return df

# bucket

# bucket
def bucket(df, p, B):
    df[B]= np.nan
    for idx, v in enumerate(df[p]):
        v = int(v)
        print(f'{idx}-{v}')
        if v<=30:
            df[B].loc[idx]='30Days'
        elif (v >= 31) & (v <=60):
            df[B].loc[idx]='60Days'
        elif (v >= 61) & (v <=90):
            df[B].loc[idx]='90Days'
        elif (v >= 91) & (v <=120):
            df[B].loc[idx]='120Days'
        elif( v >= 121) & (v <=150):
            df[B].loc[idx]='150Days'
        elif( v >= 151) & (v <=180):
            df[B].loc[idx]='180Days'
        else :
            df[B].loc[idx]='6+Months'
        
    return df

def pos(col):
       '''MRP'''
       return col[col > 0].sum()
  
def neg(col):
    '''DISCOUNT'''
    return col[col < 0].sum()

def sort_val(col):
    ''' function to sort values in a column
    in ascending order'''
    return ','.join(str(x)for x in (sorted(col.unique().tolist())))
    
def monthly_sale_breakup(df_raw):
    
        # MONTHLY AMOUNT BREAKUP 
        df_raw2 = df_raw.copy()
        
        # creating MONTH and YEAR Column
        df_raw2['MONTH'] = df_raw2['BILL_DATE_YYYY_MM_DD'].dt.month
        df_raw2['YEAR']  = df_raw2['BILL_DATE_YYYY_MM_DD'].dt.year
        
        # observation End date
        # obs_end_date = df_raw['BILL_DATE_YYYY_MM_DD'].max()
        obs_end_date = pd.to_datetime('2022-06-30')
        print(f'Last Bill Date Available in Dataset: {obs_end_date}')
        
        # getting total days of a month
        cur_year      = obs_end_date.year
        cur_month     = obs_end_date.month
        cur_day       = obs_end_date.day
    
        cur_period    = pd.Period(str(obs_end_date))
        curYear_month_totaldays = cur_period.daysinmonth
    
        obs_end_date2 = pd.to_datetime(f'{cur_year}-{cur_month}-{curYear_month_totaldays}')
        print(f'Observation End date:{obs_end_date2}')
    
        # last 12 months from observation End date
        last_12month_from_obs = obs_end_date2 + timedelta(days=-365)
        
        # below code is to take care FEBRUARY MONTH Cacluation
        prev_year         = last_12month_from_obs.year
        prev_obs_end_date = pd.to_datetime(f'{prev_year}-{cur_month}-{cur_day}')                         
        prev_period       = pd.Period(str(prev_obs_end_date))
        prevYear_month_totaldays = prev_period.daysinmonth
    
        last_12month_from_obs2 = pd.to_datetime(f'{prev_year}-{cur_month}-{prevYear_month_totaldays}')
        print(f'Last 12 Months Date from Observation End Date:{last_12month_from_obs2}')
    
    
        # taking last one year data from observation date and calculating last 12month total sale
        df_raw2 = df_raw2[df_raw2['BILL_DATE_YYYY_MM_DD']>last_12month_from_obs2]
        print(f"Min Date: {df_raw2['BILL_DATE_YYYY_MM_DD'].min()}")
        print(f"Max Date: {df_raw2['BILL_DATE_YYYY_MM_DD'].max()}")
    
        df_sales =  df_raw2.groupby('CUSTOMER_ID').agg({'NET_SALES_AMOUNT':'sum'}).reset_index()
        df_sales.columns = ['CUSTOMER_ID','LAST_12_MONTH_TOTAL_SALES']
        
        # Rounding-off to 0
        # df_sales['LAST_12_MONTH_TOTAL_SALES'] = np.round(df_sales['LAST_12_MONTH_TOTAL_SALES'],0)
        
        print('LAST 12 MONTH TOTAL SALES Column Created')
        print(f'Shape:{df_sales.shape}')
        
        
        # creating monthly sales from last one year data
        df_raw3 = df_raw2.groupby(['CUSTOMER_ID','MONTH','YEAR']).agg({'NET_SALES_AMOUNT':'sum'}).reset_index()
        
        # calculating last 12 months sale individually
        c = 12
        for i in range(0,12):
            
            # for labelling column name
            j = i + 1 
            
            # calculating last month sales from observation end month, 
            # Note: observation end month is last 1st month
            # example: feb is observation end month, then last_month1 = feb, last_month2 = jan, last_month3 = dec
            m = obs_end_date.month - i
            
            # to avoid 0 & Negative Numbers
            # example: for feb, 2-2=0, which means 12months, so taking 12th month and decreasing from 12th month
            if m <= 0:
                m = c
                c -= 1
            
            df_last_month_j_sale = df_raw3[df_raw3['MONTH']==m]
            df_last_month_j_sale.columns = ['CUSTOMER_ID','MONTH','YEAR',f'LAST_MONTH_{j}_SALES']
            df_last_month_j_sale = df_last_month_j_sale.fillna(0)
            
            # merging 
            df_sales = df_sales.merge(df_last_month_j_sale[['CUSTOMER_ID',f'LAST_MONTH_{j}_SALES']], on=['CUSTOMER_ID'], how='left')
            df_sales = df_sales.fillna(0)
            
            print(f'merged LAST 12MONTH TOTAL SALES WITH LAST_MONTH_{j}_SALES')
        
        print(df_sales.head())
        
        df_sales['LAST_3_MONTHS_SALES'] = df_sales['LAST_MONTH_1_SALES'] + df_sales['LAST_MONTH_2_SALES'] + df_sales['LAST_MONTH_3_SALES']
        df_sales['2ND_LAST_3_MONTHS_SALES'] = df_sales['LAST_MONTH_4_SALES'] + df_sales['LAST_MONTH_5_SALES'] + df_sales['LAST_MONTH_6_SALES']
        df_sales['3RD_LAST_3_MONTHS_SALES'] = df_sales['LAST_MONTH_7_SALES'] + df_sales['LAST_MONTH_8_SALES'] + df_sales['LAST_MONTH_9_SALES']
        df_sales['4TH_LAST_3_MONTHS_SALES'] = df_sales['LAST_MONTH_10_SALES'] + df_sales['LAST_MONTH_11_SALES'] + df_sales['LAST_MONTH_12_SALES']
        
        df_sales2 = df_sales[['CUSTOMER_ID','LAST_3_MONTHS_SALES','2ND_LAST_3_MONTHS_SALES',
                              '3RD_LAST_3_MONTHS_SALES','4TH_LAST_3_MONTHS_SALES']]
        
        return df_sales2

# Labeling Target
def label_class(df,p):
       df['STATUS'] = np.nan
       for idx, cid in enumerate(df['CUSTOMER_ID']):
           print(idx, cid)
           lss  = df['LAST_SEEN_SINCE_IN_DAYS'].loc[idx]
           p90  = df[f'PERCENTILE_{p}'].loc[idx]
           p100 = df['P100_PLUS_1CYCLE_BUFFER'].loc[idx]
           sale_trend = df['SALES_TREND'].loc[idx]
            
           # RISK OF CHURN
           # 1. LSS < P90 & SALES TREND = DECREASING
           # 2. LSS IN BETWEEN P90 AND P100_PLUS_1CYCLE_BUFFER
           if (lss < p90) & (sale_trend == 'DECREASING'):
               df['STATUS'].loc[idx] = 'RISK OF CHURN' 
          
           elif p90 <= lss <= p100:
               df['STATUS'].loc[idx] = 'RISK OF CHURN'
           
           elif lss > p100:
               df['STATUS'].loc[idx] = 'CHURNED'   
        
           else:
               df['STATUS'].loc[idx] = 'NON CHURN'
           
       return df
   
                
    
if __name__ == '__main__':
    
    # EU
    # server   = 
    # database = 
    # username = 
    # password = 
    # env =
    
    
    # cnxn = establish_con(server, database, username, password, env) 
    # cur = cnxn.cursor()
    
    # # loading from sql query
    # sql_query_path   = 'C:/Users/SqlFiles/'
    # sql_file = 'view.sql' 
    
    # sql_data = open(sql_query_path + sql_file, 'r')
    # df_raw   = pd.read_sql_query(sql_data.read(), cnxn)
    
    # print('Data loaded from sql to dataframe')
    # sql_data.close()
    # close_connection(cur, cnxn)
   
    # saving as excel
    file_path = 'C:/Users/PyFiles/MULTI_CLASS_CHURNMODELING/'
    filename  = 'DATA.xlsx'
    # df_raw.to_excel(file_path +filename , header = True, index=None)
    df_raw = pd.read_excel(file_path+filename)
    
    
    #COLUMNS
    columns = df_raw.columns.tolist()
    
    # basic checks
    df_raw.shape
    df_raw.info()
    df_raw.columns
    df_raw.describe()
    df_raw.isnull().sum()
    df_raw['CUSTOMER_ID'].nunique() #6702 as of 2022-09-15
    
    # value_counts
    df_raw['SALES_CATEGORY'].value_counts()
    df_raw['BILL_TYPE_DESC'].value_counts()
    
    # changing Data type
    df_raw['BILL_DATE_YYYY_MM_DD'] = pd.to_datetime(df_raw['BILL_DATE_YYYY_MM_DD'])
    df_raw['BILL_DATE_YYYY_MM_DD'].min() # 2019-01-02
    df_raw['BILL_DATE_YYYY_MM_DD'].max() # 2022-09-15
    
    
    # Taking only Invoice
    # considering them as actual transactions
    df_inv = df_raw[df_raw['BILL_TYPE_DESC']=='Invoice']

    # Aggregating Sales
    obs_end_date = pd.to_datetime('2022-06-30')
    df_ES2 = prepare_data(df_inv, obs_end_date)
    df_ES2['BILL_DATE_YYYY_MM_DD'].min()
    df_ES2['BILL_DATE_YYYY_MM_DD'].max()
    df_ES2['CUSTOMER_ID'].nunique()  # 6369 as of 2022-06-30
    
    
    # DATA ENGINEERING: LAST SEEN SINCE, FREQUENCY
    df_RF  = df_ES2.groupby('CUSTOMER_ID').agg({'BILL_DATE_YYYY_MM_DD':[lambda x:obs_end_date - x.max(),
                                                             lambda x:x.count()]}
                                              ).reset_index()
    df_RF.columns = ['CUSTOMER_ID','LAST_SEEN_SINCE_IN_DAYS','FREQUENCY']
    df_RF['LAST_SEEN_SINCE_IN_DAYS'] =  df_RF['LAST_SEEN_SINCE_IN_DAYS'].dt.days
    
    # customer should have atleast 
    # 3 Bill date to identify purchase behaviour
    df_RF2 = df_RF[df_RF['FREQUENCY']>2]
    df_RF2['FREQUENCY'].value_counts()
    
    # INNER JOIN LAST_SEEN_SINCE_IN_DAYS & FREQUENCY w
    df_ES3 = df_ES2.merge(df_RF2, on='CUSTOMER_ID', how='inner')
    df_ES3['CUSTOMER_ID'].nunique()
    # 4083 CUSTOMERS got purchase pattern
    
    # LAG BILL DATE COLUMN
    df_ES3['LAG_BILL_DATE'] = df_ES3.sort_values('BILL_DATE_YYYY_MM_DD',ascending=True) \
                                    .groupby(['CUSTOMER_ID'])['BILL_DATE_YYYY_MM_DD'].shift(1)
    # INTER PURCHASE DAYS COLUMN
    df_ES3['INTER_PURCHASE_DAYS'] = df_ES3['BILL_DATE_YYYY_MM_DD'] - df_ES3['LAG_BILL_DATE']
    df_ES3['INTER_PURCHASE_DAYS'] = df_ES3['INTER_PURCHASE_DAYS'].dt.days
    
    # REMOVING OUTLIERS
    # [001f02994f2fe38df50137a02de75ef4, 001f02994f2fe38df50137a02de75ef4]
    counter = 0
    start_time = datetime.now()
    df_noOutlier = df_ES3.groupby('CUSTOMER_ID').apply(remove_outlier)
    end_time = datetime.now()
    print('time taken to remove_outlier:', end_time-start_time)
 
    # dropping CUSTOMER_ID as it is there in index level
    df_noOutlier = df_noOutlier.drop('CUSTOMER_ID', axis=1)
    df_noOutlier = df_noOutlier.reset_index()
    df_noOutlier.drop('level_1', axis=1, inplace=True)
    print(f'{df_ES3.shape[0]}-{df_noOutlier.shape[0]}')
    # observation
    # before dropping outliers, total records = 95341
    # after  dropping outliers, total records = 91582

    
    # Calculating Percentile
    df_noOutlier = calculate_percentile(df_noOutlier,90)
    df_noOutlier = calculate_percentile(df_noOutlier,100)
    
    # Getting 90th & 100th percentile of each CUSTOMER
    df_noOutlier2 = df_noOutlier.groupby('CUSTOMER_ID') \
                                .agg({'PERCENTILE_90':lambda x :x.unique(),
                                      'PERCENTILE_100':lambda x :x.unique()}) \
                                .reset_index()
    
    # TAKING LAST TRANSACTION DETAILS
    # unique()[-1] picks up the last transaction
    df_ES3 = df_ES3.sort_values(by=['CUSTOMER_ID','BILL_DATE_YYYY_MM_DD'])
    df_ES4 = df_ES3.groupby('CUSTOMER_ID') \
                    .agg({'BILL_DATE_YYYY_MM_DD': lambda x: x.unique()[-1],
                          'LAST_SEEN_SINCE_IN_DAYS': lambda x: x.unique(),
                         })\
                    .reset_index()
    df_ES4['OBS_START_DATE'] = pd.to_datetime('2019-01-01')
    df_ES4['OBS_END_DATE'] = pd.to_datetime('2021-06-30')   
                 
    df_ES4 = df_ES4.merge(df_noOutlier2, on='CUSTOMER_ID', how='left')
    df_ES4.isnull().sum()
    
    # TARGET LABELLING
    df_ES4 = label_class(df_ES4, '90')
    df_ES4['STATUS'].value_counts(normalize=True)
    # Observation
    # NON CHURN        0.619515
    # CHURNED          0.349105
    # RISK OF CHURN    0.031380
    
    # BUCKETING PURCHASE PATTERN 
    df_ES4=bucket(df_ES4,'PERCENTILE_90','BUCKET')
    df_ES4['BUCKET'].value_counts(normalize=True)[['30Days','60Days',
                                                   '90Days','120Days',
                                                   '150Days','180Days',
                                                   '6+Months']]
    # Observation
    # 30Days      0.200098
    # 60Days      0.262552
    # 90Days      0.173647
    # 120Days     0.104825
    # 150Days     0.070536
    # 180Days     0.046289
    # 6+Months    0.142052
    
    df_ES5 = df_ES4.copy()
    df_ES5.rename(columns={'BILL_DATE_YYYY_MM_DD':'LAST_BILLDATE_AVAILABLE'}, inplace=True)
    df_ES5.to_excel(file_path + 'SPAIN_CID_PURCHASE_PATTERN.xlsx', header=True, index=None)

    
    # pp filtered
    df_ES4.columns
    df_pp_filtered = df_ES4[['CUSTOMER_ID','BILL_DATE_YYYY_MM_DD','OBS_END_DATE','LAST_SEEN_SINCE_IN_DAYS','PERCENTILE_90','PERCENTILE_100']]
    df_pp_filtered['P100_PLUS_1CYCLE_BUFFER'] = df_pp_filtered['PERCENTILE_90']+df_pp_filtered['PERCENTILE_100']
    
    df_qtr_breakup = monthly_sale_breakup(df_ES2)
    
    # adding Sales Trend column
    df_sale_trend = df_qtr_breakup.copy()
    df_sale_trend.columns
    df_sale_trend['SALES_TREND'] = np.nan
    
    for idx, v in enumerate(df_sale_trend['CUSTOMER_ID']):
        
        print(f'{idx}-{v}')
        
        value_3M  = df_sale_trend['LAST_3_MONTHS_SALES'].loc[idx]
        value_6M  = df_sale_trend['2ND_LAST_3_MONTHS_SALES'].loc[idx]
        value_9M  = df_sale_trend['3RD_LAST_3_MONTHS_SALES'].loc[idx]
        value_12M = df_sale_trend['4TH_LAST_3_MONTHS_SALES'].loc[idx]
        
        if (value_3M < value_6M) :  # < value_9M < value_12M 
        
            df_sale_trend['SALES_TREND'].loc[idx] = 'DECREASING'
        
        elif (value_3M > value_6M) :#> value_9M > value_12M 
            df_sale_trend['SALES_TREND'].loc[idx] = 'INCREASING'
            
        elif (value_3M ==0) & (value_6M==0) :#> value_9M > value_12M 
            df_sale_trend['SALES_TREND'].loc[idx] = 'NO SALE'
        
        elif (value_3M!=0) & (value_3M ==value_6M) :#> value_9M > value_12M 
            df_sale_trend['SALES_TREND'].loc[idx] = 'STABLE'
            
            
    
    df_sale_trend.isnull().sum()
    df_sale_trend['SALES_TREND'].value_counts(normalize=True)
    
    # merging with pp
    df_pp_sale = df_pp_filtered.merge(df_sale_trend, on='CUSTOMER_ID', how='left')
    
    # filling missing values
    df_pp_sale['SALES_TREND'] = df_pp_sale['SALES_TREND'].fillna('NO SALE')
    df_pp_sale.fillna(0.0, inplace=True)
    
    # saving file
    df_pp_sale.to_excel(file_path + 'SALE_TREND_PP.xlsx', header = True, index=None)
    
    # creating CHURN, NON-CHURN, RISK-OF-CHURN FLAGS
    df_flagged = label_class(df_pp_sale, '90')
    df_flagged['STATUS'].value_counts(normalize=True)
    
