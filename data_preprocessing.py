import pandas as pd
import numpy as np
import os
import os.path as op
import yfinance as yf

### Directories
root_dir = os.path.dirname(os.getcwd())
dest_dir = root_dir+'\\data'
create_directory(dest_dir) 
print("Created data directory")  

def create_directory(logdir):
    try:
        os.makedirs(logdir)
    except FileExistsError:
        pass

def create_dataset(df_, sp, dest_dir): 
    '''
        Select only the companies that existed at the beginning of testing period
        In a split of 1000 days, we are checking the companies that were present on the 750th day
        Then we are only using those companies
    '''
    
    cols = df_.iloc[750].dropna().index.values # Columns on 750th date of the split
    df_X = df_[cols] #Selecting only those columns
    target_df = calculate_target_df(df_X)
    normalized_df = normalize_df(df_X)
    slice_train_dataset(normalized_df[:750], target_df[:750], sp=sp)
    slice_test_dataset(normalized_df[750-240:],target_df[750-240:], dest_dir, sp)

def calculate_target_df(df):
    ''' 
        Stock returns that are above the daily median are labeled as one, and zero otherwise.
        Returns a dataframe with the classification labels.
    '''
    df = prepare_target_df(df)
    
    # Calculate the median for each row
    median = df.loc[:, :].median(axis=1)
    
    # Create the target dataframe
    target_df = pd.DataFrame(0, index=df.index, columns=df.columns)
    target_df[df.sub(median, axis=0).ge(0)] = 1
    
    return target_df

def prepare_target_df(df):
    ''' 
        Clean dataframe to create targets. 
        Remove any returns that don't have enough history so they don't count towards the labeling.
    '''
    copy_of_df = df.copy()
    for cols in df.columns:
        for i in range(240, len(df)):
            if df[cols].iloc[i-240:i].isnull().values.any():
                copy_of_df.iloc[i][cols] = np.nan
    return copy_of_df

def normalize_df(df):
    mean_ = df.iloc[:750].mean()
    std_ = df.iloc[:750].std()
    return (df - mean_) / std_


def slice_train_dataset(df_X, df_target, sp=None):
    cols = df_X.columns
    X_list = []
    Y_list = []
    for i in range(len(df_X)-240):
        for col in cols:
            X = df_X[col][i:i+240].values
            Y = df_target[col][i+240]
            if np.isnan(X).any() or np.isnan(Y):
                continue
            else:
                X_list.append(X)
                Y_list.append(Y)
    X_train = np.array(X_list).reshape(-1,240,1)
    Y_train = np.array(Y_list).reshape(-1,1) 
    np.save(op.join(dest_dir, 'study_period_X_'+str(sp)+'_train.npy'), X_train)
    np.save(op.join(dest_dir, 'study_period_Y_'+str(sp)+'_train.npy'), Y_train)

def slice_test_dataset(df_X, df_target, dest_dir, sp):
    cols = df_X.columns
    index_list, dates = [], []
    X_list = []
    Y_list = []
    lookback = 240
    for i in range(lookback, len(df_X)):
        dates.append(df_X.index[i])
        for j,col in enumerate(cols):
            X = df_X[col][i-lookback:i].values
            Y = df_target[col][i]
            if np.isnan(X).any() or np.isnan(df_X[col].iloc[i]):
                continue
            else: 
                index_list.append([i-240, j])
                X_list.append(X)
                Y_list.append(Y)
    columns = np.array(df_X.columns)
    dates_array = np.array(dates)
    index_array = np.array(index_list)
    inference_dir = op.join(dest_dir, 'sp'+str(sp))
    X_test = np.array(X_list).reshape(-1,240,1)
    Y_test = np.array(Y_list).reshape(-1,1)
    create_directory(inference_dir)
    np.save(op.join(inference_dir, 'columns.npy'), columns)
    np.save(op.join(inference_dir, 'dates.npy'), dates_array)
    np.save(op.join(inference_dir, 'index_array.npy'), index_array)
    np.save(op.join(dest_dir, 'study_period_X_'+str(sp)+'_test.npy'), X_test)
    np.save(op.join(dest_dir, 'study_period_Y_'+str(sp)+'_test.npy'), Y_test)


def process_dataset(dest_dir, df):
    j = 0
    count = 0
    while count+1000 < len(df):
        print("Split :"+str(j+1))
        df_ = df.iloc[count:count+1000]
        create_dataset(df_, j, dest_dir)
        count += 250
        j += 1


# Read the CSV file containing the constituents information
constituents = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")

# Get the ticker symbol for each constituent
tickers = constituents['Symbol'].tolist()

print("Downloaded S&P 500 tickers")

returns=[]   
for firm in tickers:
    df_firm = yf.download(tickers=[firm], start="1990-01-01")
    df_firm["Return"] = df_firm["Close"].pct_change()
    df_firm = df_firm[["Return"]]
    df_firm = df_firm.rename(columns={"Return": f"{firm}"})
    returns.append(df_firm)
df = pd.concat(returns, axis=1, sort=False)
df = df.dropna(how='all',axis=1)

process_dataset(dest_dir, df)

