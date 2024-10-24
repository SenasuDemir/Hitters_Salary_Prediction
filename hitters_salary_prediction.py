################################################################
# PROJECT: SALARY PREDICTION WITH MACHINE LEARNING
################################################################

# IS PROBLEMI
# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol oyuncularının
# maaş tahminleri için bir makine öğrenmesi modeli geliştiriniz.

# Veri seti hikayesi

# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
# Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır.
# Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan
# 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.

# Batters - Hitters (Vurucularin)

# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş (bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör


#############################################################
# Gerekli Kutuphane ve Fonksiyonlar
#############################################################

import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate,cross_val_score,validation_curve

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.impute import  KNNImputer

pd.set_option('display.max_columns',None)
pd.set_option('display.width',170)
pd.set_option('display.max_rows',20)
pd.set_option('display.float_format',lambda x:'%.3f' % x)

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore',category=ConvergenceWarning)
warnings.filterwarnings('ignore',category=pd.errors.SettingWithCopyWarning)

#############################################################
# GELISMIS FONKSIYONEL KESIFCI VERI ANALIZI (ADVANCED FUNCTIONAL EDA)
#############################################################
# 1. Genel Resim
# 2. Kategorik Degisken Analizi (Analysis of Categorical Variables)
# 3. Sayisal Degisken Analizi (Analysis of Numerical Variables)
# 4. Hedef Degisken Analizi (Analysis of Target Variables)
# 5. Korelasyon Analizi (Analysis of Correlation)

###########################
# 1. Genel Resim
###########################


df=pd.read_csv('datasets/hitters.csv')

def check_df(dataframe,head=5):
    print('########################### Shape ###########################')
    print(dataframe.shape)
    print('########################### Types ###########################')
    print(dataframe.dtypes)
    print('########################### Head ###########################')
    print(dataframe.head(head))
    print('########################### Tail ###########################')
    print(dataframe.tail(head))
    print('########################### NA ###########################')
    print(dataframe.isnull().sum())
    print('########################### Quantiles ###########################')
    print(dataframe.describe([0,0.05,0.50,0.95,0.99,1]).T)


check_df(df)

def show_salary_skewness(df):
    # Salary sutunudkai eksik degerleri kaldirma
    salary=df['Salary'].dropna()

    # Carpiklik degerini hesaplama
    skewness=salary.skew()

    # Histogram ve yogunluk grafigi cizme
    plt.figure(figsize=(10,6))
    sns.histplot(salary,kde=True)
    plt.title(f'Salary Distribution Skewness: {skewness:.2f}')
    plt.xlabel('Salary (thousands)')
    plt.ylabel('Frequency')

    # Ortalama ve medyani gosterme
    plt.axvline(salary.mean(),color='r',linestyle='dashed',linewidth=2,label=f'Mean: {salary.mean():.2f}')
    plt.axvline(salary.median(), color='g', linestyle='dashed', linewidth=2, label=f'Median: {salary.median():.2f}')
    plt.legend()

    # Istatistiksek bilgileri yazdirma
    print(f'Skewness: {skewness:.2f}')
    print(f'Mean: {salary.mean():.2f}')
    print(f'Median: {salary.median():.2f}')
    print(f'Standard Deviation: {salary.std():.2f}')
    print(f'Minimum:{salary.min():.2f}')
    print(f'Maximum: {salary.max():.2f}')

    # Shapiro-wilk normallik testi
    _,p_value =stats.shapiro(salary)
    plt.show(block=True)

show_salary_skewness(df)

def grab_col_names(dataframe, cat_th=10,car_th=20):
    """
    Define the names of categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Categorical variables also include those that appear numeric but are actually categorical

    Parameters
    ----------
    dataframe: dataframe
                dataframe from which variable names are to be extracted
    cat_th: int, optional
                Threshold value for the number of classes for variables that are numerical but actually categorical
    car_th: int,optional
                Threshold value for the number of classes for variables that are categorical but cardinal

    Returns
    -------
        cat_cols: list
                List of categorical variables
        num_cols: list
                List of numerical variables
        cat_but_car: list
                List of cardinal variables that appear categorical


    Examples
    --------
        import seaborn as sns
        df=sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = Total number of variables
        num_but_cat is within cat_cols'
    """

    #cat_cols, cat_but_car
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique()< cat_th and dataframe[col].dtypes != 'O']
    cat_but_car=[col for col in dataframe.columns if dataframe[col].nunique()>car_th and dataframe[col].dtypes =='O']
    cat_cols=cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols=[col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols=[col for col in num_cols if col not in num_but_cat]

    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols,num_cols,cat_but_car

cat_cols,num_cols,cat_but_car=grab_col_names(df)

###########################
# 2. Kategorik Degisken Analizi (Analysis of Categorical Variables)
###########################

def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        'Ratio':100*dataframe[col_name].value_counts()/len(dataframe)}))
    print('################################################################')
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df,col,plot=True)

###########################
# 3. Sayisal Degisken Analizi (Analysis of Numerical Variables)
###########################

def num_summary(dataframe,numerical_col,plot=False):
    quantiles =[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
       dataframe[numerical_col].hist(bins=20)
       plt.xlabel(numerical_col)
       plt.title(numerical_col)
       plt.show(block=True)

for col in num_cols:
    num_summary(df,col,plot=True)


###########################
# 4. Hedef Degisken Analizi (Analysis of Target Variables)
###########################

def target_summary_with_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({'Target_Mean':dataframe.groupby(categorical_col)[target].mean()}),end='\n\n\n')

for col in cat_cols:
    target_summary_with_cat(df,'Salary',col)


###########################
# 5. Korelasyon Analizi (Analysis of Correlation)
###########################

df[num_cols].corr(method='spearman')

fig,ax=plt.subplots(figsize=(25,10))
sns.heatmap(df[num_cols].corr(),annot=True,linewidths=5,ax=ax)
plt.show(block=True)

# correlation with the final state of the variables
plt.figure(figsize=(45,45))
corr=df[num_cols].corr()
mask=np.triu(np.ones_like(corr,dtype=bool))
sns.heatmap(df[num_cols].corr(),mask=mask,cmap='coolwarm',vmax=3,center=0,
            square=True,linewidths=5,annot=True)
plt.show(block=True)

def find_correlation(dataframe,numeric_cols,corr_limit=0.50):
    high_correlations=[]
    low_correlations=[]
    for col in numeric_cols:
        if col =='Salary':
            pass
        else:
            correlation=dataframe[[col,'Salary']].corr().loc[col,'Salary']
            print(col,correlation)
            if abs(correlation)>corr_limit:
                high_correlations.append(col+': '+str(correlation))
            else:
                low_correlations.append(col+': '+str(correlation))
    return low_correlations,high_correlations

low_cor,high_cor=find_correlation(df,num_cols)

#############################################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################################

# 1. Outliers (Aykırı Değerler)
# 2. Missing Values (Eksik Değerler)
# 3. Feature Extraction (Özellik Çıkarımı)
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling (Özellik Ölçeklendirme)


###########################
# 1. Outliers (Aykırı Değerler)
###########################

sns.boxplot(x=df['Salary'],data=df)
plt.show(block=True)

for col in num_cols:
    sns.boxplot(x=df[col], data=df)
    plt.show(block=True)

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col,check_outlier(df,col))

for col in num_cols:
    if check_outlier(df,col):
        replace_with_thresholds(df,col)

###########################
# 2. Missing Values (Eksik Değerler)
###########################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

# Eksik veri analizine uygun olarak 3 farkli yontem kullanabiliriz.
df1=df.copy()
df1.head()
cat_cols,num_cols,cat_but_car=grab_col_names(df1)

method=int(input('Eksik veri icin hangi yontemi uygulamak istersiniz (1/2/3): '))

from sklearn.impute import KNNImputer



def fill_missing_values(dataframe,method):
    df1=dataframe.copy()
    cat_cols, num_cols, cat_but_car = grab_col_names(df1)
    if method == 1:
        dff = pd.get_dummies(df1[cat_cols + num_cols], drop_first=True)
        scaler = RobustScaler()
        dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
        imputer = KNNImputer(n_neighbors=5)
        dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
        dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
        df1 = dff
    elif method == 2:
        df1.loc[(df1['Salary'].isnull()) & (df1['League'] == 'A') & (df1['Division'] == 'E'), 'Salary'] = \
            df1.groupby(['League', 'Division'])['Salary'].mean()['A', 'E']

        df1.loc[(df1['Salary'].isnull()) & (df1['League'] == 'A') & (df1['Division'] == 'W'), 'Salary'] = \
            df1.groupby(['League', 'Division'])['Salary'].mean()['A', 'W']

        df1.loc[(df1['Salary'].isnull()) & (df1['League'] == 'N') & (df1['Division'] == 'E'), 'Salary'] = \
            df1.groupby(['League', 'Division'])['Salary'].mean()['N', 'E']

        df1.loc[(df1['Salary'].isnull()) & (df1['League'] == 'N') & (df1['Division'] == 'W'), 'Salary'] = \
            df1.groupby(['League', 'Division'])['Salary'].mean()['N', 'W']

    elif method == 3:
        # Drop NA
        # Delete all rows containing missing values
        df1.dropna(inplace=True)
    return df1

df1=fill_missing_values(df,method=1)

df1.head()
df1.isnull().sum()

###########################
# 3. Feature Extraction
###########################

new_num_cols=[col for col in num_cols if col!='Salary']
df1[new_num_cols]=df1[new_num_cols]+0.0000000001

df1["Hits_Success"] = (df1["Hits"] / df1["AtBat"]) * 100
df1['NEW_RBI'] = df1['RBI'] / df1['CRBI']
df1['NEW_Walks'] = df1['Walks'] / df1['CWalks']
df1['NEW_PutOuts'] = df1['PutOuts'] * df1['Years']
df1['NEW_Hits'] = df1['Hits'] / df1['CHits'] + df1['Hits']
df1["NEW_CRBI*CATBAT"] = df1['CRBI'] * df1['CAtBat']
df1["NEW_Chits"] = df1["CHits"] / df1["Years"]
df1["NEW_CHmRun"] = df1["CHmRun"] * df1["Years"]
df1["NEW_CRuns"] = df1["CRuns"] / df1["Years"]
df1["NEW_Chits"] = df1["CHits"] * df1["Years"]
df1["NEW_RW"] = df1["RBI"] * df1["Walks"]
df1["NEW_CH_CB"] = df1["CHits"] / df1["CAtBat"]
df1["NEW_CHm_CAT"] = df1["CHmRun"] / df1["CAtBat"]
df1['NEW_Diff_Atbat'] = df1['AtBat'] - (df1['CAtBat'] / df1['Years'])
df1['NEW_Diff_Hits'] = df1['Hits'] - (df1['CHits'] / df1['Years'])
df1['NEW_Diff_HmRun'] = df1['HmRun'] - (df1['CHmRun'] / df1['Years'])
df1['NEW_Diff_Runs'] = df1['Runs'] - (df1['CRuns'] / df1['Years'])
df1['NEW_Diff_RBI'] = df1['RBI'] - (df1['CRBI'] / df1['Years'])
df1['NEW_Diff_Walks'] = df1['Walks'] - (df1['CWalks'] / df1['Years'])


df1.columns
df1['Salary'].isnull().sum()

cat_cols, num_cols, cat_but_car = grab_col_names(df1)
df.shape
df1.shape

