import pandas as pd
import numpy as np
import os

directory = os.path.dirname(__file__)
print(directory)    
filepath = os.path.join(directory, 'data/data1.xlsx')

dict = pd.read_excel(filepath, sheet_name=None)
for df in dict.values():
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

df_eur = dict["Eurotunnel"]
df_lor = dict["L'Oréal"]

# data = pd.DataFrame(np.nan, index=df_eurotunnel.index, columns=['eur', 'lor'])
# print(data)

# 1
print('\nPrices: Eurotunnel and L\'Oréal')
print(df_eur.loc[:, df_eur.columns != 'nb_trades'].describe().loc[['mean', 'min', 'max']])
print(df_lor.loc[:, df_lor.columns != 'nb_trades'].describe().loc[['mean', 'min', 'max']])

# print('\nReturns: Eurotunnel and L\'Oréal')
# print(df_eur.pct_change().loc[:, df_eur.columns != 'nb_trades'].describe().loc[['mean', 'min', 'max']])
# print(df_lor.pct_change().loc[:, df_eur.columns != 'nb_trades'].describe().loc[['mean', 'min', 'max']])

# 2
df_eur['return(t)'] = df_eur['close'].pct_change()
df_lor['return(t)'] = df_lor['close'].pct_change()
df_eur['return(t-1)'] = df_eur['return(t)'].shift(1)
df_lor['return(t-1)'] = df_lor['return(t)'].shift(1)

# print(df_eur)

# 4

df_eur['p(t)-p(t-1)'] = df_eur['close'] - df_eur['close'].shift(1)
df_eur['p(t-1)-p(t-2)'] = df_eur['close'].shift(1) - df_eur['close'].shift(2)
df_eur['n(t)/n(t-1)'] = df_eur['p(t)-p(t-1)'] / df_eur['p(t-1)-p(t-2)']

df_lor['p(t)-p(t-1)'] = df_lor['close'] - df_lor['close'].shift(1)
df_lor['p(t-1)-p(t-2)'] = df_lor['close'].shift(1) - df_lor['close'].shift(2)
df_lor['n(t)/n(t-1)'] = df_lor['p(t)-p(t-1)'] / df_lor['p(t-1)-p(t-2)']
# print(df_eur)
# print(df_eur)



# 5 - Explanation
print('Price changes: Eurotunnel and L\'Oréal')
print(df_eur.loc[:, df_eur.columns == 'p(t)-p(t-1)'].describe().loc[['mean', 'min', 'max']])
print(df_lor.loc[:, df_lor.columns == 'p(t)-p(t-1)'].describe().loc[['mean', 'min', 'max']])

eur_mean_change = df_eur['p(t)-p(t-1)'].mean()
lor_mean_change = df_lor['p(t)-p(t-1)'].mean()

eur_mean_price = df_eur['close'].mean()
lor_mean_price = df_lor['close'].mean()

print(f'Mean price change for Eurotunnel: {eur_mean_change}')
print(f'Mean price change for L\'Oréal: {lor_mean_change}')

print(f'Ratio of mean price change to mean price for Eurotunnel: {eur_mean_change / eur_mean_price:.7f}')
print(f'Ratio of mean price change to mean price for L\'Oréal: {lor_mean_change / lor_mean_price:.7f}')

print('Length of Eurotunnel data: ', len(df_eur))
print('Length of L\'Oréal data: ', len(df_lor))