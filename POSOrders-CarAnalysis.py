'LIBRARIES'
import json
import pandas as pd
from datetime import datetime
import pytz

'DATA IMPORT'
pos = pd.read_csv('D:/Rollingpinn/Cartanalysis/POS/POS-OCT.csv')
keys = pd.read_excel('D:/Rollingpinn/Cartanalysis/POS/Product Keys/Keys.xlsx')

'SORTING DATE'
pos['Orderdate']= pos['Orderdate'].str.strip()
keys['Keys']=keys['Keys'].str.strip()

'SEPARATING LOCATIONS'
loc1 = pos[pos['address_id']==1]
loc2 = pos[pos['address_id']==2]
loc11 = pos[pos['address_id']==11]
loc16 = pos[pos['address_id']==16]
loc23 = pos[pos['address_id']==23]

'SELECTING COLUMNS'
All = pos[['Orderdate','cart']]
loc1 = loc1[['Orderdate','cart']]
loc2 = loc2[['Orderdate','cart']]
loc11 = loc11[['Orderdate','cart']]
loc16 = loc16[['Orderdate','cart']]
loc23 = loc23[['Orderdate','cart']]

'EMPTY DATAFRAMES FOR ITERATING'
df = pd.DataFrame(columns = ['Orderdate', 'Keys', 'Quantity'])
df1 = pd.DataFrame(columns = ['Orderdate', 'Keys', 'Quantity'])
df2 = pd.DataFrame(columns = ['Orderdate', 'Keys', 'Quantity'])
df11 = pd.DataFrame(columns = ['Orderdate', 'Keys', 'Quantity'])
df16 = pd.DataFrame(columns = ['Orderdate', 'Keys', 'Quantity'])
df23 = pd.DataFrame(columns = ['Orderdate', 'Keys', 'Quantity'])

'ALL POS ORDERS'
for ind,row in All.iterrows():
    d = json.loads(row[1])
    for i in d.keys():
        r = {'Orderdate':row[0],'Keys':i,'Quantity':d[i]}
        df = df.append(r,ignore_index=True)
        df=pd.DataFrame(df)

df = pd.DataFrame(df.groupby(['Orderdate','Keys'])['Quantity'].sum())
df = df.reset_index()

All = df.merge(keys, how='inner', on='Keys')
All['TotalPrice'] = All['Unitprice']*All['Quantity']

'LOCATION1 - SAMYAN MITRTOWN'
for ind,row in loc1.iterrows():
    d = json.loads(row[1])
    for i in d.keys():
        r = {'Orderdate':row[0],'Keys':i,'Quantity':d[i]}
        df1 = df1.append(r,ignore_index=True)
        df1=pd.DataFrame(df1)

df1 = pd.DataFrame(df1.groupby(['Orderdate','Keys'])['Quantity'].sum())
df1 = df1.reset_index()
df1product = pd.DataFrame(df1.groupby(['Keys'])['Quantity'].sum())
df1product = df1product.reset_index()

Location1 = df1.merge(keys, how='inner', on='Keys')
Location1['TotalPrice'] = Location1['Unitprice']*Location1['Quantity']
Location1product = df1product.merge(keys, how='inner', on='Keys')
Location1product['TotalPrice'] = Location1product['Unitprice']*Location1product['Quantity']

## Datewise revenue
Location1['Orderdate'] = pd.to_datetime(Location1['Orderdate'])
Location1 = Location1.sort_values(by = 'Orderdate')
daywisesamyanmitrtown = pd.DataFrame(Location1.groupby(['Orderdate'])['TotalPrice'].sum())

'LOCATION2 - EMQUARTIER'
for ind,row in loc2.iterrows():
    d = json.loads(row[1])
    for i in d.keys():
        r = {'Orderdate':row[0],'Keys':i,'Quantity':d[i]}
        df2 = df2.append(r,ignore_index=True)
        df2=pd.DataFrame(df2)

df2 = pd.DataFrame(df2.groupby(['Orderdate','Keys'])['Quantity'].sum())
df2 = df2.reset_index()
df2product = pd.DataFrame(df2.groupby(['Keys'])['Quantity'].sum())
df2product = df2product.reset_index()

Location2product = df2product.merge(keys, how='inner', on='Keys')
Location2product['TotalPrice'] = Location2product['Unitprice']*Location2product['Quantity']
Location2 = df2.merge(keys, how='inner', on='Keys')
Location2['TotalPrice'] = Location2['Unitprice']*Location2['Quantity']

## Datewise revenue
Location2['Orderdate'] = pd.to_datetime(Location2['Orderdate'])
Location2 = Location2.sort_values(by = 'Orderdate')
daywisesemquartier = pd.DataFrame(Location2.groupby(['Orderdate'])['TotalPrice'].sum())

'LOCATION11 - SIAM PARAGON'
for ind, row in loc11.iterrows():
    d = json.loads(row[1])
    for i in d.keys():
        r = {'Orderdate': row[0], 'Keys': i, 'Quantity': d[i]}
        df11 = df11.append(r, ignore_index=True)
        df11 = pd.DataFrame(df11)

df11 = pd.DataFrame(df11.groupby(['Orderdate','Keys'])['Quantity'].sum())
df11 = df11.reset_index()
df11product = pd.DataFrame(df11.groupby(['Keys'])['Quantity'].sum())
df11product = df11product.reset_index()

Location11product = df11product.merge(keys, how='inner', on='Keys')
Location11product['TotalPrice'] = Location11product['Unitprice']*Location11product['Quantity']
Location11 = df11.merge(keys, how='inner', on='Keys')
Location11['TotalPrice'] = Location11['Unitprice']*Location11['Quantity']

## Datewise revenue
Location11['Orderdate'] = pd.to_datetime(Location11['Orderdate'])
Location11 = Location11.sort_values(by = 'Orderdate')
daywisesiamparagon = pd.DataFrame(Location11.groupby(['Orderdate'])['TotalPrice'].sum())


'LOCATION16-CENTRALWORLD'
for ind,row in loc16.iterrows():
    d = json.loads(row[1])
    for i in d.keys():
        r = {'Orderdate':row[0],'Keys':i,'Quantity':d[i]}
        df16 = df16.append(r,ignore_index=True)
        df16=pd.DataFrame(df16)

df16 = pd.DataFrame(df16.groupby(['Orderdate','Keys'])['Quantity'].sum())
df16 = df16.reset_index()
df16product = pd.DataFrame(df16.groupby(['Keys'])['Quantity'].sum())
df16product = df16product.reset_index()

Location16 = df16.merge(keys, how='inner', on='Keys')
Location16['TotalPrice'] = Location16['Unitprice']*Location16['Quantity']
Location16product = df16product.merge(keys, how='inner', on='Keys')
Location16product['TotalPrice'] = Location16product['Unitprice']*Location16product['Quantity']

## Datewise revenue
Location16['Orderdate'] = pd.to_datetime(Location16['Orderdate'])
Location16 = Location16.sort_values(by = 'Orderdate')
daywisecw = pd.DataFrame(Location16.groupby(['Orderdate'])['TotalPrice'].sum())

'LOCATION23-SILOM'
for ind,row in loc23.iterrows():
    d = json.loads(row[1])
    for i in d.keys():
        r = {'Orderdate':row[0],'Keys':i,'Quantity':d[i]}
        df23 = df23.append(r,ignore_index=True)
        df23=pd.DataFrame(df23)

df23 = pd.DataFrame(df23.groupby(['Orderdate','Keys'])['Quantity'].sum())
df23 = df23.reset_index()
df23product = pd.DataFrame(df23.groupby(['Keys'])['Quantity'].sum())
df23product = df23product.reset_index()

Location23 = df23.merge(keys, how='inner', on='Keys')
Location23['TotalPrice'] = Location23['Unitprice']*Location23['Quantity']
Location23product = df23product.merge(keys, how='inner', on='Keys')
Location23product['TotalPrice'] = Location23product['Unitprice']*Location23product['Quantity']

## Datewise revenue
Location23['Orderdate'] = pd.to_datetime(Location23['Orderdate'])
Location23 = Location23.sort_values(by = 'Orderdate')
daywisesilom = pd.DataFrame(Location23.groupby(['Orderdate'])['TotalPrice'].sum())


'EXPORT DATA'
All.to_excel('D:/Rollingpinn/Cartanalysis/POS/All/All-POSORDERS.xlsx')
Location1.to_excel('D:/Rollingpinn/Cartanalysis/POS/Locations/Location1-SamyanMitrtown.xlsx')
Location1product.to_excel('D:/Rollingpinn/Cartanalysis/POS/Locations/Location1product-SamyanMitrtown.xlsx')
daywisesamyanmitrtown.to_excel('D:/Rollingpinn/Cartanalysis/POS/Daywise Revenue/SamyanMitrtown.xlsx')
Location23.to_excel('D:/Rollingpinn/Cartanalysis/POS/Locations/Location23-Silom.xlsx')
Location23product.to_excel('D:/Rollingpinn/Cartanalysis/POS/Locations/Location23product-Silom.xlsx')
daywisesilom.to_excel('D:/Rollingpinn/Cartanalysis/POS/Daywise Revenue/Silom.xlsx')
Location16.to_excel('D:/Rollingpinn/Cartanalysis/POS/Locations/Location16-CentralWorld.xlsx')
Location16product.to_excel('D:/Rollingpinn/Cartanalysis/POS/Locations/Location16product-Centralworld.xlsx')
daywisecw.to_excel('D:/Rollingpinn/Cartanalysis/POS/Daywise Revenue/CentralWorld.xlsx')
Location11.to_excel('D:/Rollingpinn/Cartanalysis/POS/Locations/Location11.xlsx')
Location11product.to_excel('D:/Rollingpinn/Cartanalysis/POS/Locations/Location11product.xlsx')
daywisesiamparagon.to_excel('D:/Rollingpinn/Cartanalysis/POS/Daywise Revenue/SiamParagon.xlsx')
Location2.to_excel('D:/Rollingpinn/Cartanalysis/POS/Locations/Location2-Emquartier.xlsx')
Location2product.to_excel('D:/Rollingpinn/Cartanalysis/POS/Locations/Location2product-Emquartier.xlsx')
daywiseemquartier.to_excel('D:/Rollingpinn/Cartanalysis/POS/Daywise Revenue/Emquartier.xlsx')