'LIBRARY'
import pandas as pd
import numpy as np

'DATA'
admin = pd.read_csv('D:/Rollingpinn/Customers/ADMIN.csv')
web = pd.read_csv('D:/Rollingpinn/Customers/WEB.csv')
grab = pd.read_csv('D:/Rollingpinn/Customers/grab.csv')

'CONCAT'
concat = pd.concat([admin,web,grab],axis=0)
concat = concat.groupby(['billing_phone'])['PRICE'].sum().reset_index()
concat = concat.sort_values(by='PRICE',ascending = False)

                          'CUSTOMERS'
l = [i for i in concat["billing_phone"]]
#Aligning with country code
l2 = list()
for ph in l:
  ph = str(ph)
  if ph[0] == "0":
    ph = ph[1:]
    ph = "+66" + ph
    l2.append(ph)
  else:
    ph = "+66" + ph
    l2.append(ph)

l2 = set(l2)
data = pd.DataFrame(l2)
