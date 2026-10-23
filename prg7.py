import pandas as pd

date=pd.date_range('2021-10-1','2021-10-10')
x=pd.Series(date)
print(x)