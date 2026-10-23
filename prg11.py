import pandas as pd
data={ 'name':['naji','mohammed','the'],
      'age':[20,10,30],
      'mark':[100,200,300]}
x=pd.DataFrame(data)
print(x.sort_values(['age','mark']))