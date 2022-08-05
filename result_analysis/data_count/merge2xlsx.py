import os
import pandas as pd

csv_list =  [ i for i in os.listdir('./') if '.csv' in i ]

data_list = []
for csv_name in csv_list:
    data_list.append(pd.read_csv(csv_name))

csv_name_list = ['Dataset number: ' + i[37:-4] for i in csv_list]
result_merge = pd.concat(data_list, axis=1, keys=csv_name_list)

result_merge.to_excel('result_merge.xlsx')