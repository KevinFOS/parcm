import numpy as np
import glob
import csv
from sklearn.decomposition import PCA
names_raw=csv.reader(open("./std/list.csv",encoding="ISO-8859-1"))
names=[]
for row in names_raw:
    names.append(row)
cur=[]
basic_features=[]
full_features=np.empty([64,0],dtype=float)
for num in range(0,245):
    cur.append(np.loadtxt('./std/vggish_csv/'+str(num+1)+'.csv',delimiter=','))
    cur[num]=cur[num][1:][0:64]
    mean_features=np.mean(cur[num],axis=0)
    std_features=np.std(cur[num],axis=0)
    max_features=np.max(cur[num],axis=0)
    min_features=np.min(cur[num],axis=0)
    aggregated_features=np.hstack([mean_features,std_features,max_features,min_features])
    basic_features.append(aggregated_features)
basic_dist=[]
dt=np.dtype([('no',int),('basic',float)])
files=glob.glob("./data/csvs/*.csv")
for file in files:
    print(file)
    target=np.loadtxt(file,delimiter=',')
    mindis=1000000000
    minsong=0
    minpos=0
    for row in range(len(target)):
        if row==0:
            continue
        split=target[row:row+64]
        if(len(split)<64):
            break
        aggregated_split=np.hstack([np.mean(split,axis=0),np.std(split,axis=0),np.max(split,axis=0),np.min(split,axis=0)])
        basic_dist=[]
        music=[]
        for num in range(0,245):
            basic_dist.append(np.sqrt(np.sum(np.square(aggregated_split - basic_features[num]))))
            cur_obj=np.array((num,basic_dist[num]),dtype=dt)
            music.append(cur_obj)
        music=np.sort(music,order='basic')
        if mindis>music[0]['basic']:
            mindis=music[0]['basic']
            minsong=music[0]['no']
            minpos=row
    print(names[minsong])
    print("Starting At (vggish frame):",minpos)
    print("Dist:",mindis,"Score:",int(100/(mindis/256)))
