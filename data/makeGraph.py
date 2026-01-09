import pandas as pd
import networkx as nx
import os
import sys

if len(sys.argv)>1:
    citybase=sys.argv[1]
else:
    citybase= 'C:/Users/16409/Downloads/city/detroit/'
path=os.path.join(citybase,'network_temporal.csv')




start='from_stop_I'
target='to_stop_I'
departureTimeC='dep_time_ut'
arrivalTimeC='arr_time_ut'

dataframe=pd.read_csv(path,sep=",", skipinitialspace=True, engine='python', encoding='latin1')

if departureTimeC not in dataframe.columns or arrivalTimeC not in dataframe.columns:#check validity before processing
    print("col error")
    exit()

#calculate average,combined multiple rides between two edges into 1 single ride(Eg: there are eight rides from Zurich HB to oerlikon per day, this steps reduce the number of ride to 1, calculating the average travel distance)
dataframe['duration']=dataframe[arrivalTimeC]-dataframe[departureTimeC]
routes=dataframe.groupby([start,target])
avg=routes['duration'].mean()
result=avg.reset_index()
dfCombined= result.rename(columns={'duration':'avg_travel_time'})



print("there are "+str(len(dataframe))+" trips， now aggregate to" +str(len(dfCombined))+" static routes")


G=nx.DiGraph()
#graphing
for row in dfCombined.itertuples():
    u=row.from_stop_I
    v=row.to_stop_I
    w=row.avg_travel_time
    G.add_edge(u, v, weight_AKA_average_time=w)


print(f" Constructed a static DiGraph with {G.number_of_nodes()} stops and {G.number_of_edges()} average routes.")

output= 'avg_travel_graph2.gml'

nx.write_gml(G, output)

print(f" saved to {output}")




