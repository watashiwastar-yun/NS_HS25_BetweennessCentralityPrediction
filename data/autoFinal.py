import networkx as nx
import pandas as pd
import os
import sys
import subprocess


cityRoot= r'C:/Users/16409/Downloads/city'
cities=['adelaide', 'belfast', 'berlin', 'bordeaux', 'brisbane',
    'canberra', 'detroit', 'dublin', 'grenoble', 'helsinki',
    'kuopio', 'lisbon', 'luxembourg', 'melbourne', 'nantes',
    'palermo', 'paris', 'prague', 'rennes', 'rome',
    'sydney', 'toulouse', 'turku', 'venice', 'winnipeg']

Script_DIR=os.path.dirname(os.path.abspath(__file__))


def automaton():
    summary=[]
    graph=os.path.join(Script_DIR,'makeGraph.py')
    centrality=os.path.join(Script_DIR,'centrality.py')
    visualize=os.path.join(Script_DIR,'visualize.py')

    for city in cities:
        print(f"\n Processing {city}")
        result_folder=os.path.join(Script_DIR,f"{city} rsults")
        os.makedirs(result_folder,exist_ok=True)
        cityDataPath=os.path.join(cityRoot,city)

        print(f"[{city}]:running make graph!")
        subprocess.run([sys.executable, graph,cityDataPath],cwd=result_folder,check=True)
        print(f"[{city}]:calculating centrality!")
        subprocess.run([sys.executable,centrality, cityDataPath], cwd=result_folder, check=True)
        print(f"[{city}]:visualizing centrality!")
        subprocess.run([sys.executable, visualize, cityDataPath], cwd=result_folder, check=True)
        #print("********************")

        final= os.path.join(result_folder,'graph_with_centrality.gml')
        if os.path.exists(final):
            G=nx.read_gml(final)
            line = f"city:{city.upper()}|nodes:{G.number_of_nodes()}|edges:{G.number_of_edges()}"
            summary.append(line)
            print(f"{city} recorded")





    summarytxt=os.path.join(Script_DIR,'summary.txt')
    with open(summarytxt,'w') as f:
        for line in summary:
            f.write(line + "\n")
            print(1)


    print("done")




automaton()
