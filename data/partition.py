import networkx as nx
import numpy as np
import os
import sys



Script_DIR=os.path.dirname(os.path.abspath(__file__))
Processed= os.path.join(Script_DIR,"cityProcessed")


"""divide G into k parts, where k== |n|//2000 using 
Kernighan-Lin algorithm
"""
def partition(G,k):
    communities =[set(G.nodes())]

    while len(communities)<k:
        #split larges community first
        communities.sort(key=len,reverse=True)
        currentNodes=communities.pop(0)

        if len(currentNodes)<2:
            communities.append(currentNodes)
            continue

        subG=G.subgraph(currentNodes).to_undirected()
        #KL works on undirected, need to change them to undirected first. the final subgraphs will still be directed though0
        remainingParts=k-len(communities)
        ratio =1.0/remainingParts
        targetSize=int(len(currentNodes)*ratio)

        try:
            nodesList=list(currentNodes)
            part1=set(nodesList[:targetSize])
            part2=set(nodesList[targetSize:])#initial 2 parts, we cut recursively
            p1,p2=nx.community.kernighan_lin_bisection(subG,partition=(part1,part2),weight="weight_AKA_average_time")
            communities.append(p1)
            communities.append(p2)



        except:
            #fallback for disconnectted componnents, if theres any
            nodelist=list(currentNodes)
            communities.append(set(nodelist[:targetSize]))
            communities.append(set(nodelist[targetSize:]))

    return communities

def run():

    with open("summary.txt","r") as f:
        lines=f.readlines()

    for line in lines:
        if "|" not in line:
            continue
        parts=line.strip().split('|')
        cityName=parts[0].split(':')[1].lower()
        nodeCount=int(parts[1].split(":")[1])

        numParts=nodeCount//2000

        if numParts<2:
            print(f"{cityName} has skipped because node is less than 4k)")
            continue

        print(f"\n processing {cityName}")

        #load city
        foldName=f"{cityName} rsults"
        graph=os.path.join(Processed,foldName,"graph_with_centrality.gml")
        if not os.path.exists(graph):
            print("!!!!!!!")
            print("!!!!!!!")
            print("file not found")
            print("!!!!!!!")
            print("!!!!!!!")
            continue
        G=nx.read_gml(graph)

        print(f"parition into {numParts} subgraphs")
        subset=partition(G,numParts)

        #save output

        outputDir=os.path.join(Processed,foldName,"partitions")
        os.makedirs(outputDir,exist_ok=True)
        for i, nodes in enumerate(subset):
            subG=G.subgraph(nodes).copy()
            filename =f"{cityName}_part_{i+1}.gml"
            nx.write_gml(subG,os.path.join(outputDir,filename))
            print(f"saved{filename}:{subG.number_of_nodes()} nodes")

    print("all partition done")



run()