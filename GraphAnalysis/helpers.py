import numpy as np
import networkx as nx


def calculateNodeAngles(G, cutoff=45):

    candidates = [node for node, degree in G.degree() if degree == 2]

    node_data ={}
    pos=nx.get_node_attributes(G,'pos')

    for node in candidates: 
        neighbors = list(G.neighbors(node))
        
        vector_1 = [pos[node][0] - pos[neighbors[0]][0], pos[node][1] - pos[neighbors[0]][1]]
        vector_2 = [pos[neighbors[1]][0] - pos[node][0], pos[neighbors[1]][1] - pos[node][1]]

        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.rad2deg(np.arccos(dot_product))


        #angle = np.arctan2(*pos[neighbors[0]]) - np.arctan2(*pos[neighbors[1]])
        node_data[node] = {"edge_angle" : angle}

        # if(angle < -np.pi/12  or angle > np.pi/12):
        #     node_data[node] = {"isSupport" : True}

        if(abs(angle) > cutoff):
             node_data[node] = {"isSupport" : True}

    return node_data


def findSupportNodes(G, cutoff=45):

    candidates = [node for node, degree in G.degree() if degree == 2]

    node_data ={}
    pos=nx.get_node_attributes(G,'pos')

    for node in candidates: 
        direct_neighbors = list(G.neighbors(node))
        
        pre = list(G.neighbors(direct_neighbors[0]))
        pre.remove(node)

        if len(pre) == 1:
            vector_1 = [pos[node][0] - pos[pre[0]][0], pos[node][1] - pos[pre[0]][1]]
        else:
            #direct_neighbors[0] is a endpoint or crossroad, can not use its neighbors for calculation
            vector_1 = [pos[node][0] - pos[direct_neighbors[0]][0], pos[node][1] - pos[direct_neighbors[0]][1]]

        suc = list(G.neighbors(direct_neighbors[1]))
        suc.remove(node)

        if len(suc) == 1:
            vector_2 = [pos[suc[0]][0] - pos[node][0], pos[suc[0]][1] - pos[node][1]]
        else:
            #direct_neighbors[1] is a endpoint or crossroad, can not use its neighbors for calculation
            vector_2 = [pos[direct_neighbors[1]][0] - pos[node][0], pos[direct_neighbors[1]][1] - pos[node][1]]

        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.rad2deg(np.arccos(dot_product))

        #angle = np.arctan2(*pos[neighbors[0]]) - np.arctan2(*pos[neighbors[1]])
        node_data[node] = {"edge_angle" : angle}

        # if(angle < -np.pi/12  or angle > np.pi/12):
        #     node_data[node] = {"isSupport" : True}

        if(abs(angle) > cutoff):
             node_data[node] = {"isSupport" : True}

    for node in dict(node_data):
        if node_data[node].get("isSupport", False):
            direct_neighbors = G.neighbors(node)
            for neighbor in direct_neighbors:
                if neighbor in node_data:
                    if node_data[neighbor].get("isSupport", False):
                        node_data.pop(node, None)



    return node_data


def pruneAlongPath(F, starts=[], ends=[], min_length=1):
    shortDeadEnds =[]

    for seed in starts:

        total_length = 0
        currentNode = seed
        nextNode = None
        lastNode = None
        tempDeadEnds = []
        #Follow path from endnnode to next crossroads, track length of path
        while True:
            
            for neighbor in F.neighbors(currentNode):
                #Identifying next node (there will at most be two edges connected to every node)
                if (neighbor == lastNode):
                    #this is last node
                    continue
                else:
                    #found the next one
                    nextNode = neighbor
                    break
            # keep track of route length
            total_length += F.edges[currentNode, nextNode]["weight"]
            #Stop if route is longer than min_length
            if total_length > min_length:
                break

            if nextNode in ends:
                tempDeadEnds.append(currentNode)
                shortDeadEnds.extend(tempDeadEnds)
                break 
            else:
                tempDeadEnds.append(currentNode)
                lastNode = currentNode
                currentNode = nextNode
    
    return shortDeadEnds





#Backup
# shortDeadEnds =[]

#     for endpoint in old_endpoints:

#         total_length = 0
#         currentNode = endpoint
#         nextNode = None
#         lastNode = None
#         tempDeadEnds = []

#         #Follow path from endnnode to next crossroads, track length of path
#         while True:
#             for neighbor in F.neighbors(currentNode):
#                 #Identifying next node (there will at most be two edges connected to every node)
#                 if (neighbor == lastNode):
#                     #this is last node
#                     continue
#                 else:
#                     #found the next one
#                     nextNode = neighbor
#                     break
#             # keep track of route length
#             total_length += F.edges[currentNode, nextNode]["weight"]
#             #Stop if route is longer than min_length
#             if total_length > MINDEADEND_LENGTH:
#                 break

#             if "isCrossroads" in F.nodes[nextNode]:
#                 tempDeadEnds.append(currentNode)
#                 shortDeadEnds.extend(tempDeadEnds)
#                 break
#             else:
#                 tempDeadEnds.append(currentNode)
#                 lastNode = currentNode
#                 currentNode = nextNode
