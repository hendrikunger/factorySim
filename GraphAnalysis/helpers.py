import networkx as nx

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
