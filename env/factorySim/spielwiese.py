# %%
from matplotlib import pyplot as plt
from matplotlib.patches import Arc
import numpy as np
import networkx as nx


# %%

a = np.array([3,5])
b = np.array([-4,-2])
origin = np.array([[0],[0]])

#node_list = np.array([[7,16],[4,10],[1,6],[6,8],[10,4],[2,2]])
node_list = np.array([[0,0],[4,2],[8,5],[12,8],[13,14],[15,16]])

# create Graph
G = nx.Graph()
for i, element in enumerate(node_list):
    G.add_node(i, pos=element)
    if i>0:
        G.add_edge(i-1,i)

pos = nx.get_node_attributes(G, "pos")
#%%
fig, ax = plt.subplots(figsize=(7,7))
plt.xlim(0, 16)
plt.ylim(0, 16)

#loop over edges and plot Graph edges as lines
previous = None
r = 1
for n in G.nodes():
    if previous is not None:
        neighbors = list(G.neighbors(n))
        if len(neighbors) == 2:
            vector_1 = pos[neighbors[0]] - pos[n]
            vector_2 = pos[neighbors[1]] - pos[n]
            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.rad2deg(np.arccos(np.clip(dot_product, -1.0, 1.0)))      
            print(angle)

            
            theta1 = np.rad2deg(np.arctan2(vector_1[1],vector_1[0]))
            theta2 = np.rad2deg(np.arctan2(vector_2[1],vector_2[0]))
            vector_1 = pos[n] - pos[neighbors[0]]
            gamma1 = np.rad2deg(np.arctan2(vector_1[1],vector_1[0]))
            gamma2 = np.rad2deg(np.arctan2(vector_2[1],vector_2[0]))

            if gamma1 < gamma2:
                theta1, theta2 = theta2, theta1

            ax.add_patch(Arc(pos[n],r,r, theta1=theta1, theta2=theta2,  color='blue'))

        plt.arrow(*pos[previous], *(pos[n]-pos[previous]), color="red", length_includes_head=True, head_width=0.1, head_length=0.3)
    
    previous = n

plt.show()

  # %% 

fig, ax = plt.subplots(1)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.quiver(*origin, *a, color="red", scale_units='xy', angles='xy', scale=1)
ax.quiver(*origin, *b, color="blue", scale_units='xy', angles='xy', scale=1)


plt.show()
# %%
import numpy as np
a = np.array([3,5, 7])
b = np.array([9,6, 8])
temp= np.power(a/b,2)
print(temp)
print(np.mean(temp))
# %%
