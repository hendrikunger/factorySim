class FactoryThread(Thread):
    def __init__(self, multi, bb):
        Thread.__init__(self)
        self.multi = multi
        self.bb = bb
        self.daemon = True
        self.start()
        self.G = None
        self.I = None



    def run(self):
        walkableArea = self.bb.difference(unary_union(self.multi))
        if walkableArea.geom_type ==  'self.multiPolygon':
            walkableArea = walkableArea.geoms[0]


        #   Create Voronoi -----------------------------------------------------------------------------------------------------------------
        #Points around boundary

        distances = np.arange(0,  self.bb.boundary.length, BOUNDARYSPACING)
        points = [ self.bb.boundary.interpolate(distance) for distance in distances]

        #Points on Machines
        distances = np.arange(0,  self.multi.boundary.length, BOUNDARYSPACING)
        points.extend([ self.multi.boundary.interpolate(distance) for distance in distances])
        self.bb_points = unary_union(points) 

        voronoiBase = GeometryCollection([walkableArea, self.bb_points])
        voronoiArea = voronoi_diagram(voronoiBase, edges=True)

        route_lines = []
        lines_touching_machines = []


        processed_multi = prep(self.multi)
        processed_bb = prep(self.bb)

        for line in voronoiArea.geoms[0].geoms:
            #find routes close to machines
            if not (processed_multi.intersects(line) or processed_bb.crosses(line)): 
                route_lines.append(line)

        # Find closest points in voronoi cells
        hitpoints = points + list(MultiPoint(walkableArea.exterior.coords).geoms)
        #hitpoints = self.multiPoint(points+list(walkableArea.exterior.coords))
        hit_tree = STRtree(hitpoints)


        # Create Graph -----------------------------------------------------------------------------------------------------------------
        self.G = nx.Graph()

        memory = None
        memomry_distance = None

        for line in route_lines:

            first = line.boundary.geoms[0]
            firstTuple = (first.x, first.y)
            first_str = str(firstTuple)
            #find closest next point in boundary for path width calculation
            if memory == first:
                first_distance = memomry_distance
            else:
                nearest_point_first = hit_tree.nearest_geom(first)
                first_distance = first.distance(nearest_point_first)
                #nearest_point_first = nearest_points(hitpoints,first)[0]
                #first_distance = first.distance(nearest_point_first)

            second = line.boundary.geoms[1]
            secondTuple = (second.x, second.y)
            second_str = str(secondTuple)
            #find closest next point in boundary for path width calculation
            nearest_point_second = hit_tree.nearest_geom(second)
            second_distance = second.distance(nearest_point_second)
            #nearest_point_second = nearest_points(hitpoints,second)[0]
            #second_distance = second.distance(nearest_point_second)
            memory, memomry_distance = second, second_distance

            #edge width is minimum path width of the nodes making up the edge
            smallestPathwidth = min(first_distance, second_distance)


        #This is replaced by the version below. Delete Line Filtering below as well 
            self.G.add_node(first_str, pos=firstTuple)
            self.G.add_node(second_str, pos=secondTuple)
            self.G.add_edge(first_str, second_str, weight=first.distance(second), pathwidth=smallestPathwidth)


        #For later --------------------------
            # if smallestPathwidth < MINPATHWIDTH:
            #     continue
            # else:
            #     self.G.add_node(first_str, pos=firstTuple, pathwidth=smallestPathwidth)
            #     self.G.add_node(second_str, pos=secondTuple, pathwidth=smallestPathwidth)
            #     self.G.add_edge(first_str, second_str, weight=first.distance(second), pathwidth=smallestPathwidth)



        # Filter  Graph -----------------------------------------------------------------------------------------------------------------

        narrowPaths = [(n1, n2) for n1, n2, w in self.G.edges(data="pathwidth") if w < MINPATHWIDTH]
        self.G.remove_edges_from(narrowPaths)

        #Find largest connected component to filter out "loose" parts
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        self.G = self.G.subgraph(Gcc[0]).copy()

        #find crossroads
        old_crossroads = [node for node, degree in self.G.degree() if degree >= 3]
        #Set isCrossroads attribute on cross road nodes
        nx.set_node_attributes(self.G, dict.fromkeys(old_crossroads, True), 'isCrossroads')
        #find deadends
        old_endpoints = [node for node, degree in self.G.degree() if degree == 1]

        shortDeadEnds = pruneAlongPath(self.G, starts=old_endpoints, ends=old_crossroads, min_length=MINDEADEND_LENGTH)

        self.G.remove_nodes_from(shortDeadEnds)
        endpoints = [node for node, degree in self.G.degree() if degree == 1]
        crossroads = [node for node, degree in self.G.degree() if degree >= 3]

        # Prune unused dead ends
        pos=nx.get_node_attributes(self.G,'pos')

        repPoints = [poly.representative_point() for poly in self.multi.geoms]
        #Create Positions lists for nodes, since we need to querry shapley for shortest distance
        endpoint_pos = [pos[endpoint] for endpoint in endpoints ]
        crossroad_pos = [pos[crossroad] for crossroad in crossroads]
        total = endpoint_pos + crossroad_pos

        endpoints_to_prune = endpoints.copy()

        for point in repPoints:
            hit = nearest_points(point, MultiPoint(total))[1]
            key = str((hit.x, hit.y))
            if key in endpoints_to_prune: endpoints_to_prune.remove(key)

        nodes_to_prune = pruneAlongPath(self.G, starts=endpoints_to_prune, ends=crossroads, min_length=10)

        self.G.remove_nodes_from(nodes_to_prune)

        endpoints = [node for node, degree in self.G.degree() if degree == 1]
        crossroads = [node for node, degree in self.G.degree() if degree >= 3]

        nx.set_node_attributes(self.G, findSupportNodes(self.G, cutoff=SIMPLIFICATION_ANGLE))
        support = list(nx.get_node_attributes(self.G, "isSupport").keys())


        # Alternative Simpicification and Path Generation ------------------------------------------------------------------------------------
        self.I = nx.Graph()

        visited = set() # Set to keep track of visited nodes.
        tempPath = [] # List to keep track of visited nodes in current path.
        paths = [] # List to keep track of all paths.

        ep = endpoints
        cross = crossroads
        stoppers = set(ep + cross + support)

        if ep: 
            nodes_to_visit = [ep[0]]
        else:
            nodes_to_visit = [cross[0]]

        maxpath = 0
        minpath = float('inf')
        totalweight = 0
        currentInnerNode = None

        #DFS Start
        while(nodes_to_visit):

            currentOuterNode = nodes_to_visit.pop()
            if currentOuterNode in visited:
                continue
            else:
                visited.add(currentOuterNode)

            for outerNeighbor in self.G.neighbors(currentOuterNode):
                if outerNeighbor in visited: continue

                maxpath = 0
                minpath = float('inf')
                totalweight = 0

                lastNode = currentOuterNode
                currentInnerNode = outerNeighbor
                tempPath.append(currentOuterNode)


                while True:
                    #check if next node is deadend or crossroad
                    currentEdgeKey = str((lastNode,currentInnerNode))
                    totalweight += self.G[lastNode][currentInnerNode]["weight"]
                    pathwidth = self.G[lastNode][currentInnerNode]["pathwidth"]
                    maxpath = max(maxpath, pathwidth)
                    minpath = min(minpath, pathwidth)

                    if currentInnerNode in stoppers:
                        #found a crossroad or deadend
                        tempPath.append(currentInnerNode)
                        paths.append(tempPath)
                        
                        #Prevent going back and forth between direct connected crossroads 
                        if lastNode != currentOuterNode:
                            visited.add(lastNode)
                        nodes_to_visit.append(currentInnerNode)

                        pathtype = "oneway"
                        if minpath > MINTWOWAYPATHWIDTH: pathtype = "twoway"


                        self.I.add_node(currentOuterNode, pos=pos[currentOuterNode])
                        self.I.add_node(currentInnerNode, pos=pos[currentInnerNode])
                        self.I.add_edge(currentOuterNode, 
                            currentInnerNode, 
                            weight=totalweight,
                            pathwidth=minpath, 
                            max_pathwidth=maxpath, 
                            nodelist=tempPath,
                            pathtype=pathtype
                        )
                        tempPath = [] 
                        break 
                    else:
                        #going along path
                        tempPath.append(currentInnerNode)

                    for innerNeighbor in self.G.neighbors(currentInnerNode):
                        #Identifying next node (there will at most be two edges connected to every node)
                        if (innerNeighbor == lastNode):
                            #this is last node
                            continue
                        else:
                            #found the next one
                            lastNode = currentInnerNode
                            currentInnerNode = innerNeighbor
                            break