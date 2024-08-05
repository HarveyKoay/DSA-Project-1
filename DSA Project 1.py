import math

# ==========
# Q1

def fuse(fitmons: list[list[float, int, float]]) -> int:
    """
    Func Desc:
    Essentially what this does is that it fuses the adjacent FITMONs or list to create a final FITMON. Each FITMON's cuteness score
    and affinities contribute to the cuteness score of the resulting FITMON. The goal is to maximize the cuteness score through 
    strategic fusions.
    Written by Harvey Koay Wern Shern
    
    Approach Desc:
    The function follows a dynamic programming approach to fuse FITMONs efficiently. It initializes a memoization
    matrix depending on the the number of fitmons to store the maximum cuteness score achievable by fusing a 
    range of FITMONs. It then iterates over increasing lengths of FITMON combinations, calculating the maximum 
    cuteness score for each combination based on the fusion of adjacent FITMONs. This approach then uses the memoisation
    table which will reuse the overlapping values from the smaller combinations on the larger combinations to ensure a 
    faster time complexity. 

    Input:
        fitmons (list): A list of FITMONs to fuse. Each FITMON is basically represented by a list containing three values:
                        [affinity_left, cuteness_score, affinity_right].
                        - affinity_left is a Positive float (0.1 to 0.9) representing left affinity except the 
                        leftmost fitmon since it has no one to fuse with on the left
                        - cuteness_score is a Non-zero positive integer representing the FITMON's cuteness score.
                        - affinity_right is a Positive float (0.1 to 0.9) representing right affinity except the right most
                        fitmon since it has no one to fuse with on the right

    Return:
        cuteness_score (int): The cuteness score of the ultimate FITMON after fusing all FITMONs.

    Time complexity:
        Best: O(N^2)
        Worst: O(N^3)
        Where N is the number of FITMONs in the input list.

    Time complexity analysis:
        Best: It occurs when there are only two fitmons thus making the N to be 2. The outer loop only
        runs once which effectively enable us to only combine 2 fitmons leading to a time complexity of N.
        However. Since the auxilary space complexity is still O(N^2). It will still result in a best case 
        overall time complexity of O(N^2). 
        
        Worst: The function iterates over all possible combinations of FITMONs, where it enters into three loops 
        respectively. This is to use the memoisation table filling it up diagonally and eventually leading to 
        cubic time complexity.

    Space complexity:
        Input: O(N^2)
        Aux: O(N^2)
        Where N is the number of FITMONs in the input list.

    Space complexity analysis:
        Input: The function takes in an input list of FITMONs which is stored in a 2D list.
        Aux: The function utilizes a memoization matrix to store intermediate results, resulting in quadratic
        space complexity. Additionally, the matrix dimensions are proportional to the square of the input
        size, contributing to the overall space complexity of O(N^2)
    """
    n = len(fitmons) 

    # Memoisation matrix
    memo = [[0] * n for _ in range(n)]

    # Base case to add the fitmon to the matrix
    for i in range(n):
        memo[i][i] = fitmons[i][1] 

    #l is the fitmon length
    for l in range(2, n+1): 
        # Get the starting index of the fitmon
        for i in range(n-l+1):
            # Get the ending index of the fitmon
            j = i + l - 1
            max_cuteness = 0
            # Get the cuteness score of the fitmon
            for k in range(i, j): 
                cuteness = int(memo[i][k] * fitmons[k][2] + memo[k+1][j] * fitmons[k+1][0])
                # Update the max cuteness score
                max_cuteness = max(cuteness, max_cuteness)
            # Update the memoisation matrix
            memo[i][j] = max_cuteness
    
    # Return the cuteness score of the ultimate FITMON in the top right corner of the memoisation matrix
    return memo[0][n-1]



# ==========
# Q2
class Edge:
    """
    Edge class to represent an edge in the graph.
    """
    def __init__(self, u: int, v: int, w: float) -> None:
        """
        The constructor for the Edge class.
        Initialize the edge with a source vertex, a destination vertex, and a weight.
        
        Input:
            u (int): The source vertex id.
            v (int): The destination vertex id.
            w (float): The weight of the edge.
        
        Return:
            None
        
        Time complexity:
            Best = Worst = O(1) as there are only constant time operations.
        
        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """
        self.u = u
        self.v = v
        self.w = w
    
class Vertex:
    """
    Vertex class to represent a vertex in the graph. 
    """
    def __init__(self, id: int) -> None:
        """
        The constructor for the Vertex class.
        Initialize the vertex with an ID and an empty list of edges. Set the vertex as not discovered and not visited.
        The distance is set to 0 and the previous vertex is set to None. 

        Input:
            id (int): The ID of the vertex.

        Return:
            None

        Time complexity:
            Best = Worst = O(1) as there are only constant time operations.

        Space complexity: 
            Input = Aux = O(1) as there are only constant space operations.
        """
        self.id = id 
        self.edges = []
        self.discovered = False
        self.visited = False
        self.distance = 0

        # Used for backtracking/ where i was from
        self.previous = None

        # Used for the heap since we need to update the position of the vertex in the heap
        # This is essentially the index mapping array
        self.position = 0

    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge from the current vertex to another vertex.

        Input:
            edge (Edge): The edge to add to the current vertex.
        Return:
            None

        Time complexity: 
            Best = Worst = O(1) as there are only constant time operations.
        Space complexity: 
            Input = Aux = O(1) as there are only constant space operations.
        """
        self.edges.append(edge)

    def backtrack_route(self, source, destination, lst_of_vertices) -> list:
        """
        Backtrack from the exit tree to the starting tree to obtain the shortest route.

        Input:
            source (Vertex): The starting tree.
            destination (Vertex): The exit tree.
            lst_of_vertices (list of Vertex): List of vertices in the graph.

        Return:
            list: List of tree nodes representing the shortest route from the starting tree to the exit tree.

        Time complexity: 
            Best = O(1) when the source is the destination and the solulu tree is at the same position.
            It will only backtrack once. The best case can also happen when the destination is unreachable.
            This will mean that the previous vertex will be None. Thus, the function will return an empty list.

            Worst = O(V) when the destination is reachable and the source is not the destination. The function will
            backtrack from the destination to the source.

            N is the number of vertices in the graph.

        Space complexity: 
            Input = O(V) where V is the number of vertices in the graph.
            Aux = O(N) as the function uses a list to store the route from the destination to the source.
        """
        # Initialize the route list
        route = []

        # Start from the destination vertex
        current_vertex = destination

        # Backtrack until the source vertex is reached
        while current_vertex is not None:
            # Move to the previous vertex since i dont add the final added vertex
            current_vertex = current_vertex.previous
            # Add the vertex to the route list if it is not the source vertex
            if current_vertex is not source and current_vertex is not None:
                # Get the actual vertex id since the graph is split into two
                if current_vertex.id >= len(lst_of_vertices) // 2:
                    new_id =current_vertex.id - len(lst_of_vertices) // 2
                else:
                    new_id = current_vertex.id
                # Add the vertex id to the route list if it is not the previous vertex
                if current_vertex.previous is not None:
                    if new_id != current_vertex.previous.id:
                        route.append(new_id)
            else:
                break

        # Add the source vertex to the route list if the route is not empty
        if route != []:
            route.append(source.id)
        return route[::-1]
    
class MinHeap:
    """
    MinHeap class to represent a min heap data structure.
    """
    def __init__(self, size: int) -> None:
        """
        The constructor for the MinHeap class.
        Initialize the heap with a size and an array to store the heap elements.
        The self.array has a length of size + 1 to account for the 1-based indexing and ensure 
        the first item of the heap stays as None. The length of the heap is set to the size of the array minus 1.

        Input:
            size (int): The size of the heap.

        Return:
            None
        
        Time complexity:
            Best = Worst = O(V) as the function initializes the heap with a size of V.
            V is the number of vertices in the graph.
        
        Space complexity:
            Input = Aux = O(V) as the function initializes the heap with a size of V.
        """
        # Initialize array
        self.array = [None] * (size+1)  
        self.length = len(self.array) -1
    
    def swap(self, i: int, j: int) -> None:
        """
        Swap the elements at indices i and j in the heap array.

        Input:
            i (int): The index of the first element to swap.
            j (int): The index of the second element to swap.

        Return:
            None
        
        Time complexity:
            Best = Worst = O(1) as there are only constant time operations.

        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """
        # Swap the elements
        self.array[i], self.array[j] = self.array[j], self.array[i]
        # Update the vertex position in the index map
        self.array[i][0].position = i
        self.array[j][0].position = j

    def rise(self, k: int) -> None:
        """
        Rise element at index k to its correct position.

        Input:
            k (int): The index of the element to rise.

        Return:
            None

        Time complexity:
            Best = O(1) when the element is already at its correct position or the element is the root of the heap.
            Worst = O(log V) as the function rises the element at index k to its correct position in the heap.
            V is the number of vertices in the graph.

        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """
        # Get the item at index k
        item = self.array[k]
        # Rise the element to its correct position comparing with its parent
        while k > 1 and item[1] < self.array[k // 2][1]:
            self.swap(k, k // 2)
            k = k // 2

        # Update vertex position
        self.array[k][0].position = k  

    def add(self, element: tuple) -> None:
        """
        Add an element to the heap.

        Input:
            element - Tuple(Vertex, int): The element to add to the heap. The element is a tuple containing a vertex and a distance.
        
        Return:
            None
        
        Time complexity:
            Best = Worst = O(log V) as the function adds an element to the heap and rises it to its correct position.
            V is the number of vertices in the graph.
        
        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """
        self.array[element[0].position] = element
        # Rise the element to its correct position
        self.rise(element[0].position)

    def smallest_child(self, k: int) -> int:
        """
        Get the index of the smallest child of the element at index k.

        Input:
            k (int): The index of the element to find the smallest child.

        Return:
            int: The index of the smallest child of the element at index k.

        Time complexity:
            Best = Worst = O(1) as there are only constant time operations.

        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """
        if 2 * k == self.length or self.array[2 * k][1] < self.array[2 * k + 1][1]:
            return 2 * k
        else:
            return 2 * k + 1

    def sink(self, k: int) -> None:
        """
        Sink element at index k to its correct position.

        Input:
            k (int): The index of the element to sink.

        Return:
            None

        Time complexity:
            Best = O(1) when the element is already at its correct position or the element is a leaf node.
            Worst = O(log V) as the function sinks the element at index k to its correct position in the heap.
            V is the number of vertices in the graph.

        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """
        item = self.array[k]

        while 2 * k <= self.length:
            min_child = self.smallest_child(k)
            if item[1] <= self.array[min_child][1]:
                break
            self.swap(k, min_child)
            k = min_child

        # Update vertex position
        self.array[k][0].position = k  

    def get_min(self) -> tuple:
        """
        Get the minimum element from the heap or since it is a min heap we get the root of the heap.

        Input:
            None
        Return:
            tuple (Vertex, int): The minimum element from the heap. The element is a tuple containing a vertex and a distance.
        
        Time complexity:
            Best = O(1) when the new root is already at its correct position.
            Worst = O(log V) as the function gets the minimum element from the heap and sinks the new root to its correct position.
            V is the number of vertices in the graph.
            
        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """
        # Get the smallest vertex from the heap
        smallest_vertex = self.array[1]
        self.length -= 1

        # Swap the root with the last element and sink the new root to its correct position
        if self.length > 0:
            self.array[1] = self.array[self.length + 1]
            self.sink(1)
        
        # Update vertex position in the index map array
        smallest_vertex[0].position = self.length + 1  
        return smallest_vertex

    def update(self, vertex: Vertex, new_distance: float) -> None:
        """
        Update the distance of a vertex in the heap.

        Input:
            vertex (Vertex): The vertex to update.
            new_distance (float): The new distance of the vertex.

        Return:
            None
        
        Time complexity:
            Best = O(1) when the new distance is greater than the old distance.
            Worst = O(log V) as the function updates the distance of the vertex and rises or sinks the vertex to its correct position.
            V is the number of vertices in the graph.

        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """

        # Get the index of the vertex in the heap
        index = vertex.position
        # Get the old distance of the vertex
        old_distance = self.array[index][1]

        # Update the distance if the new distance is less than the old distance 
        # and rise or sink the vertex to its correct position
        if new_distance < old_distance:
            self.array[index] = (vertex, new_distance)
            self.rise(index) 
            self.sink(index)


class Graph:
    """
    Graph class to represent a graph data structure.
    """
    def __init__(self, vertices_count: int) -> None:
        """
        The constructor for the Graph class.
        Initialize the graph with a number of vertices and an empty list of vertices.

        Input:
            vertices_count (int): The number of vertices in the graph.

        Return:
            None

        Time complexity:
            Best = Worst = O(V) as the function initializes the graph with V vertices.

        Space complexity:
            Input = O(1) as there are only constant space operations.
            Aux = O(V) as the function initializes the graph with V vertices.
        """
        self.vertices = [None] * vertices_count
        # Instantiate the vertices
        for i in range(vertices_count):
            self.vertices[i] = Vertex(i)
    
    def reset(self) -> None:
        """
        Remove the last edge if the edge is to the exit vertex. This is to ensure that the exit vertex 
        is not connected to any other vertex. This ensures that the exit vertex is only connected to the 
        new exit vertex with distance 0. Reset the graph by setting the discovered and visited status of 
        the vertices to False. Set the distance of the vertices to 0 and the previous vertex to None.
        
        Input:
            None

        Return:
            None

        Time complexity:
            Best = Worst = O(V) as the function resets the graph by iterating over V vertices.
        
        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """

        for vertex in self.vertices:
            if len(vertex.edges) > 0 and vertex.edges[len(vertex.edges)-1].v == (len(self.vertices)-1):
                vertex.edges.pop()
            vertex.discovered = False
            vertex.visited = False
            vertex.distance = 0
            vertex.previous = None

    def add_edges(self, edges: list[tuple]) -> None:
        """
        Add edges to the graph.

        Input:
            edges (list): List of tuples representing edges in the graph.

        Return:
            None

        Time complexity:
            Best = Worst = O(E) as the function adds edges to the graph.
            E is the number of edges or the number of roads in the input list.

        Space complexity:
            Input = O(E) where E is the number of edges in the graph.
            Aux = O(E) where E is the number of edges in the graph.
        """
        # add u to v for first part of the graph
        for edge in edges:
            u = edge[0]
            v = edge[1]
            w = edge[2]
            # add an edge from u to v
            current_edge = Edge(u,v,w)
            current_vertex = self.vertices[u]
            current_vertex.add_edge(current_edge)
        
        # add u to v for second part of the graph
        for edge in edges:
            u = int(edge[0] + len(self.vertices)/2)
            v = int(edge[1] + len(self.vertices)/2)
            w = edge[2]
            current_edge = Edge(u,v,w)
            current_vertex = self.vertices[u]
            current_vertex.add_edge(current_edge)
        
    
    def add_cross_edges(self, edges: list[tuple]) -> None:
        """
        Add cross edges to the graph.
        This cross edges are the edges that connect the two parts of the graph.
        It is basically making the graph a multiverse graph. where there is a teleportaion 
        from one part of the graph to the other.

        Input:
            edges (list): List of tuples representing cross edges in the graph.

        Return:
            None

        Time complexity:
            Best = Worst = O(C) as the function adds cross edges to the graph.
            C is the number of cross edges in the input list or the number of solulu trees.

        Space complexity:
            Input = O(C) where C is the number of cross edges in the input list. 
            Aux = O(C) where C is the numbert of cross edges in the input list.
        """
        # add u in the first part to v for the second part of the graph
        for edge in edges:
            u, w, v = edge
            # v is represented as the exit vertex of the solulu tree which is in the second part of the graph
            v = int(v + len(self.vertices)/2)
            current_edge = Edge(u, v, w)
            current_vertex = self.vertices[u]
            current_vertex.add_edge(current_edge)

    def add_initial_heap(self, discovered: MinHeap, source: Vertex) -> None:
        """
        Add the initial vertices to the heap. The function initializes the heap with vertices and distances.
        The function adds the vertices to the heap based on their ID. The source vertex is added first to the heap.
        The vertices have a distance of infinity and are not visited or discovered.

        Input:
            discovered (MinHeap): The heap to add the vertices to.
            source (Vertex): The source vertex.
        
        Return:
            None
        
        Time complexity:
            Best = Worst = O(V) as the function adds the initial vertices to the heap.
            N is the number of vertices in the graph.

        Space complexity:
            Input = O(V) where N is the number of vertices in the graph.
            Aux = O(1) as there are only constant space operations.
        """
        for i, vertex in enumerate(self.vertices):
            # Add the vertex to the heap if it is not the source vertex
            if vertex != source:
                # If the vertex ID is less than the source vertex ID, add the vertex to the heap with a position of i + 2
                vertex.distance = math.inf
                if vertex.id < source.id:
                    vertex.position = i +2
                    discovered.add((vertex, vertex.distance)) 
                # If the vertex ID is greater than or equal to the source vertex ID, add the vertex to the heap with a position of i + 1
                else:
                    vertex.position = i + 1
                    discovered.add((vertex, vertex.distance))
        
    def dijkstra(self, source: Vertex, destination: Vertex) -> None:
        """
        Perform Dijkstra's algorithm to find the shortest path from the source vertex to the destination vertex.
        The function initializes a MinHeap to store the vertices and their distances. The source vertex is added to the
        root of the heap. The function adds the initial vertices to the heap and updates the heap with the vertices 
        and their distances. The function iterates over the heap to find the shortest path from the source vertex until
        it reaches the destination vertex or until the heap is empty.

        Input:
            source (Vertex): The source vertex.
            destination (Vertex): The destination vertex.

        Return:
            None

        Time complexity:
            Best = O(V) when the destination vertex is the source vertex. The while loop will only run once
            Worst = O(E * log V) The worst case occurs when the destination vertex is reachable and the source 
            vertex is not the destination vertex. as the function performs Dijkstra's algorithm to find the 
            shortest path from the source vertex to the destination vertex. The function iterates over the vertices
            and edges in the graph to find the shortest path. 
            V is the number of vertices in the graph.
            E is the number of edges of each vertex in the graph.
        Space complexity:
            Input = O(1) as there are only constant space operations.
            Aux = O(V) where a MinHeap is used to store the vertices and their distances.
        """
        discovered = MinHeap(len(self.vertices))
        # Add the source vertex with its distance to the MinHeap
        source.position = 1
        discovered.add((source, source.distance))  

        self.add_initial_heap(discovered, source)

        while discovered.length > 0:

            u, _ = discovered.get_min()
            u.visited = True
            if u == destination:
                return
            for edge in u.edges:
                v = self.vertices[edge.v]
                if not v.discovered:
                    v.discovered = True
                    # for backtracking
                    v.distance = u.distance + edge.w
                    v.previous = u
                    discovered.add((v, v.distance))
                # it is in heap but not yet finalised
                elif not v.visited and v.distance > u.distance + edge.w:
                    # update distance
                    v.distance = u.distance + edge.w
                    v.previous = u
                    # update heap
                    discovered.update(v, v.distance)

class TreeMap:
    """
    TreeMap class to represent a graph data structure.
    """

    def __init__(self, roads: list[tuple], solulus: list[tuple]) -> None:
        """
        Function Desc:
        The constructor for the TreeMap class which instantiates the graph and solulu teleportation trees. 
        Written by Harvey Koay Wern Shern
        
        Approach Desc:
        The function creates a 2 graphs representation of the forest and adds the roads to the tree map.
        It is then represented as a single connected tree map in a single min heap. The function combines the start of
        solulu trees to graph 1 and end of soulu trees to graph 2.
        
        Input:
            roads (list): List of tuples representing roads between trees.
            solulus (list): List of tuples representing solulu trees.
        Return:
            None
            
        Time complexity: 
            Best = Worst =  O(R + T + T) = O(R + 2T) = O(R + T)
            R is the number of roads in the tree map
            T is the number of trees in the tree map.

        Time complexity Analysis: 
            Best = Worst = The maximum number of solulu trees is T .The function creates a graph 
            representation of the forest. It also adds the roads to the tree map. It also combines 
            the start of solulu trees to graph 1 and end of soulu trees to graph 2. 

        Space complexity: 
            Input: O(R + T)
            Aux: O(R + T)
            R is the number of roads in the tree map 
            T is the number of trees in the tree map.

        Space complexity Analysis:
            Input = The function takes in a list of roads which is stored in a list of tuples.
            It also takes in a list of solulu trees which is stored in a list of tuples. The maximum number of solulu 
            trees is T which is the number of trees in the tree map.
            Aux = The function creates a graph representation of the forest. It also adds the roads to the tree map.
        """
        # Create a graph representation of the forest
        self.graph = self.create_graph(roads)

        # Combine the start of solulu trees to graph 1 and end of soulu trees to graph 2
        self.graph.add_cross_edges(solulus)

        # Add an final vertex with an edge to the final vertex
        self.add_exit_vertex(self.graph)
        
    def create_graph(self, roads) -> Graph:
        """
        This function creates a 2 graphs representation of the forest and adds the roads to the tree map.
        It is then represented as a single connected tree map in a single min heap.

        Input:
            roads (list): List of tuples representing roads between trees.
        Returns:
            Graph: Graph representation of the forest.

        Time complexity:
            Best = Worst = O(R + 2T) = O(R + T) since it doubles the number of trees in the tree map.
            The function also reates a graph representation of the forest. It also adds the roads to the tree map. 
            T is the number of trees in the tree map and
            R is the number of roads in the tree map.

        Space complexity:
            Input = O(R) where R is the number of roads in the tree map.
            Aux = O(T + R) as the function creates a graph representation of the forest.
            It also adds the roads to the tree map. The route can go through all the trees in the forest.
        """
        total_roads = 0
        for road in roads:
            total_roads = max(max(road[0], road[1]), total_roads) 
        graph = Graph((total_roads + 1) * 2)
        graph.add_edges(roads)

        return graph

    def add_exit_vertex(self, graph: Graph) -> None:
        """
        Add an exit vertex to the graph which is the last vertex in the graph.

        Input:
            graph (Graph): The graph representation of the forest.
        Returns:
            None
        
        Time complexity:
            Best = Worst = O(1) as there are only constant time operations.
        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """
        # Assuming that the exit vertex ID is the maximum vertex ID + 1
        last_vertex_id = len(graph.vertices)
        random_vertex = Vertex(last_vertex_id)
        graph.vertices.append(random_vertex)

    def escape(self, start: int, exits: list[int]) -> tuple[int, list[int]]:
        """
        Function Desc:
        The function finds the shortest time and route to escape the forest. It uses Dijkstra's algorithm to find the
        shortest path from the start tree to the exit tree. The function resets the graph and adds the roads to the tree map.
        
        Approach Desc:
        The function resets the graph everytime it runs and so it can be used multiple times. This is to ensure that the
        final vertex is not connected to any other vertex whhen i want to rerun dijkstra as it will mess the algorithm up.
        I then add an exit vertex connected to all destination vertices with distance 0. This will ensure that the one to 
        multiple destination will work for the dijkstra algorithm by running dijkstra from the source to the final vertex.
        I then get the shortest time and route from the start tree to the exit tree if it is reachable. Finally, I return 
        the shortest time and route for the current exit tree. The shortest route is done by backtracking from the exit 
        tree to the starting tree to obtain the shortest route. The function returns None if the destination is unreachable.

        Input:
            roads (list): List of tuples representing roads between trees.
            solulus (list): List of tuples representing solulu trees.
        Return:
            shortest_time (int): The shortest time to escape the forest.
            shortest_route (list): List of ID of tree nodes representing the shortest route from the starting tree to the exit tree.
            
        Time complexity: 
            Best = O(T + T + e)  = O(2T + e) = O(T)
            Worst = O(R log T + T + e + T) = O(R log T + 2T + e) = O(R log T)
            R is the number of roads in the tree map 
            T is the number of trees in the tree map
            e is the number of exits in the tree map

        Time complexity Analysis: 
            Best = This occurs when the destination is unreachable. The function will return None. It also does not need to
            backtrack since the destination is unreachable. The time complexity is O(T).
            Worst = The function run dijkstra. There are multiple exit trees. However, since the number of exit
            will be less than the number of trees in the forest, we can say that the time complexity is O(R log T).
            I assume that since the number of trees are also less than the number of roads, the time complexity is O(R log T).

        Space complexity: 
            Input = O(e)
            Aux = O(R + T + T) = O(R + 2T) = O(R + T)
            R is the number of roads in the tree map
            T is the number of trees in the tree map

        Space complexity Analysis: 
            Input = The function takes in a list of exits which is stored as a list of integers
            Aux = The function runs dijkstra. The function creates a min heap which has a space complexity of O(T).
            It also adds the roads to the tree map. The route can go through all the trees in the forest.
            Thus, the space complexity is O(R + T).
        """
        self.graph.reset()
        last_vertex_id = len(self.graph.vertices) -1
        # Add an exit vertex connected to all destination vertices with distance 0
        for exit in exits:
            exit = exit + int(len(self.graph.vertices)/2)
            # Add an edge from the exit tree to the random vertex with distance 0
            self.graph.vertices[exit].add_edge(Edge(exit, last_vertex_id, 0))

        shortest_time = math.inf
        shortest_route = []
        source = self.graph.vertices[start]
        destination = self.graph.vertices[len(self.graph.vertices) - 1]
        self.graph.dijkstra(source, destination)

        # Get the shortest time and route for the current exit tree
        shortest_time = destination.distance
        shortest_route = destination.backtrack_route(source ,destination, self.graph.vertices)
        
        # Return the shortest time and route for the current exit tree if it is reachable
        if shortest_time == math.inf:
            return None
        else:
            return shortest_time, shortest_route
        
