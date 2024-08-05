from collections import deque

# ==========
# Q1
class Node:
    """
    Node class to represent a Node in the graph. 
    """
    def __init__(self):
        """
        The constructor for the Node class.
        Initialize the node with an empty list of children and the initial data to be empty

        Input:
            None

        Return:
            None

        Time complexity:
            Best = Worst = O(1) as there are only constant time operations.

        Space complexity: 
            Input = Aux = O(1) as there are only constant space operations.
        """
        # Assuming only uppercase letters [$ + A-D]
        self.children = [None] * 4
        self.data = []

class OrfFinder:
    """
    OrfFinder class which represents a suffix trie
    """
    def __init__(self, genome: str) -> None:
        """
        Function Desc:
        The constructor for the OrfFinder class which instantiates the suffix trie.
        Written by Harvey Koay

        Apporach Desc:
        It creates the suffix trie by adding each suffix of the genome into the suffix trie. This utilises 
        the insert_suffix method which is a recursion. Loops through based on the len of the genome and 
        insert the genome in the list 1 by 1 at the front of the list from the back which creates a suffix 
        every time and places it into the list.

        Input:
            genome (string): single non-empty string consisting only of uppercase [A-D]

        Return:
            None

        Time complexity:
            Best = Worst = O(N * (N+N))
            N is the length of the genome

        Time complexity analysis:
            Best = Worst = The function creates a suffix trie representation of the OrfFinder.
            The for loop runs in O(N) time as it uses the length of the genome. The inserting of
            the genome in the front of the list also takes O(N) time. Lastly, the insert_suffix
            method is a recursion where it recurse through the entire len of the suffix until
            the idx reaches a the length of the suffix which results in O(N) as well.

        Space complexity: 
            Input = O(N)
            Aux = O(N)
            N is the length of the genome

        Space complexity Analysis:
            Input = The function takes in a string which is the genome
            Aux = The function creates a list to store the entire genome in reverse order
        """
        self.genome = genome
        # Create a node as the root
        self.root = Node()
        # Build the suffix trie using the genome
        lst = []
        for i in range(len(genome)):
            # inserting the element from the end of the genome to the start at the start of list
            lst.insert(0, self.genome[len(genome)-1-i])
            # The data of the node is the index of that specific character in the string
            self.insert_suffix(lst, data = len(genome)-1-i)


    def insert_suffix(self, suffix: str, node=None, idx=0, data = None) -> None:
        """
        Inserting the suffix recursively in the children of each node in the graph. This function
        essentially creates a trie with the data being the index of that character in the string/ genome.

        Input:
            suffix (list[str]): A list containing the suffixes of the string 
            node (Node): The new node after creating a new node as a children of the original node.
            idx (int): Integer value to store the idx to check the character of each suffix
            data (int): The index of that character in the genome/ string

        Return:
            None

        Time complexity:
            Best = Worst = O(N) where the it calls itself recursively until the index reaches
            the length of the entire suffix.
            N is the length of the suffix

        Space complexity: 
            Input = O(N) as N is the length of the suffix
            Aux = O(1) as it is all constant time operations
        """
        if node is None:
            node = self.root

        if idx == len(suffix):                    
            return

        char = suffix[idx]
        char_index = ord(char) - ord('A')
        if node.children[char_index] is None:
            node.children[char_index] = Node()

        node.children[char_index].data.append(data)

        self.insert_suffix(suffix, node.children[char_index], idx + 1, data + 1)
    

    def find(self, start: str, end: str) -> list[str]:
        """
        Function Desc:
        The function effectively finds all the substrings of genome which have start as a prefix 
        and the end as a suffix where the start and end cannot overlap
        Written by Harvey Koay

        Apporach Desc:
        It uses the find_start and find_end functions to search for the ending indices of both the suffix 
        and the prefix. The end indices of the prefix are being processed to find the next index. The 
        end indices of the suffix are being processed to find the first index of the suffix. From there
        we can loop from each of the start index to each of the end index in between adding the 
        'start' at the beginning of the string and the 'end' at the end of the string. A few early terminations
        are being placed to reduce the complexity as a whole. 

        Firstly, when the start indices or end indicies is None, It will return an empty list immediately after 
        the search. Besides that if the smallest from the start index is bigger than the biggest in the end index,
        it means the start index is after the end index and we terminate as well. Lastly we also check if the start 
        and end overlaps which will return None. However, there oso lies an early termination in the for loop to 
        effectively bring down the complexity of V to N^2 instead of N^3 where N is the length of the genome. It will 
        only enter the inner for loop when the index of a certain element is smaller than the biggest index in the
        end indices list. At the end i also checks if that specific index overlaps with the next item in the end indices 
        list, thus it will break the loop and not look at the following index since the indices list is in descending order

        Input:
            start (str):  A single non-empty string consisting of only uppercase [A-D]
            end (str):  A single non-empty string consisting of only uppercase [A-D]

        Return:
            return_lst (List): A string of all the substrings of genome which have start
            as a prefix and end as a suffix

        Time complexity:
            Best = O(T + U)
            Worst = O(T + U + V)
            T is the length of the string 'start'
            U is the length of the string 'end'
            V is the number of characters in the output list 

        Time complexity analysis:
            Best = This occurs when the start_idx or end_idx is not found which means that the
            'start' or 'end' is not found in suffix trie. It also occurs when the smallest from 
            the start indices is bigger than the biggest in the end indices list. Lastly, it also
            occurs the start and the end overlaps it effectively. This will terminate it early
            and effectively make the number of characters in the output list to be 0 making the 
            time complexity O(1). The finding of the start and end indices runs in O(T) time and
            O(U) time respectively.
            Worst = This occurs when none of the early termination passes and we have to loop 
            through every start indices and for each start index it needs to loop through the
            end indices and find the substring of the genome where the start is the prefix
            and the end is the suffix. 

        Space complexity: 
            Input = O(T + U)
            Aux = O(V)
            T is the length of the string 'start'
            U is the length of the string 'end'
            V is the number of characters in the output list 

        Space complexity Analysis:
            Input = we take in two string respectively which is the start, T and end, U
            Aux = The function creates a list to store all the substrings where the start is the prefix
            and the end is the suffix of the genome.
        """
        current = self.root
        start_node = self.find_start(start, current)
        end_node = self.find_end(end, current)

        if start_node is None or end_node is None:
            return []

        # smallest from start_node bigger than biggest in end
        if start_node[len(start_node)-1] + 1 > end_node[0]:
            return []
    
        # check if the start and end overlaps
        if start_node[len(start_node)-1] == end_node[0]-len(end) + 1:
            return []
        
        return_lst = []
        count = 0
        for i in range(len(start_node)):
            start_idx = start_node[i]-len(start) + 1
            if start_node[i] < end_node[0]:
                # check if the start and end overlaps
                for j in range(len(end_node)):
                    string = ''
                    end_idx = end_node[j]+1
                    # if it overlap
                    for k in range(start_idx, end_idx):
                        count += 1
                        string += self.genome[k]
                    if start_node[i] <= end_node[j]-len(end):
                        return_lst.append(string)
                    if j != len(end_node) -1:
                        if start_node[i] == end_node[j+1]:
                            break
        print(count)
        new_count = 0
        for string in return_lst:
            for char in string:
                new_count += 1
        print(new_count)
        print(len(self.genome) ** 2)
        return return_lst

    def find_start(self, start: str, current: Node) -> list[int]:
        """
        Searches for the indices of the 'start' string in the suffix trie to find whether it exist.
        If yes, it returns the indices list containing where the 'start' string is found in the 
        genome string

        Input:
            start (str):  A single non-empty string consisting of only uppercase [A-D]

        Return:
            current.data (list[int]): An indices list consisting where the 'start' string
            is found in the genome string

        Time complexity:
            Best = Worst = O(T) as the function loop through each character in the 'start' string
            and iterates through the suffix trie
            T is the length of the string 'start'

        Space complexity: 
            Input = O(T) where T is the length of the string 'start'
            Aux = O(1) as it is all constant time operations
        """
        for char in start:
            char_index = ord(char) - ord('A')
            if current.children[char_index]:
                current = current.children[char_index]
            else:
                return None
        
        return current.data

    def find_end(self, end: str, current: Node) -> list[int]:
        """
        Searches for the indices of the 'end' string in the suffix trie to find whether it exist.
        If yes, it returns the indices list containing where the 'end' string is found in the 
        genome string

        Input:
            end (str):  A single non-empty string consisting of only uppercase [A-D]

        Return:
            current.data (list[int]): An indices list consisting where the 'end' string
            is found in the genome string

        Time complexity:
            Best = Worst = O(U) as the function loop through each character in the 'end' string
            and iterates through the suffix trie
            U is the length of the string 'end'

        Space complexity: 
            Input = O(U) where U is the length of the string 'end'
            Input = Aux = O(1) as it is all constant time operations
        """
        for char in end:
            char_index = ord(char) - ord('A')
            if current.children[char_index]:
                current = current.children[char_index]
            else:
                return None 
            
        return current.data
    

# ==========
# Q2
class Edge:
    """
    Edge class to represent an edge in the graph.
    """
    def __init__(self, u: int, v: int, w: int) -> None:
        """
        The constructor for the Edge class.
        Initialize the edge with a source vertex, a destination vertex, and a capacity.
        
        Input:
            u (int): The source vertex id.
            v (int): The destination vertex id.
            w (int): The capacity of the edge.
        
        Return:
            None
        
        Time complexity:
            Best = Worst = O(1) as there are only constant time operations.
        
        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """
        self.u = u
        self.v = v
        self.capacity = w
        self.flow = 0
        self.lower_bound = 0
        # Used for the reference of a reversed edge in the residual network to update the flow correctly.
        self.reversed_edge = None

    @property
    def residual_capacity(self) -> int:
        """
        Residual_capacity attribute in the residual network
        """
        return self.capacity - self.flow
    
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
            graph: The graph the vertex resides in.

        Return:
            None

        Time complexity:
            Best = Worst = O(1) as there are only constant time operations.

        Space complexity: 
            Input = Aux = O(1) as there are only constant space operations.
        """
        # list
        self.id = id 
        self.edges = []
        # for traversal
        self.discovered = False
        self.visited = False
        # distance
        self.distance = 0
        # backtracking/ where i was from
        self.previous_edge = None
        self.demand = 0

    def add_edge(self, edge: Edge):
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

    def backtrack_route(self, source: int, lst_of_vertices: list) -> list[Edge]:
        """
        Backtrack from the super_sink back to the super_source to update the flow

        Input:
            source (int): An integer representing the source of the graph.
            lst_of_vertices (list of Vertex): List of vertices in the graph.

        Return:
            path: List of edges representing the list of edges from the source to the sink

        Time complexity: 
            Best = O(1) when the source only traverse one time and reaches the destination. This means that
            it will only backtrack once. This will also work if the previous edge is None where it will break
            out of the loop immediately

            Worst = O(N) when the source and sink is different. The function backtracks the entire
            graph back from the sink all the way to source. 

            N is the number of vertices in the graph.

        Space complexity: 
            Input = O(N) where N is the number of vertices in the graph.
            Aux = O(P) as the function uses a list to store the route from the destionation to the source.
        """
        # initialise a routes list to store the list of edges
        route = []
        current_vertex = self
        # if it have not reached the source
        while current_vertex.id != source:
            if current_vertex.previous_edge is not None:
                # append the previous edge
                route.append(current_vertex.previous_edge)
                # go to the previous edge
                current_vertex = lst_of_vertices[current_vertex.previous_edge.u]
            else:
                break
        # reverse the list to go from source to sink instead
        route.reverse()
        return route


class NetworkFlow:
    """
    Network flow class to represent a Network flow graph.
    """
    def __init__(self, vertices_count: int) -> None:
        """
        The constructor for the Network flow graph class.
        Initialize the graph with a number of vertices count with the vertex stored in a list.

        Input:
            vertices_count (int): The number of vertices in the graph.

        Return:
            None

        Time complexity:
            Best = Worst = O(V) as the function initializes the graph with V vertices and loops through the 
            vertices count to create v number of vertex.

        Space complexity:
            Input = O(1) as there are only constant space operations.
            Aux = O(V) as the function initializes the graph with V vertices.
        """
        # store the total vertices count
        self.vertices_count = vertices_count
        self.network_flow = [None] * vertices_count
        # Create a Vertex and store it in each index tallying with the ID
        for i in range(vertices_count):
            self.network_flow[i] = Vertex(i)
    
    def add_edges(self, argv_edges):
        """
        Add edges to the graph.

        Input:
            edges (list): List of tuples representing edges in the graph.

        Return:
            None

        Time complexity:
            Best = Worst = O(E) as the function adds edges to the graph.
            E is the number of edges or the number of total edges in the input list.

        Space complexity:
            Input = O(E) where E is the number of edges in the graph.
            Aux = O(E) where E is the number of edges in the graph.
        """
        # store the edges
        self.argv_edges = argv_edges
        for edge in argv_edges:
            u = edge[0]
            v = edge[1] 
            w = edge[2]
            # add u to v
            current_edge = Edge(u,v,w)
            current_vertex = self.network_flow[u]
            current_vertex.add_edge(current_edge)

    def ford_fulkerson(self, source, sink):
        """
        Performs the Ford-Fulkerson algorithm to find the maximum flow from the source vertex to the sink vertex.
        The function initialises a residual network using the vertices count from this network. We also add the
        edges from the network flow graph to the residual network using the same edges. We then check 
        if there is the shortest path from the source to the sink if there is one using the residual network.
        If there is a path, we will add the residual capacity from the entire path using the residual path method.
        We then recheck the shortest path which updates the return path and loops until there is no more path.

        Input:
            source (Vertex): The source vertex.
            sink (Vertex): The final destination vertex.

        Return:
            None

        Time complexity:
            Best = O(V + E) when there is no path from the source to the sink. The while loop doesn't run and the flow
            is 0. The best case can also mean that each augmenting path found adds substantial amount to the flow,
            minimising the total number of iterations needed. V is due to the creation of the residual network and E is due
            to the number of edges in the graph. 
            Worst = O(FE) when the while loop runs F times and the for loop runs E times. The while loop runs F times
            as the function checks for the augmenting path each time. The worst case can also mean that each 
            augmenting path found adds a small amount to the flow, maximising the total number of iterations needed. The minimum 
            increment of flow is 1. Thus, the maximum number of iterations.
            is F where
            F is the maximum flow of the graph.
            V is the number of vertices in the graph.
            E is the number of edges of each vertex in the graph.
        Space complexity:
            Input = O(1) as there are only constant space operations.
            Aux = O(V) where V is the number of vertices in the graph. It is the initialisation of the residual network.
        """
        # Initialize flow
        flow = 0
        # Initialize the residual network
        residual_network = ResidualNetwork(self.vertices_count)
        # Add the edges to the residual network
        residual_network.add_edges(self.argv_edges)
        # Check for an augmenting path initially
        return_path = residual_network.has_AugmentingPath(source, sink)
        while return_path:
            # Augment the flow equal to the residual capacity of the path
            flow += residual_network.residual_path(return_path)
            # Check for another augmenting path
            return_path = residual_network.has_AugmentingPath(source, sink)
        
        self.residual_network = residual_network
        # Return the flow
        if flow > 0:
            return flow
        return None

class ResidualNetwork(NetworkFlow):
    """
    Residual Network class to represent a Residual Network graph.
    """
    def __init__(self, argv_vertices_count: int):
        """
        The constructor for the Residual Network graph class.
        Initialize the graph with a number of vertices count with the vertex stored in a list.

        Input:
            vertices_count (int): The number of vertices in the graph.

        Return:
            None

        Time complexity:
            Best = Worst = O(V) as the function initializes the graph with V vertices and loops through the 
            vertices count to create v number of vertex.

        Space complexity:
            Input = O(1) as there are only constant space operations.
            Aux = O(V) as the function initializes the graph with V vertices.
        """
        self.vertices_count = argv_vertices_count
        self.residual_network = [None] * argv_vertices_count
        for i in range(argv_vertices_count):
            self.residual_network[i] = Vertex(i)

    def reset(self):
        """
        Reset the graph by setting the discovered and visited status of the vertices to False.
        
        Input:
            None

        Return:
            None

        Time complexity:
            Best = Worst = O(V) as the function resets the graph by iterating over V vertices.
        
        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """
        # for all the vertices
        for vertex in self.residual_network:
            vertex.discovered = False
            vertex.visited = False
    
    def add_edges(self, argv_edges):
        """
        Add edges to the graph. Add the forward and backward edges to the graph.
        While adding the edges, we also add the reversed edge to the forward edge and the forward edge to the reversed edge.

        Input:
            edges (list): List of tuples representing edges in the graph.

        Return:
            None

        Time complexity:
            Best = Worst = O(E) as the function adds edges to the graph.
            E is the number of edges or the number of total edges in the input list.

        Space complexity:
            Input = O(E) where E is the number of edges in the graph.
            Aux = O(E) where E is the number of edges in the graph.
        """
        for edge in argv_edges:
            u = edge[0]
            v = edge[1] 
            w = edge[2]
            # add u to v
            forward_edge = Edge(u,v,w)
            backward_edge = Edge(v,u,0)
            forward_edge.reversed_edge = backward_edge
            backward_edge.reversed_edge = forward_edge
            source_vertex = self.residual_network[u].id
            end_vertex = self.residual_network[v].id
            self.residual_network[source_vertex].add_edge(forward_edge)
            self.residual_network[end_vertex].add_edge(backward_edge)


    def has_AugmentingPath(self, source, sink):
        """
        Function for BFS, starting from soruce. Find the shortest path from source to sink.
        If there is a path, return the path. If there is no path, return None.

        Input:
            source (int): The source vertex.
            sink (int): The final destination vertex.

        Return:
            path: List of edges representing the list of edges from the source to the sink

        Time complexity:
            Best = O(V) where the source only traverse one time and reaches the destination. This means that
            it will only backtrack once. This will also work if the previous edge is None where it will break
            out of the loop immediately

            Worst = O(V + E) when the source and sink is different. The function backtracks the entire
            graph back from the sink all the way to source.

            V is the number of vertices in the graph.
            E is the number of edges in the graph.
        
        Space complexity:
            Input = O(1) as there are only constant space operations.
            Aux = O(V) as the function uses a queue to store the vertices.
        """
        self.reset()
        source = self.residual_network[source]
        discovered = deque([])    # discovered is a queue
        discovered.append(source)
        while len(discovered) > 0:
            # serve from queue
            u = discovered.popleft() 
            u.visited = True      # means I have visit u
            if u.id == sink:
                return u.backtrack_route(source, self.residual_network)
            for edge in u.edges:
                v = edge.v
                v = self.residual_network[v]
                if v.discovered == False and v.visited == False and edge.residual_capacity > 0:
                    discovered.append(v)
                    v.discovered = True  #means I have discovered v, adding it to the queue
                    v.previous_edge = edge

        return None

    def residual_path(self, path) -> int:
        """
        Update the flow of the path in the residual network.
        The function takes in a path and finds the minimum residual capacity of the path.

        Input:
            path (list): List of edges representing the path from the source to the sink.

        Return:
            min_residual_capacity (int): The minimum residual capacity of the path.

        Time complexity:
            Best = Worst = O(P) where P is the number of edges in the path.

        Space complexity:
            Input = Aux = O(1) as there are only constant space operations.
        """
        min_residual_capacity = min(edge.residual_capacity for edge in path)
        for edge in path:
            edge.flow += min_residual_capacity
            edge.reversed_edge.flow -= min_residual_capacity
        return min_residual_capacity

def allocate(preferences, offices_per_org, min_shifts, max_shifts):
    """
    Function Desc:
    Allocate the officers to the companies based on the preferences and the number of shifts. The function creates a network 
    flow graph and adds the edges to the graph. The function then calculates the maximum flow from the source to the sink. 

    Approach Desc:
    Firs we have some of the preconditions which include:
    1. The function first checks if the maximum shifts is less than the minimum shifts. If it is, return None.
    2, The function then calculates the total number of officers, total needed officers each shift and the officers each 
    shift.
    3. The function then checks if the total officers multiplied by the minimum shifts is greater than the total shifts needed.
    4. The function then checks if the total needed officers each shift multiplied by 30 is greater than the officers each shift
    5. The function also checks if the total officers multiplied by the minimum shifts is greater than the total shifts needed.
    We then initialise the network flow graph and add the edges to the graph. We then calculate the maximum flow from the source
    to the sink using ford fulkerson. If the maximum flow is not equal to the total shifts needed, return None. The function then
    gets the allocation from the flow graph and returns the allocation.
    
    The total number of edges in the graph: = 1 + n + n*30 + n*30*3 + n*30*3*m + m*3*30 = O(n*m`)
    1. edges from supersource to source: 1 edge 
    2. edges from source and supersource to each officer: n edges
    3. edges from each officer to each officer day: n*30 edges
    4. edges from each officer day to each officer day shift: n*30*3 edges
    5. edges from each officer day shift to each company day shift: n*30*3*m edges
    6. edges from each company day shift to sink: m * 3 * 30

    So O(E) = O(n*m)
    The function then initialises the network flow graph and adds the edges to the graph. The function then calculates the
    maximum flow from the source to the sink using ford fulkerson. If the maximum flow is not equal to the total shifts needed,
    return None. The function then gets the allocation from the flow graph and returns the allocation.
    

    Input:
        preferences (list): A list of lists representing the preferences of the officers.
        offices_per_org (list): A list of lists representing the number of officers needed for each company.
        min_shifts (int): An integer representing the minimum number of shifts.
        max_shifts (int): An integer representing the maximum number of shifts.
    Return:
        allocation (list): A list of lists representing the allocation of the officers to the companies.
        
    Time complexity:
        Best = O(N+M)
        Worst = O(N^2 * M)
        where N is the number of officers in the graph.
        M is the number of companies in the graph.

    Time complexity analysis:
        Best = when the function checks if the maximum shifts is less than the minimum shifts and returns None.
        Worst = The function initialises the network flow graph and adds the edges to the graph. The function then calculates 
        the maximum flow from the source to the sink using ford fulkerson. This takes O`(N^2 * M) time complexity Since F < N*30,
        so total complexity is O(n*m *n * 30) so total complexity is O(N^2 * M) 

    Space complexity:
        Input = O(N + M) 
        Aux = O(N * M) 
        where N is the number of officers and M is the number of companies.

    Space complexity analysis:
        Input space: O(N + M) as the function takes in the preferences and the number of officers needed for each company.
        Auxiliary space: O(N * M) as the function creates a network flow graph with N * M edges.
    """
    total_officers = len(preferences)
    total_company = len(offices_per_org)
    days = 30
    shifts = 3
    edges = []

    # check if max shifts is less than min shifts
    if max_shifts < min_shifts:
        return None

    # calculate the total number of nodes in the graph
    total_nodes = (total_officers) + (days * total_officers) + (days * total_officers * shifts) + (days * shifts * total_company) + 4
    # create a network flow graph
    networkflow = NetworkFlow(total_nodes)
    # get the source, super source, sink and super sink    
    source = networkflow.network_flow[total_nodes-4].id
    super_source = networkflow.network_flow[total_nodes-3].id
    sink = networkflow.network_flow[total_nodes-2].id
    super_sink = networkflow.network_flow[total_nodes-1].id


    # calculate the total number of officers needed, the number of officers needed each shift and the number of officers each shift
    total_needed_officers = 0
    officers_needed_each_shift = [0,0,0]
    for company in offices_per_org:
        for i, officer in enumerate(company):
            total_needed_officers += officer
            officers_needed_each_shift[i] += officer

    officers_each_shift = [0,0,0]
    for preference in preferences:
        count = 0
        for i, shift in enumerate(preference):
            if shift == 1:
                count += 1
            officers_each_shift[i] += shift
        # if min_shifts is not 0 and count is 0, return None
        # because the officer cannot work 0 shifts
        if min_shifts != 0:
            if count == 0:
                return None
    
    # calculate the total shifts needed
    total_shifts_needed = total_needed_officers * 30 

    # check if the total needed officers each shift multiplied by 30 is greater than the officers each shift
    for i in range(3):
        if (officers_needed_each_shift[i] * 30) > (officers_each_shift[i] * max_shifts):
            return None
    
    # check if total officers multiplied by min shifts is greater than total shifts needed
    if total_officers * min_shifts > total_shifts_needed:
        return None
    
    # calculate the source demand
    source_demand = -(total_needed_officers * 30 - total_officers * min_shifts)

    # Connect super source to source or super sink to sink depending on the source demand
    if source_demand < 0:
        edges.append((super_source, source, abs(source_demand)))
    else:
        edges.append((source, super_sink, abs(source_demand)))

    # add edges from source to each officer and super source to each officer
    for i in range(total_officers):
        edges.append((source, i, max_shifts-min_shifts))
        edges.append((super_source, i, min_shifts))
    
    # add edges from each officer to each officer day
    day_start_idx = total_officers
    for i in range(total_officers):
        for j in range(days):
            edges.append((i, i * days + j + day_start_idx, 1))

    # add edges from each officer day to each officer day shift
    shift_start_idx = total_officers * days + total_officers
    for i in range(total_officers):
        for j in range(days):
            for k in range(shifts): 
                edges.append((i * days + j + day_start_idx, 
                              i * days * shifts + j * shifts + k + shift_start_idx, preferences[i][k]))

    # add edges from each officer day shift to each company day shift
    # order by company, then day, then shift
    company_start_idx = total_officers * days * shifts + total_officers * days + total_officers
    for i in range(total_officers):
        for j in range(days):
            for k in range(shifts):
                for l in range(total_company):
                    edges.append((i * days * shifts  + j * shifts + k + shift_start_idx, 
                            j * total_company + k * days * total_company + l + company_start_idx , 1))
    
    # add edges from each company day shift to sink
    for i in range(shifts):
        for j in range(days):
            for k in range(total_company):
                edges.append((i * days * total_company + j * total_company + k + company_start_idx, sink, offices_per_org[k][i]))

    # add edges from sink to super sink using the total shifts needed
    edges.append((sink, super_sink, total_shifts_needed))

    # add all the edges to the network flow graph
    networkflow.add_edges(edges)

    # calculate the maximum flow from the source to the sink
    maxflow = networkflow.ford_fulkerson(super_source, super_sink)
    # check if the maximum flow is equal to the total shifts needed
    if maxflow != total_shifts_needed and maxflow is not None:
        return None
    
    allocation = [[[[False for _ in range(shifts)] for _ in range(days)] for _ in range(total_company)] for _ in range(total_officers)]

    # get the allocation from the flow graph
    for i in range(total_officers):
        for j in range(days):
            for k in range(shifts):
                for edge in networkflow.residual_network.residual_network[i * days * shifts + j * shifts + k + shift_start_idx].edges:
                    if edge.flow == 1:
                        company_node = edge.v
                        company = (company_node - company_start_idx) % total_company
                        allocation[i][company][j][k] = True
    return allocation
    
    
genome = "AAAAAA"
start = "A"
end = "A"
orf = OrfFinder(genome)
print(orf.find(start, end))