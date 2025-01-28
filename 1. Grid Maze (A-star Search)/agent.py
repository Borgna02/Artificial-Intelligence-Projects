from queue import PriorityQueue
from typing import Literal
from environment import Environment


class Agent:

    def __init__(self, heuristic, env: Environment):
        self.heuristic = heuristic
        self.env = env
        self.x = env.spawn_point[0]
        self.y = env.spawn_point[1]
        self.visited_cells = set()

    def search(self, algorithm: Literal["best_first", "a_star"]):
        """
        Implements Best First Search and A* Search based on the selected algorithm.

        Args:
            algorithm (Literal["best_first", "a_star"]): Determines whether to use Best First or A*.

        Returns:
            tuple: A reconstructed path to the goal (if found) and the number of iterations performed.
        """
        
        # Priority queue for the frontier
        open_set = PriorityQueue()
        
        # Add the starting point to the frontier
        start = (self.x, self.y)
        open_set.put((0, start))
        
        # Dictionary to track the nodes preceding each node
        came_from = {}
        
        # Set of visited nodes
        visited = set()
        
        # Dictionary to track g-scores
        g_score_dict = {start: 0}
            
        n_iterations = 0    
        
        while not open_set.empty():
            n_iterations += 1
            # Get the node with the lowest priority (the best f-score)            
            _, current = open_set.get()
            x,y = current
            
            # If current is the goal, the path is reconstructed and returned
            if current == self.env.goal:
                return self.reconstruct_path(came_from, current), n_iterations
            
            # Skip current if it has been already visited
            if current in visited:
                continue
            
            # Mark the node as visited
            visited.add(current)
            
            # Explore neighbors in the four cardinal directions
            for dx, dy in [(-1,0), (0,1), (1,0), (0,-1)]:
                neighbor = (x + dx, y + dy)
                nx, ny = neighbor
                
                # Skip the neighbor if it is an edge of the grid or an obstacle
                if self.env.is_outside(nx, ny) or self.env.is_obstacle(nx, ny):
                    continue
                
                # Compute f-score based on the selected algorithm
                g_score = 0 if algorithm == "best_first" else g_score_dict.get(current, float('inf')) + 1
                h_score = self.heuristic(nx, ny, self.env)
                f_score = h_score + g_score
                
                
                # Update the neighbor's g-score if it's a better path or the node hasn't been visited
                if neighbor not in g_score_dict or g_score < g_score_dict[neighbor]:
                    g_score_dict[neighbor] = g_score
                    open_set.put((f_score, neighbor))
                    came_from[neighbor] = current
              
                    
        # If no path is found, return None  
        return None, None
    
    def move(self, x, y):
        self.visited_cells.add((self.x, self.y))
        self.x = x
        self.y = y
    
    def reconstruct_path(self, came_from, current):
        path = [current]  
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path