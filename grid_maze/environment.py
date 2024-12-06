import random
import pygame


class Environment:
    def __init__(self, width, height, n_obstacles):
        self.width = width
        self.height = height
        self.spawn_point = (0, 0)
        self.goal = (width-1, height-1)
        
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.obstacles = []
        
        self.generate_obstacles(n_obstacles)
        

        
    def generate_obstacles(self, n=100):
        for _ in range(n):
            while True:
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.height-1)
                if (x, y) != self.spawn_point and (x, y) != self.goal:
                    break
            self.obstacles.append((x, y))
            self.grid[y][x] = 1
            
    def show_maze(self, agent):
        """
        Visualizes the grid maze and the agent's execution path using Pygame.
        
        Args:
            agent: The agent object containing its current position and visited cells.
        """
        # Initialize Pygame
        pygame.init()
        screen = pygame.display.set_mode((self.width * 20, self.height * 20))
        pygame.display.set_caption("Grid Maze")
        
        # Fill the screen with a white background
        screen.fill((255, 255, 255))

        # Draw the grid and its components
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 1:  # Obstacle
                    color = (0, 0, 0)
                elif (x, y) == self.goal:  # Goal
                    color = (0, 255, 0)
                elif (x, y) == self.spawn_point:  # Spawn point
                    color = (255, 0, 0)
                elif (x, y) in agent.visited_cells:  # Visited cells
                    color = (255, 165, 0)  # Orange
                else:  # Empty cells
                    color = (255, 255, 255)
                
                # Draw the cell and its border
                pygame.draw.rect(screen, color, pygame.Rect(x * 20, y * 20, 20, 20))
                pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(x * 20, y * 20, 20, 20), 1)
        
        # Draw the agent's current position
        pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(agent.x * 20, agent.y * 20, 20, 20))
        
        # Update the display
        pygame.display.flip()

        # Handle events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


        
    def get_distances_from_obstacles(self, x, y):
        distances = {
            'left': x,
            'up': y,
            'right': self.width - x - 1,
            'down': self.height - y - 1
        }
        
        for ox, oy in self.obstacles:
            if oy == y:
                if ox < x:
                    distances['left'] = min(distances['left'], x - ox)
                elif ox > x:
                    distances['right'] = min(distances['right'], ox - x)
            elif ox == x:
                if oy < y:
                    distances['up'] = min(distances['up'], y - oy)
                elif oy > y:
                    distances['down'] = min(distances['down'], oy - y)
        
        return distances
    
    def is_obstacle(self, x, y):
        return self.grid[y][x] == 1
    
    def is_outside(self, x, y):
        return x < 0 or y < 0 or x >= self.width or y >= self.height
    
 
    
