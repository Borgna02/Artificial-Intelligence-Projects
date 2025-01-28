from typing import Literal
from environment import Environment
from agent import Agent
import time
import numpy as np

import pygame
from IPython.display import display, clear_output, Image as IPImage
from PIL import Image
from io import BytesIO

# Function to display Pygame in Jupyter Notebook
def show_pygame_in_notebook():
    """
    Captures the Pygame display surface and shows it as an image in a Jupyter Notebook cell.
    """
    surface = pygame.display.get_surface()
    if surface is None:
        raise ValueError("Pygame display surface is not initialized.")

    # Capture the Pygame display as a numpy array
    view = pygame.surfarray.array3d(surface)
    view = np.flip(view, axis=1)  # Flip horizontally to match PIL format
    view = view.swapaxes(0, 1)  # Swap axes to match PIL format

    # Convert numpy array to a PIL image
    img = Image.fromarray(view)

    # Display the image in the notebook
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    clear_output(wait=True)  # Clear the previous output
    display(IPImage(buf.read()))
    return img  # Return the image for final capture


# Function for execution in a Jupyter Notebook
def execute_in_notebook(env, heuristic, algorithm):
    """
    Executes the agent's search algorithm and returns the final frame as a PIL Image object.
    
    Args:
        env (Environment): The maze environment.
        heuristic: The heuristic function used by the agent.
        algorithm (str): The search algorithm ("best_first" or "a_star").
    
    Returns:
        tuple: A tuple (PIL.Image, str), where the image is the final frame and the string is the label.
    """
    try:
        agent = Agent(heuristic, env)

        # Display the initial maze
        env.show_maze(agent)
        show_pygame_in_notebook()

        # Perform the search
        path, _ = agent.search(algorithm)
        if path:
            for (x, y) in path:
                # Handle Pygame events to prevent the window from freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return None

                # Move the agent and update the visualization
                agent.move(x, y)
                env.show_maze(agent)
                show_pygame_in_notebook()
                time.sleep(0.2)
        else:
            print("There is no path to the goal")

        # Capture the final frame as an image
        final_image = show_pygame_in_notebook()
        return final_image, f"{heuristic.__name__}_{algorithm}"

    finally:
        # Close the Pygame virtual window
        pygame.display.quit()
        pygame.quit()


def display_images_in_notebook(images_list):
    """
    Displays a list of images in a Jupyter Notebook.
    
    Args:
        images_list (list): List of tuples (image, label).
    """
    for img, label in images_list:
        print(f"Result for {label}:")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        display(IPImage(data=buf.read(), format="png"))

# Function for execution in a normal environment
def execute_normally(env: Environment, heuristic, algorithm: Literal["best_first", "a_star"]):
    """
    Executes the agent's search algorithm and visualizes the process in a standalone Pygame window.
    
    Args:
        env (Environment): The maze environment.
        heuristic: The heuristic function used by the agent.
        algorithm (str): The search algorithm ("best_first" or "a_star").
    """
    agent = Agent(heuristic, env)

    # Display the initial maze
    env.show_maze(agent)
    time.sleep(2)

    # Perform the search
    path, _ = agent.search(algorithm)
    if path:
        for (x, y) in path:
            agent.move(x, y)
            env.show_maze(agent)
            time.sleep(0.3)

        input("Press Enter to close the game...")
    else:
        print("There is no path to the goal")
        time.sleep(2)

