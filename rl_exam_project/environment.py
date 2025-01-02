from enum import Enum
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Actions(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3
    OTHER_SIDE = 4


class CellType(Enum):
    FRYING_PAN = 0
    OVEN = 1
    EGG_BEATER = 2
    GATE = 3
    EMPTY = 4


class CookingStates(Enum):
    EB_FOR_SCRAMBLED = 0  # The agent is searching the egg beater for preparing scrambled eggs
    # The agent is searching the egg beater in the other side for preparing scrambled eggs
    EB_FOR_SCRAMBLED_OS = 4

    EB_FOR_PUDDING = 1  # The agent is searching the egg beater for preparing pudding
    # The agent is searching the egg beater in the other side for preparing pudding
    EB_FOR_PUDDING_OS = 5

    PAN = 2  # The agent is searching pan (for preparing scrambled eggs)
    # The agent is searching pan in the other side (for scrambled eggs)
    PAN_OS = 6

    OVEN = 3  # The agent is searching oven (for preparing pudding)
    # The agent is searching oven in the other side (for preparing pudding)
    OVEN_OS = 7

    COOKING = 8  # The agent is cooking


class Environment:
    def __init__(self, map_width, map_height, walls, frying_pans: list[tuple], ovens: list[tuple], egg_beaters: list[tuple], gates: list[tuple], recipe: Literal["scrambled", "pudding"]):
        self.map_width = map_width
        self.map_height = map_height

        self.walls = walls
        self.frying_pans = frying_pans
        self.ovens = ovens
        self.egg_beaters = egg_beaters
        self.gates = gates
        
        self.recipe = recipe

        # Assegno la lista di azioni
        self.actions = [action for action in Actions]

        cells = [(x, y) for y in range(1, map_height+1) for x in range(1, map_width+1)
                 ]

        complete_cells = []
        for x, y in cells:
            if (x, y) in frying_pans:
                complete_cells.append((x, y, CellType.FRYING_PAN))
            elif (x, y) in ovens:
                complete_cells.append((x, y, CellType.OVEN))
            elif (x, y) in egg_beaters:
                complete_cells.append((x, y, CellType.EGG_BEATER))
            elif (x, y) in gates:
                complete_cells.append((x, y, CellType.GATE))
            else:
                complete_cells.append((x, y, CellType.EMPTY))

        # Lista di tuple (x, y, cell_type, cooking_state)
        self.states = [(x, y, cell_type, cooking_state) for x, y,
                       cell_type in complete_cells for cooking_state in CookingStates]

        # Per ogni stato ho la lista di azioni possibili e per ogni azione ho la lista di stati successivi con probabilità 0
        self.dynamics = {state: {action: {next_state: 0 for next_state in self.states}
                                 for action in self.actions} for state in self.states}

        # Assegno le dynamics
        for current_s in self.states:
            current_x, current_y, current_cell_type, current_cooking_state = current_s

            # Se la cella corrente è vuota, l'agente non può cambiare cooking state ma può solo muoversi o iniziare a cercare il gate
            for action in self.actions:
                # Other side
                if action == Actions.OTHER_SIDE:
                    next_state = (current_x, current_y, current_cell_type,
                                  self.other_side_counterpart(current_cooking_state))
                    self.dynamics[current_s][action][next_state] = 1
                # Move
                else:
                    if action == Actions.UP:
                        next_x = current_x
                        next_y = current_y+1
                    elif action == Actions.LEFT:
                        next_x = current_x-1
                        next_y = current_y
                    elif action == Actions.DOWN:
                        next_x = current_x
                        next_y = current_y-1
                    elif action == Actions.RIGHT:
                        next_x = current_x+1
                        next_y = current_y

                    next_cell_type = self.get_cell_type(next_x, next_y)
                    next_state = (
                        next_x, next_y, next_cell_type, self.get_next_cooking_state(next_cell_type, current_cooking_state))
                    # Assegno 1 solo allo stato raggiungibile da current_s con l'azione action e solo se non c'è un muro
                    if ((current_x, current_y), (next_x, next_y)) not in self.walls:
                        self.dynamics[current_s][action][next_state] = 1
                    else:
                        self.dynamics[current_s][action][current_s] = 1


        # Aggiungi ricompense uguali a 10 se l'agente raggiunge la cella cercata in quel cooking state
        self.rewards = {state: {action: {next_state: 0 for next_state in self.states}
                                 for action in self.actions} for state in self.states}
        
        for current_s in self.states:
            _, _, _, current_cooking_state = current_s
            for action in self.actions:
                for next_s in self.states:
                    _, _, next_cell_type, _ = next_s
                    if current_cooking_state == CookingStates.EB_FOR_PUDDING or current_cooking_state == CookingStates.EB_FOR_SCRAMBLED and next_cell_type == CellType.EGG_BEATER:
                        self.rewards[current_s][action][next_s] = 10
                    elif current_cooking_state == CookingStates.PAN and next_cell_type == CellType.FRYING_PAN:
                        self.rewards[current_s][action][next_s] = 10
                    elif current_cooking_state == CookingStates.OVEN and next_cell_type == CellType.OVEN:
                        self.rewards[current_s][action][next_s] = 10
                    elif current_cooking_state == CookingStates.EB_FOR_SCRAMBLED_OS or current_cooking_state == CookingStates.EB_FOR_PUDDING_OS or current_cooking_state == CookingStates.PAN_OS or current_cooking_state == CookingStates.OVEN_OS and next_cell_type == CellType.GATE:
                        self.rewards[current_s][action][next_s] = 10
                        


    def get_episode(self):
        episode_length = np.random.randint(4, 8)

        # Ottieni lo stato iniziale
        if self.recipe == "scrambled":
            valid_states = [tuple(state) for state in self.states if state[3] == CookingStates.EB_FOR_SCRAMBLED]
        else:
            valid_states = [tuple(state) for state in self.states if state[3] == CookingStates.EB_FOR_PUDDING]

        # Scegli uno stato casuale
        if len(valid_states) == 0:
            raise ValueError(f"No valid states found for recipe '{self.recipe}'")
        current_state = valid_states[np.random.choice(len(valid_states))]

        # Initialize episode
        episode = []

        for _ in range(episode_length):
            # Get action randomly
            action = np.random.choice(self.actions)

            # Get next state according to dynamics
            next_states = list(self.dynamics[current_state][action].keys())
            probabilities = list(self.dynamics[current_state][action].values())

            if sum(probabilities) == 0:
                raise ValueError(f"No transitions defined for state {current_state} and action {action}")

            next_state = next_states[np.random.choice(len(next_states), p=probabilities)]

            # Get reward
            reward = self.rewards[current_state][action][next_state]

            # Append to episode
            episode.append((current_state, action, reward))

            # Update current state
            current_state = next_state

        return episode
        # Funzione per disegnare la griglia e la mappa
    def draw_map(self):
        fig, ax = plt.subplots(figsize=(self.map_width, self.map_height / 2))

        # Disegna la griglia
        for x in range(self.map_width + 2):
            ax.plot([x, x], [1, self.map_height + 1],
                    color="black", linewidth=0.5)
        for y in range(self.map_height + 2):
            ax.plot([0, self.map_width + 1], [y, y],
                    color="black", linewidth=0.5)

        # Disegna i muri
        for (x1, y1), (x2, y2) in self.walls:
            # Muro verticale
            if y1 == y2:
                ax.plot([x1+1, x1+1], [y1, y1+1], color="black", linewidth=3)
            # Muro orizzontale
            elif x1 == x2:
                ax.plot([x1, x1+1], [y1+1, y1+1], color="black", linewidth=3)

        # Disegna le padelle
        for x, y in self.frying_pans:
            rect = patches.Rectangle(
                (x, y), 1, 1, linewidth=1, edgecolor="red", facecolor="red", label="Frying Pan")
            ax.add_patch(rect)

        # Disegna i forni
        for x, y in self.ovens:
            rect = patches.Rectangle(
                (x, y), 1, 1, linewidth=1, edgecolor="green", facecolor="green", label="Oven")
            ax.add_patch(rect)

        # Disegna le fruste
        for x, y in self.egg_beaters:
            rect = patches.Rectangle(
                (x, y), 1, 1, linewidth=1, edgecolor="blue", facecolor="blue", label="Egg Beater")
            ax.add_patch(rect)

        # Disegna i cancelli
        for x, y in self.gates:
            rect = patches.Rectangle(
                (x, y), 1, 1, linewidth=1, edgecolor="yellow", facecolor="yellow", label="Gate")
            ax.add_patch(rect)

        # Scrivi le coordinate dentro ogni cella
        # for x in range(1, self.map_width + 1):
        #     for y in range(1, self.map_height + 1):
        #         ax.text(
        #             x + 0.5, y + 0.5, f'({x},{y})', ha='center', va='center', fontsize=8, color='black')

        # Configurazione del grafico
        ax.set_xlim(1, self.map_width + 1)
        ax.set_ylim(1, self.map_height + 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Aggiungi la legenda a destra della mappa
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),
                   loc='center left', bbox_to_anchor=(0.72, 0.5))

        plt.show()

    def other_side_counterpart(self, cooking_state):
        if cooking_state == CookingStates.EB_FOR_SCRAMBLED:
            return CookingStates.EB_FOR_SCRAMBLED_OS
        elif cooking_state == CookingStates.EB_FOR_SCRAMBLED_OS:
            return CookingStates.EB_FOR_SCRAMBLED
        elif cooking_state == CookingStates.EB_FOR_PUDDING:
            return CookingStates.EB_FOR_PUDDING_OS
        elif cooking_state == CookingStates.EB_FOR_PUDDING_OS:
            return CookingStates.EB_FOR_PUDDING
        elif cooking_state == CookingStates.PAN:
            return CookingStates.PAN_OS
        elif cooking_state == CookingStates.PAN_OS:
            return CookingStates.PAN
        elif cooking_state == CookingStates.OVEN:
            return CookingStates.OVEN_OS
        elif cooking_state == CookingStates.OVEN_OS:
            return CookingStates.OVEN
        else:
            return None

    def get_cell_type(self, x, y):
        if (x, y) in self.frying_pans:
            return CellType.FRYING_PAN
        elif (x, y) in self.ovens:
            return CellType.OVEN
        elif (x, y) in self.egg_beaters:
            return CellType.EGG_BEATER
        elif (x, y) in self.gates:
            return CellType.GATE
        else:
            return CellType.EMPTY

    def get_next_cooking_state(self, next_cell_type, current_cooking_state):
        # Se sto cercando qualcosa dall'altro lato e raggiungo il gate, passo alla controparte dell'altro lato
        if current_cooking_state == CookingStates.EB_FOR_PUDDING_OS or current_cooking_state == CookingStates.EB_FOR_SCRAMBLED_OS or current_cooking_state == CookingStates.PAN_OS or current_cooking_state == CookingStates.OVEN_OS and next_cell_type == CellType.GATE:
            return self.other_side_counterpart(current_cooking_state)

        # Se sto cercando l'egg beater per pudding e raggiungo l'egg beater, passo alla ricerca del forno
        elif current_cooking_state == CookingStates.EB_FOR_PUDDING and next_cell_type == CellType.EGG_BEATER:
            return CookingStates.OVEN

        # Se sto cercando l'egg beater per scrambled e raggiungo l'egg beater, passo alla ricerca della padella
        elif current_cooking_state == CookingStates.EB_FOR_SCRAMBLED and next_cell_type == CellType.EGG_BEATER:
            return CookingStates.PAN

        # Se sto cercando la padella e raggiungo la padella, passo alla fase di cottura
        elif current_cooking_state == CookingStates.PAN and next_cell_type == CellType.FRYING_PAN:
            return CookingStates.COOKING

        # Se sto cercando il forno e raggiungo il forno, passo alla fase di cottura
        elif current_cooking_state == CookingStates.OVEN and next_cell_type == CellType.OVEN:
            return CookingStates.COOKING

        else:
            return current_cooking_state
