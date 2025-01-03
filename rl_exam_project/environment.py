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
    def __init__(self, map_width, map_height, walls, frying_pans: list[tuple], ovens: list[tuple], egg_beaters: list[tuple], gates: list[tuple], recipe: Literal["scrambled", "pudding"], no_spawn_points=[(5,y) for y in range(1, 5)]):
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

        cells = [(x, y) for y in range(1, map_height+1)
                 for x in range(1, map_width+1)]

        self.no_spawn_points = no_spawn_points

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

                    # Se l'agente vuole cambiare lato e raggiunge il gate, si teletrasporta dall'altro lato
                    if (next_x, next_y) in self.gates and current_cooking_state in [CookingStates.EB_FOR_PUDDING_OS, CookingStates.EB_FOR_SCRAMBLED_OS, CookingStates.PAN_OS, CookingStates.OVEN_OS]:
                        next_x, next_y = self.get_other_gate(next_x, next_y)

                    next_cell_type = self.get_cell_type(next_x, next_y)
                    next_state = (
                        next_x, next_y, next_cell_type, self.get_next_cooking_state(next_cell_type, current_cooking_state))
                    # Assegno 1 solo allo stato raggiungibile da current_s con l'azione action e solo se non c'è un muro
                    if not self.is_a_wall(current_x, current_y, next_x, next_y) and current_cooking_state != CookingStates.COOKING:
                        self.dynamics[current_s][action][next_state] = 1
                    else:
                        self.dynamics[current_s][action][current_s] = 1


        # Rewards
        self.rewards = {state: {action: {next_state: -1 for next_state in self.states}  # Penalità per passo
                                for action in self.actions} for state in self.states}

        for current_s in self.states:
            current_x, current_y, _, current_cooking_state = current_s
            for action in self.actions:
                for next_s in self.states:
                    next_x, next_y, next_cell_type, _ = next_s

                    # Premiazione per il raggiungimento dell'egg beater
                    if current_cooking_state == CookingStates.EB_FOR_PUDDING or current_cooking_state == CookingStates.EB_FOR_SCRAMBLED:
                        if next_cell_type == CellType.EGG_BEATER:
                            self.rewards[current_s][action][next_s] = 40
                        # elif (next_x, next_y) in [(x+dx, y+dy) for x, y in self.egg_beaters for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]:
                            
                        #     adjacent_egg_beaters = [(x,y) for x, y in self.egg_beaters if (x, y) in [(next_x + dx, next_y + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]]
                        #     goal_x, goal_y = adjacent_egg_beaters[0]
                           
                        #     # Se l'agente raggiunge una cella adiacente all'egg beater, assegna una ricompensa di 20
                        #     self.rewards[current_s][action][next_s] = 15 if not self.is_a_wall(goal_x, goal_y, next_x, next_y) else -1

                    # Premiazione per il raggiungimento della padella
                    elif current_cooking_state == CookingStates.PAN:
                        if next_cell_type == CellType.FRYING_PAN:
                            self.rewards[current_s][action][next_s] = 100
                        # elif (next_x, next_y) in [(x+dx, y+dy) for x, y in self.frying_pans for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]:
                            
                        #     adjacent_frying_pans = [(x,y) for x, y in self.frying_pans if (x, y) in [(next_x + dx, next_y + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]]
                        #     goal_x, goal_y = adjacent_frying_pans[0]
                            
                        #     self.rewards[current_s][action][next_s] = 40 if not self.is_a_wall(goal_x, goal_y, next_x, next_y) else -1

                    # Premiazione per il raggiungimento del forno
                    elif current_cooking_state == CookingStates.OVEN:
                        if next_cell_type == CellType.OVEN:
                            self.rewards[current_s][action][next_s] = 100
                        # elif (next_x, next_y) in [(x+dx, y+dy) for x, y in self.ovens for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]:
                            
                        #     adjacent_ovens = [(x,y) for x, y in self.ovens if (x, y) in [(next_x + dx, next_y + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]]
                        #     goal_x, goal_y = adjacent_ovens[0]
                            
                            
                        #     self.rewards[current_s][action][next_s] = 40 if not self.is_a_wall(goal_x, goal_y, next_x, next_y) else -1

                    # Premiazione per il passaggio di gate
                    # elif current_cooking_state == CookingStates.EB_FOR_SCRAMBLED_OS or current_cooking_state == CookingStates.EB_FOR_PUDDING_OS or current_cooking_state == CookingStates.PAN_OS or current_cooking_state == CookingStates.OVEN_OS:
                    #     if next_cell_type == CellType.GATE:
                    #         self.rewards[current_s][action][next_s] = 5
                    #     elif (next_x, next_y) in [(x+dx, y+dy) for x, y in self.gates for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]:
                            
                    #         adjacent_gates = [(x,y) for x, y in self.gates if (x, y) in [(next_x + dx, next_y + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]]
                    #         goal_x, goal_y = adjacent_gates[0]
                            
                    #         self.rewards[current_s][action][next_s] = 1 if not self.is_a_wall(goal_x, goal_y, next_x, next_y) else -1
                    
                    # Penalità se la transizione attraversa un muro
                    if self.is_a_wall(current_x, current_y, next_x, next_y):
                        # Penalità per muro
                        self.rewards[current_s][action][next_s] = -50
                        

    def get_episode(self):
        episode_length = np.random.randint(400, 500)

        # Ottieni lo stato iniziale random
        if self.recipe == "scrambled":
            valid_states = [state for state in self.states if state[3] in [CookingStates.EB_FOR_SCRAMBLED, CookingStates.EB_FOR_SCRAMBLED_OS, CookingStates.PAN] and (state[0], state[1]) not in self.no_spawn_points]
        else:
            valid_states = [state for state in self.states if state[3] in [CookingStates.EB_FOR_PUDDING, CookingStates.EB_FOR_PUDDING_OS, CookingStates.OVEN] and (state[0], state[1]) not in self.no_spawn_points]


        # Scegli uno stato casuale
        if len(valid_states) == 0:
            raise ValueError(
                f"No valid states found for recipe '{self.recipe}'")

        current_state = valid_states[np.random.choice(len(valid_states))]
        print("\nSpawn state: ", current_state)

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

            next_state = next_states[np.random.choice(
                len(next_states), p=probabilities)]

            # Get reward
            reward = self.rewards[current_state][action][next_state]
            print("CS: ", current_state, "A: ", action, "NS: ", next_state, "R: ", reward)

            # Append to episode
            episode.append((current_state, action, reward))

            # Update current state
            current_state = next_state

            if current_state[3] == CookingStates.COOKING:
                print("Cooking state reached\n")
                break

        return episode
        # Funzione per disegnare la griglia e la mappa

    def draw_map(self, policy=None, cooking_state_for_rw=None):
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

        # Disegna le padelle con immagine
        pan_image = plt.imread('pan.png')
        for x, y in self.frying_pans:
            ax.imshow(pan_image, extent=(x, x+1, y, y+1))

        # Disegna i forni con immagine
        oven_image = plt.imread('oven.png')
        for x, y in self.ovens:
            ax.imshow(oven_image, extent=(x, x+1, y, y+1))

        # Disegna le fruste
        egg_beater_image = plt.imread('beater.png')
        for x, y in self.egg_beaters:
            ax.imshow(egg_beater_image, extent=(x, x+1, y, y+1))

        # Disegna i cancelli
        for x, y in self.gates:
            rect = patches.Rectangle(
                (x, y), 1, 1, linewidth=1, edgecolor="yellow", facecolor="yellow", label="Gate")
            ax.add_patch(rect)

        # Disegna la policy
        if policy is not None:
            arrow_params = {
                "width": 0.01,
                "head_width": 0.1,
                "head_length": 0.1,
                "length_includes_head": True
            }
            action_directions = {
                Actions.UP: (0, 0.4),
                Actions.DOWN: (0, -0.4),
                Actions.LEFT: (-0.4, 0),
                Actions.RIGHT: (0.4, 0),
                Actions.OTHER_SIDE: (0.4, 0.4)
            }
            cooking_state_colors = {
                CookingStates.OVEN: "fuchsia",
                CookingStates.EB_FOR_PUDDING: "orange",
                CookingStates.OVEN_OS: "green",
                CookingStates.EB_FOR_PUDDING_OS: "purple",
                # CookingStates.PAN: "pink",
                # CookingStates.EB_FOR_SCRAMBLED: "yellow",
                # CookingStates.PAN_OS: "blue",
                # CookingStates.EB_FOR_SCRAMBLED_OS: "red",
                CookingStates.COOKING: "black"
            }

            for (x, y, _, cooking_state), actions in policy.items():
                # Default a 'black' se lo stato non è mappato
                color = cooking_state_colors.get(cooking_state, "black")
                for action, active in actions.items():
                    if active == 1:
                        dx, dy = action_directions[action]
                        arrow = patches.FancyArrow(
                            x + 0.5, y + 0.5, dx - 0.04*dx*cooking_state.value, dy - 0.04*dy*cooking_state.value, color=color, **arrow_params
                        )
                        ax.add_patch(arrow)

            # Aggiungi la legenda dei colori delle frecce
            for cooking_state, color in cooking_state_colors.items():
                ax.plot([], [], color=color, label=cooking_state.name)
                
        if cooking_state_for_rw is not None:
            # Disegna le ricompense
            for x, y, _, current_cooking_state in self.states:
                if current_cooking_state == cooking_state_for_rw:
                    reward = max(self.rewards[current_state][action][(x, y, _, current_cooking_state)]for action in self.actions for current_state in self.states if current_state[3] == cooking_state_for_rw)
                    ax.text(x + 0.9, y + 0.1, f"{reward}", ha="right", va="bottom", fontsize=8)
                    
            # Imposta il titolo del plt come il nome dell'enum current_state_for_rw
            ax.set_title(f"Rewards per Cooking State: {cooking_state_for_rw.name}")

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
        elif cooking_state == CookingStates.COOKING:
            return CookingStates.COOKING
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
        if next_cell_type == CellType.GATE and (current_cooking_state == CookingStates.EB_FOR_PUDDING_OS or current_cooking_state == CookingStates.EB_FOR_SCRAMBLED_OS or current_cooking_state == CookingStates.PAN_OS or current_cooking_state == CookingStates.OVEN_OS):
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

    def get_other_gate(self, x, y):
        for gate_x, gate_y in self.gates:
            if gate_x != x and gate_y != y:
                return gate_x, gate_y

        return None

    def is_a_wall(self, x1, y1, x2, y2):
        return ((x1, y1), (x2, y2)) in self.walls or ((x2, y2), (x1, y1)) in self.walls