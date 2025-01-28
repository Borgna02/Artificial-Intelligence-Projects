from environment import Environment


def h_manhattan_obstacle_weighted(x, y, env: Environment):
    if env.is_obstacle(x, y):
        return float('inf')

    W = 0.1

    # Get the directions to the goal
    goal_directions = ['right' if env.goal[0] > x else 'left' if env.goal[0] < x else '',
                       'down' if env.goal[1] > y else 'up' if env.goal[1] < y else '']
    goal_directions = [d for d in goal_directions if d]

    # Get the distances to the obstacles
    obstacle_distances = env.get_distances_from_obstacles(x, y)

    # Compute the heuristic value
    return abs(x - env.goal[0]) + abs(y - env.goal[1]) + sum(W * 1 / obstacle_distances[d] for d in goal_directions)


def h_manhattan(x, y, env: Environment):
    return abs(x - env.goal[0]) + abs(y - env.goal[1])

def h_euclidean(x, y, env: Environment):
    return ((x - env.goal[0]) ** 2 + (y - env.goal[1]) ** 2) ** 0.5