from matplotlib import pyplot as plt
from player import MAX_NUM_OF_PIECES

def plot_wins(players_list):
    attributes = ['wins', 'win_rate', 'draws', 'completed_runs']
    bar_width = 0.35

    fig, axs = plt.subplots(1, len(attributes), figsize=(20, 5))

    for i, attr in enumerate(attributes):
        for j, player in enumerate(players_list):
            # Assuming player[0] is the player object and player[1] is the label
            player_value = getattr(player[0], attr)

            # Create bars for the player
            axs[i].bar(
                j,  # Position of the bars
                player_value,  # Value to plot
                bar_width,  # Width of each bar
                label=player[1] if i == 0 else ""  # Label for the player (only for the first subplot)
            )

        axs[i].set_title(attr)
        axs[i].set_xticks(range(len(players_list)))
        axs[i].set_xticklabels([player[1] for player in players_list])
        axs[i].set_ylabel('Values')

    axs[0].legend()
    plt.tight_layout()
    plt.show()
    
def plot_times(players_list):
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))

    # Plot times vs number of moves
    for player, name in players_list:
        x = sorted(player.move_times_per_nmoves.keys())
        y = [
            player.move_times_per_nmoves[i] / player.completed_moves_per_nmoves[i]
            if player.completed_moves_per_nmoves[i] != 0 else 0
            for i in x
        ]
        axs[0].plot(x, y, label=name)

    axs[0].set_xlabel('Number of Moves')
    axs[0].set_ylabel('Move Time per Completed Move')
    axs[0].set_title('Move Time per Completed Move vs Number of Moves')
    axs[0].legend()

    # Plot times vs number of pieces
    x = range(1, MAX_NUM_OF_PIECES + 1)
    for player, name in players_list:
        y = [
            player.move_times_per_npieces[i] / player.completed_moves_per_npieces[i]
            if player.completed_moves_per_npieces[i] != 0 else 0
            for i in range(1, MAX_NUM_OF_PIECES + 1)
        ]
        axs[1].plot(x, y, label=name)

    axs[1].set_xlabel('Number of Pieces')
    axs[1].set_ylabel('Move Time per Completed Move')
    axs[1].set_title('Move Time per Completed Move vs Number of Pieces')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def plot_cache_rates(players_list):
    # Liste per i nomi dei giocatori e i rispettivi cache hit rate
    player_names = []
    cache_hit_rates = []

    # Calcola il cache hit rate per ogni giocatore
    for player, name in players_list:
        cache_hits = player.engine.cache_hits
        hits = player.engine.hits
        # Evita divisioni per zero
        if hits > 0:
            hit_rate = (cache_hits / hits) * 100  # Calcola la percentuale
        else:
            hit_rate = 0
        player_names.append(name)
        cache_hit_rates.append(hit_rate)

    # Crea il grafico a barre
    plt.figure(figsize=(10, 6))
    plt.bar(player_names, cache_hit_rates, color='skyblue')
    plt.xlabel('Giocatori')
    plt.ylabel('Cache Hit Rate (%)')
    plt.title('Cache Hit Rate per Giocatore')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)  # Supponendo che il tasso sia una percentuale tra 0 e 100
    plt.tight_layout()
    plt.show()