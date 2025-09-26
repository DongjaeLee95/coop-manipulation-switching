import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib

# ==== GRID SETUP ====
WIDTH, HEIGHT = 7, 7

# Obstacles: 3x3 block in the center
OBSTACLES = {(x, y) for x in range(2, 5) for y in range(2, 5)}

# Agents: (start, goal)
AGENTS = [
    {'start': (0, 0), 'goal': (6, 6)},  # Agent 0
    {'start': (0, 6), 'goal': (6, 0)},  # Agent 1
    {'start': (3, 0), 'goal': (3, 6)},  # Agent 2
    {'start': (4, 1), 'goal': (2, 6)},  # Agent 3 (your added agent)
]

# Directions (4-way + wait)
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]


# ==== A* AND PRIORITIZED PLANNING ====

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def in_bounds(pos):
    x, y = pos
    return 0 <= x < WIDTH and 0 <= y < HEIGHT

def is_free(pos, t, reserved):
    return pos not in OBSTACLES and reserved.get((pos, t), -1) == -1

def a_star(start, goal, reserved, start_time=0):
    """
    A* search with reservations and optional start delay.
    Agent will wait in start position for start_time steps before moving.
    """
    open_list = []
    # Initial node: position=start, time=start_time
    # Path starts with waiting at start
    heapq.heappush(open_list, (heuristic(start, goal) + start_time, start_time, start, [start] * start_time))
    visited = set()

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        if (current, g) in visited:
            continue
        visited.add((current, g))

        if current == goal and g >= start_time:
            return path

        for dx, dy in DIRS:
            next_pos = (current[0] + dx, current[1] + dy)
            next_time = g + 1
            if not in_bounds(next_pos):
                continue

            if not is_free(next_pos, next_time, reserved):
                continue

            # Prevent edge collisions (swapping positions)
            other_agent = reserved.get((next_pos, g), -1)
            if other_agent != -1 and reserved.get((current, next_time), -1) == other_agent:
                # This move swaps positions with another agent; forbidden
                continue

            heapq.heappush(open_list, (
                next_time + heuristic(next_pos, goal),
                next_time,
                next_pos,
                path + [next_pos]
            ))

    return None

def reserve_path(path, agent_id, reserved):
    for t, pos in enumerate(path):
        reserved[(pos, t)] = agent_id
    # Hold the goal indefinitely to avoid others entering
    for t in range(len(path), len(path) + 50):
        reserved[(path[-1], t)] = agent_id

def prioritized_planning(max_delay=10):
    paths = []
    reserved = {}

    for i, agent in enumerate(AGENTS):
        print(f"Planning Agent {i}")
        # Try no delay first, then increase delay if no path found
        path = None
        for delay in range(max_delay + 1):
            path = a_star(agent['start'], agent['goal'], reserved, start_time=delay)
            if path is not None:
                if delay > 0:
                    print(f"Agent {i} path found with delay {delay}")
                break
        if path is None:
            print(f"Agent {i} failed to find a path even with delay up to {max_delay}.")
            return paths
        reserve_path(path, i, reserved)
        paths.append(path)

    return paths

# ==== VISUALIZATION ====

def animate_paths(paths):
    max_time = max(len(p) for p in paths)

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, WIDTH - 0.5)
    ax.set_ylim(-0.5, HEIGHT - 0.5)
    ax.set_xticks(range(WIDTH))
    ax.set_yticks(range(HEIGHT))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    # Draw obstacles
    for (x, y) in OBSTACLES:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black'))

    # Generate distinct colors dynamically based on number of agents
    cmap = matplotlib.colormaps['tab10'].resampled(len(AGENTS))

    agents_patches = []
    for i, path in enumerate(paths):
        patch = plt.Circle(path[0], 0.3, color=cmap(i))
        ax.add_patch(patch)
        agents_patches.append(patch)

    def update(t):
        for i, patch in enumerate(agents_patches):
            pos = paths[i][t] if t < len(paths[i]) else paths[i][-1]
            patch.center = pos
        return agents_patches

    ani = animation.FuncAnimation(fig, update, frames=max_time + 10, interval=800, blit=True, repeat=False)
    plt.title("Prioritized Planning: Multi-Agent A* with Delay Retry & Edge Collision Prevention")
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    paths = prioritized_planning(max_delay=20)
    print(f"Planned paths: {len(paths)} / {len(AGENTS)}")
    if len(paths) == len(AGENTS):
        animate_paths(paths)
    else:
        print("Failed to plan all agents. Increase delay or check obstacles.")
