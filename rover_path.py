import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq

terrain = np.load("ohrc_terrain_array.npy").astype(float)

# Normalize brightness -> cost (bright = low cost/safe, dark = high cost/risky)
norm = (terrain - terrain.min()) / (terrain.max() - terrain.min())
cost_map = 1.0 - norm  # invert: dark (shadow) = high cost
cost_map = cost_map * 9 + 1  # scale to cost range 1-10

# Downsample for faster pathfinding (full res is overkill for A*/D*)
step = 10
cost_small = cost_map[::step, ::step]
print("Cost map shape for pathfinding:", cost_small.shape)

# Pick start (top-left, illuminated area) and goal (bottom-right, illuminated area)
start = (5, 5)
goal = (cost_small.shape[0] - 10, cost_small.shape[1] - 10)
print("Start:", start, "Goal:", goal)

def astar(cost, start, goal):
    rows, cols = cost.shape
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    def h(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = current[0]+dr, current[1]+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                move_cost = cost[nr, nc]
                tentative_g = g_score[current] + move_cost
                neighbor = (nr, nc)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + h(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current
    return None

print("Running pathfinding...")
path = astar(cost_small, start, goal)

if path:
    print(f"Path found, {len(path)} steps")
    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]

    plt.figure(figsize=(12, 6))
    plt.imshow(cost_small, cmap='inferno')
    plt.colorbar(label='Traversal cost')
    plt.plot(path_cols, path_rows, color='cyan', linewidth=2, label='Rover path')
    plt.scatter([start[1]], [start[0]], color='lime', s=80, label='Start (landing zone)')
    plt.scatter([goal[1]], [goal[0]], color='red', s=80, label='Goal (target)')
    plt.legend()
    plt.title("Rover path over OHRC-derived terrain cost map")
    plt.savefig("rover_path.png", dpi=150)
    print("Saved rover_path.png")
else:
    print("No path found - check start/goal are not blocked")