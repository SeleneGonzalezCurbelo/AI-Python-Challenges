import os
import random
import time
import heapq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
rng = random.Random(SEED)
np.random.seed(SEED)

FILAS = 20
COLUMNAS = 20

# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------
# 0-based coordinates as (row, col). The map is a list of lists where None
# marks an obstacle and an integer in [1, 5] is the entry cost of a cell.
mapa = [[None for _ in range(COLUMNAS)] for _ in range(FILAS)]

inicio = (0, 0)
destino = (FILAS - 1, COLUMNAS - 1)

# Guarantee at least one traversable path (monotonic: only down/right moves).
path_cells = set()
r, c = inicio
path_cells.add((r, c))
while (r, c) != destino:
    if r < destino[0] and (c == destino[1] or rng.random() < 0.5):
        r += 1
    else:
        c += 1
    path_cells.add((r, c))

for i in range(FILAS):
    for j in range(COLUMNAS):
        cell = (i, j)
        if cell in (inicio, destino):
            mapa[i][j] = rng.randint(1, 5)
        elif cell in path_cells:
            mapa[i][j] = rng.randint(1, 5)
        else:
            if rng.random() < 0.22:
                mapa[i][j] = None
            else:
                mapa[i][j] = rng.randint(1, 5)

# Aliases for compatibility with different naming conventions.
grid = mapa
start = inicio
end = destino
INICIO = inicio
DESTINO = destino
MAPA = mapa
ROWS = FILAS
COLS = COLUMNAS


def _mapa(mapa_arg=None):
    return mapa if mapa_arg is None else mapa_arg


def es_transitable(celda, mapa_arg=None):
    m = _mapa(mapa_arg)
    r, c = celda
    if r < 0 or r >= len(m) or c < 0 or c >= len(m[0]):
        return False
    return m[r][c] is not None


def vecinos(celda, mapa_arg=None):
    """Return the traversable 4-neighbour cells of ``celda``."""
    m = _mapa(mapa_arg)
    r, c = celda
    result = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(m) and 0 <= nc < len(m[0]):
            if m[nr][nc] is not None:
                result.append((nr, nc))
    return result


def coste_celda(celda, mapa_arg=None):
    m = _mapa(mapa_arg)
    r, c = celda
    return m[r][c]


def coste_ruta(ruta, mapa_arg=None):
    """Total entry cost of a route (cost of every cell except the start)."""
    if not ruta or len(ruta) < 2:
        return 0
    total = 0
    for cell in ruta[1:]:
        total += coste_celda(cell, mapa_arg)
    return total


def manhattan(a, b=None):
    if b is None:
        b = destino
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristica(nodo, destino_arg=None):
    """Admissible and consistent Manhattan heuristic."""
    if destino_arg is None:
        destino_arg = destino
    return manhattan(nodo, destino_arg)


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------
def dijkstra(inicio_arg, destino_arg, mapa_arg=None):
    m = _mapa(mapa_arg)
    # local references
    def nbrs(x):
        return vecinos(x, m)

    INF = float("inf")
    dist = {inicio_arg: 0}
    prev = {inicio_arg: None}
    pq = [(0, inicio_arg)]
    explored = 0
    closed = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in closed:
            continue
        if d > dist.get(u, INF):
            continue
        closed.add(u)
        explored += 1
        if u == destino_arg:
            break
        for v in nbrs(u):
            nd = d + coste_celda(v, m)
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    # Reconstruct path.
    if destino_arg not in prev and destino_arg != inicio_arg:
        return [], float("inf"), explored
    path = []
    cur = destino_arg
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    if not path or path[0] != inicio_arg:
        return [], float("inf"), explored
    cost = dist.get(destino_arg, float("inf"))
    if cost == INF:
        return [], float("inf"), explored
    return path, cost, explored


def a_estrella(inicio_arg, destino_arg, mapa_arg=None):
    m = _mapa(mapa_arg)

    def nbrs(x):
        return vecinos(x, m)

    def h(x):
        return heuristica(x, destino_arg)

    INF = float("inf")
    g = {inicio_arg: 0}
    prev = {inicio_arg: None}
    pq = [(h(inicio_arg), 0, inicio_arg)]
    explored = 0
    closed = set()

    while pq:
        f, gd, u = heapq.heappop(pq)
        if u in closed:
            continue
        if gd > g.get(u, INF):
            continue  # stale entry
        closed.add(u)
        explored += 1
        if u == destino_arg:
            break
        for v in nbrs(u):
            tentative = gd + coste_celda(v, m)
            if tentative < g.get(v, INF):
                g[v] = tentative
                prev[v] = u
                heapq.heappush(pq, (tentative + h(v), tentative, v))

    if destino_arg not in prev and destino_arg != inicio_arg:
        return [], float("inf"), explored
    path = []
    cur = destino_arg
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    if not path or path[0] != inicio_arg:
        return [], float("inf"), explored
    cost = g.get(destino_arg, float("inf"))
    if cost == INF:
        return [], float("inf"), explored
    return path, cost, explored


# ---------------------------------------------------------------------------
# Dynamic replanning extension (10 successive blockages)
# ---------------------------------------------------------------------------


def copiar_mapa(mapa_arg=None):
    m = _mapa(mapa_arg)
    return [row[:] for row in m]


def hay_camino(origen, destino_arg, mapa_arg):
    """BFS connectivity check (does not count as planning)."""
    if origen == destino_arg:
        return True
    seen = {origen}
    cola = [origen]
    while cola:
        u = cola.pop(0)
        for v in vecinos(u, mapa_arg):
            if v == destino_arg:
                return True
            if v not in seen:
                seen.add(v)
                cola.append(v)
    return False


def planificar(algoritmo, origen, destino_arg, mapa_arg):
    t0 = time.perf_counter()
    if algoritmo == "dijkstra":
        ruta, coste, exp = dijkstra(origen, destino_arg, mapa_arg)
    else:
        ruta, coste, exp = a_estrella(origen, destino_arg, mapa_arg)
    elapsed = time.perf_counter() - t0
    return ruta, coste, exp, elapsed


def simular_replanificacion(algoritmo, mapa_inicial, n_bloqueos=10):
    m = copiar_mapa(mapa_inicial)
    pos = inicio
    total_cost = 0
    total_explored = 0
    total_time = 0.0
    planes = 0
    rng_sim = random.Random(SEED + 99)
    historial = []

    for b in range(1, n_bloqueos + 1):
        ruta, coste, exp, elapsed = planificar(algoritmo, pos, destino, m)
        if not ruta:
            print(f"[{algoritmo}] Sin ruta en bloqueo {b}; se detiene la simulación.")
            break

        total_explored += exp
        total_time += elapsed
        planes += 1

        # Move to the midpoint of the current planned route.
        mid = max(1, len(ruta) // 2)
        mid = min(mid, len(ruta) - 1)
        segmento = ruta[: mid + 1]
        total_cost += coste_ruta(segmento, m)
        pos = ruta[mid]

        # Block a cell: prefer cells on the remaining route to force replanning.
        route_candidates = [cell for cell in ruta[mid + 1:-1] if cell not in (pos, destino)]
        fallback_candidates = [
            (i, j)
            for i in range(FILAS)
            for j in range(COLUMNAS)
            if m[i][j] is not None and (i, j) not in (pos, destino)
        ]
        seen_cells = set()
        candidatos = []
        for cell in route_candidates + fallback_candidates:
            if cell not in seen_cells:
                seen_cells.add(cell)
                candidatos.append(cell)

        bloqueada = None
        for _ in range(30):
            if not candidatos:
                break
            cell = rng_sim.choice(candidatos)
            if cell in (pos, destino):
                continue
            original = m[cell[0]][cell[1]]
            m[cell[0]][cell[1]] = None
            if hay_camino(pos, destino, m):
                bloqueada = cell
                break
            # Revert if it disconnects origin from destination.
            m[cell[0]][cell[1]] = original
            if cell in candidatos:
                candidatos.remove(cell)

        historial.append(
            {
                "bloqueo": b,
                "posicion": pos,
                "celda_bloqueada": bloqueada,
                "coste_plan": coste,
                "nodos_explorados": exp,
                "tiempo_plan": elapsed,
            }
        )
        print(
            f"[{algoritmo}] Bloqueo {b}: pos={pos}, bloqueada={bloqueada}, "
            f"coste_plan={coste}, explorados={exp}, tiempo={elapsed:.6f}s"
        )

    # Final leg to destination.
    ruta, coste, exp, elapsed = planificar(algoritmo, pos, destino, m)
    if ruta:
        total_explored += exp
        total_time += elapsed
        planes += 1
        total_cost += coste_ruta(ruta, m)
        print(
            f"[{algoritmo}] Tramo final: coste={coste}, explorados={exp}, tiempo={elapsed:.6f}s"
        )
    else:
        print(f"[{algoritmo}] No se pudo alcanzar el destino tras los bloqueos.")

    return {
        "algoritmo": algoritmo,
        "coste_total": total_cost,
        "nodos_explorados": total_explored,
        "tiempo_planificacion": total_time,
        "planes": planes,
        "historial": historial,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
ruta_d, coste_d, exp_d = dijkstra(inicio, destino)
ruta_a, coste_a, exp_a = a_estrella(inicio, destino)

# Required checks.
assert ruta_d[0] == inicio
assert ruta_d[-1] == destino
assert ruta_a[0] == inicio
assert ruta_a[-1] == destino

assert coste_d == coste_a
assert coste_d == coste_ruta(ruta_d)
assert coste_a == coste_ruta(ruta_a)

assert all(
    b in set(vecinos(a))
    for a, b in zip(ruta_a, ruta_a[1:])
)

assert exp_d > 0 and exp_a > 0

print("=== Pathfinding results ===")
print(f"Dijkstra: cost={coste_d}, explored={exp_d}, path_length={len(ruta_d)}")
print(f"A*:       cost={coste_a}, explored={exp_a}, path_length={len(ruta_a)}")
print()
print("Explored-node comparison:")
if exp_a < exp_d:
    print(f"A* expanded {exp_d - exp_a} fewer nodes than Dijkstra thanks to the Manhattan heuristic.")
elif exp_a > exp_d:
    print(f"A* expanded {exp_a - exp_d} more nodes than Dijkstra on this map (ties/implementation dependent).")
else:
    print("Both algorithms expanded the same number of nodes on this map.")
print("A* keeps optimality because the Manhattan heuristic is admissible and consistent")
print("(it never overestimates the remaining cost).")

# Visual representation of the route.
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

plot_path = ruta_a
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(FILAS):
    for j in range(COLUMNAS):
        if mapa[i][j] is None:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color="black"))
        else:
            ax.text(j + 0.5, i + 0.5, str(mapa[i][j]), ha="center", va="center", fontsize=7, color="gray")
path_set = set(plot_path)
for (i, j) in path_set:
    if (i, j) not in (inicio, destino):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color="lightgreen", alpha=0.6))
# draw path line
ys = [p[0] + 0.5 for p in plot_path]
xs = [p[1] + 0.5 for p in plot_path]
ax.plot(xs, ys, color="blue", linewidth=2, marker="o", markersize=3)
ax.scatter(inicio[1] + 0.5, inicio[0] + 0.5, color="green", s=120, label="Start (B)", zorder=5)
ax.scatter(destino[1] + 0.5, destino[0] + 0.5, color="red", s=120, label="Destination (D)", zorder=5)
ax.set_xlim(0, COLUMNAS)
ax.set_ylim(0, FILAS)
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_title("A* route on the 20x20 grid")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(results_dir, "route_plot.png"))
plt.close(fig)
print(f"Route plot saved to {os.path.join(results_dir, 'route_plot.png')}")

print("\nAll automatic checks passed successfully.")


if __name__ == "__main__":
    print("\n=== Dynamic replanning simulation (10 successive blockages) ===")
    res_d = simular_replanificacion("dijkstra", mapa, 10)
    res_a = simular_replanificacion("a_estrella", mapa, 10)

    print("\n=== Replanning comparison ===")
    print(f"{'Metric':<28}{'Dijkstra':>16}{'A*':>16}")
    print("-" * 60)
    print(f"{'Total cost traveled':<28}{res_d['coste_total']:>16}{res_a['coste_total']:>16}")
    print(f"{'Total explored nodes':<28}{res_d['nodos_explorados']:>16}{res_a['nodos_explorados']:>16}")
    print(f"{'Total planning time (s)':<28}{res_d['tiempo_planificacion']:>16.6f}{res_a['tiempo_planificacion']:>16.6f}")
    print(f"{'Planning calls':<28}{res_d['planes']:>16}{res_a['planes']:>16}")

    print("\n=== D* Lite (investigation) ===")
    print("D* Lite is an incremental heuristic search algorithm designed for dynamic")
    print("environments. Instead of recomputing a full search after every change, it")
    print("reuses the previous search tree and updates only the nodes affected by")
    print("cost changes, propagating corrections from the goal towards the start.")
    print("It searches in reverse (from goal to start), maintains g and rhs values,")
    print("and is efficient for repeated replanning with local map updates.")
    print("A full implementation is more involved (priority queue with key updates")
    print("and consistency propagation); this project keeps the simpler repeated")
    print("A*/Dijkstra replanning for clarity, while documenting D* Lite as the")
    print("recommended approach for large-scale incremental replanning.")