# 04 - Pathfinding Algorithms: Informed Search and Route Planning

## Context
Imagine you are developing a support system for public services. A maintenance vehicle must travel from its base to an incident through a road network. Some roads are blocked and others have different circulation costs.

Your goal is to implement an algorithm capable of finding the lowest-cost route and to compare it with an uninformed search method.

## Mission
Implement and compare Dijkstra (uninformed) and A* (informed) on a reproducible 20 × 20 grid. Both must return the same optimal cost; the comparison focuses on the number of explored nodes and on planning time. A second experiment simulates 10 successive road blockages and replans after each one.

## Learning Objectives
- Implement Dijkstra and A* from scratch using a priority queue.
- Design an admissible and consistent heuristic.
- Differentiate the cost of a route from the number of explored nodes.
- Verify that A* finds an optimal solution under the stated conditions.
- Analyze when a heuristic improves efficiency and when it provides no advantage.
- Understand the motivation for incremental search in dynamic environments.

## Example Network — Synthetic 20 × 20 Map (schematic excerpt)
`B` = base, `D` = destination, `■` = obstacle. The remaining cells are traversable.

```
B . . ■ . . .
. . . ■ . . .
. . . . . ■ .
■ ■ . . . ■ .
. . . . . . .
. . ■ ■ . . .
. . . . . . D
```

This is a schematic excerpt of the corner of the 20 × 20 map. The code uses a reproducibly generated grid with obstacles and variable movement costs. A monotonic down/right path is guaranteed to exist.

## Theoretical Foundation
Dijkstra expands the node with the lowest accumulated cost:

\[
g(n)
\]

A* adds an estimate of the remaining cost:

\[
f(n) = g(n) + h(n)
\]

where `g(n)` is the cost from the origin to the current node and `h(n)` estimates the cost from that node to the destination. The Manhattan distance is used as heuristic:

\[
h(n) = |r_n - r_g| + |c_n - c_g|
\]

Since every traversable cell costs at least 1, the Manhattan distance never overestimates the real remaining cost: it is admissible. It is also consistent, because moving one step changes both the real cost and the heuristic by at most the same amount. Under these conditions A* is optimal.

## Experiment Flow
1. **Map Generation**: reproducible 20 × 20 grid with obstacles and cell costs between 1 and 5.
2. **Modeling**: Dijkstra ordered by `g(n)`, A* ordered by `f(n) = g(n) + h(n)`.
3. **Heuristic Design**: admissible and consistent Manhattan heuristic scaled to movement costs.
4. **Evaluation**: compare path cost, path length and explored nodes.
5. **Dynamic Replanning**: 10 successive blockages, replanning from the current position each time.
6. **Analysis**: compare total traveled cost, explored nodes and planning time; discuss D* Lite.

## Results

### Single route (base → destination)

| Metric | Dijkstra | A* |
| :--- | ---: | ---: |
| Path cost | 98 | 98 |
| Explored nodes | 322 | **309** |
| Path length | 41 | 41 |

Both algorithms return the same optimal route. On this map the informed search expands 13 fewer nodes, a modest improvement of about 4 %.

### Dynamic replanning (10 blockages, 11 planning calls)

| Metric | Dijkstra | A* |
| :--- | ---: | ---: |
| Total traveled cost | 98 | 98 |
| Total explored nodes | 784 | **568** |
| Total planning time (s) | 0.001229 | **0.001015** |
| Planning calls | 11 | 11 |

The advantage of A* is much clearer across repeated searches: 216 fewer node expansions, about 28 % less work. Absolute times are tiny and dominated by Python overhead at this grid size. The route plot is saved to `results/route_plot.png`.

## Requirements
Dependencies are listed in the root `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

## Usage
The complete implementation is in `solution.py`. Run it with:

```bash
python solution.py
```

The script runs automatic checks (path endpoints, equal costs, valid adjacency), saves the route plot and prints both the single-route and replanning comparisons.

## Dynamic Replanning Extension
The simulation models 10 successive road blockages. After each blockage the vehicle moves to the midpoint of the current route, the simulator blocks a cell on the remaining route (when possible) and the vehicle replans from its new position. Blockages that would disconnect the destination are rejected. The script reports total traveled cost, total planning time and total explored nodes for both algorithms.

## D* Lite (Investigation)
D* Lite is an incremental heuristic search algorithm for dynamic environments. Instead of recomputing a full search after every change, it reuses the previous search tree and updates only the nodes affected by cost changes, propagating corrections from the goal towards the start. It searches in reverse, maintains `g` and `rhs` values, and is well suited to repeated replanning with local map updates.

A full implementation requires a priority queue with key updates and consistency propagation. This project keeps the simpler repeated A*/Dijkstra replanning for clarity and documents D* Lite as the recommended approach for large-scale incremental replanning.

## Conclusions and Limitations
- A* is optimal because its heuristic is admissible and consistent, and it consistently explores fewer nodes than Dijkstra on this map.
- The benefit of a heuristic depends on the map and on the cost scale. With uniform costs or very cheap cell costs, the heuristic can lose accuracy and A* may behave like Dijkstra.
- Timing comparisons at 20 × 20 cells are noisy because of interpreter overhead; node counts are the more meaningful metric here.
- Replanning from scratch, as done here, is simple but wasteful. In a large dynamic network an incremental algorithm such as D* Lite would reuse previous work.
