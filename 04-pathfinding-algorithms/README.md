# 04 - Pathfinding Algorithms: Informed Search and Route Planning

## Context
In the three previous challenges you worked with territorial clustering, anomaly detection, and text classification. Now we move to a different branch of artificial intelligence: informed search and route planning.

Imagine you are developing a support system for public services. A maintenance vehicle must travel from its base to an incident using a road network. Some roads are blocked and others have different circulation costs.

Your goal is to implement an algorithm capable of finding the lowest-cost route and compare it with an uninformed search method.

## Example Network — Synthetic 7 × 7 Map

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

The visualization is schematic. The code uses a reproducibly generated map with obstacles and variable costs.

## Mission
Implement and compare two search algorithms to find the lowest-cost path on a grid:

1. **Dijkstra** (uninformed search): explores nodes ordered by accumulated cost.
2. **A\*** (informed search): combines accumulated cost with an estimate of the remaining cost.

Compare both in terms of path cost and number of explored nodes.

## Learning Objectives
- Implement Dijkstra and A\* from scratch using a priority queue.
- Design an admissible and consistent heuristic.
- Differentiate the cost of a route from the number of explored nodes.
- Verify that A\* finds an optimal solution under the established conditions.
- Analyze when a heuristic improves efficiency and when it provides no advantage.

## Theoretical Foundation
Dijkstra selects the node with the lowest accumulated cost:

\[
g(n)
\]

A\* incorporates an estimate of the remaining cost:

\[
f(n) = g(n) + h(n)
\]

where \(g(n)\) is the cost from the origin to the current node and \(h(n)\) estimates the cost to the destination.

To guarantee optimality in this implementation, we use a heuristic that never overestimates the real cost (admissible and consistent).

## Experiment Flow
1. **Map Generation**: Creation of a reproducible synthetic grid with obstacles and variable movement costs.
2. **Modeling**:
   - **Dijkstra**: priority-queue search ordered by \(g(n)\).
   - **A\***: priority-queue search ordered by \(f(n) = g(n) + h(n)\).
3. **Heuristic Design**: Admissible and consistent heuristic (e.g., Manhattan/Euclidean distance scaled to movement costs).
4. **Evaluation**: Comparison of path cost and number of explored nodes between both algorithms.
5. **Analysis**: Discussion of when the heuristic improves efficiency and when it adds no advantage.

## Requirements
To run this project, ensure you have the dependencies listed in the `requirements.txt` file at the root of the project:

```bash
pip install -r ../requirements.txt
```

## Usage
The complete implementation is in `solution.py`. Run it with:

```bash
python solution.py
```

This script generates the reproducible map, runs both search algorithms, and reports path costs and explored nodes.

## Dynamic Replanning Extension
The simulation models 10 successive road blockages. After each blockage the vehicle
replans from its current position, avoiding the new restrictions. The script
reports total traveled cost, total planning time, and total explored nodes for
both Dijkstra and A*.

## D* Lite (Investigation)
D* Lite is an incremental heuristic search algorithm for dynamic environments.
Instead of recomputing a full search after every change, it reuses the previous
search tree and updates only the nodes affected by cost changes, propagating
corrections from the goal towards the start. It is well suited to repeated
replanning with local map updates.