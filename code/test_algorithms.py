"""Independent BFS/Dijkstra and exhaustive game checks, standard library only."""

import random
import unittest
from heapq import heappop, heappush

from run_experiments import (
    MAZE, WEIGHTED, POSITIONS, TREASURE_GRID, VALUES, distance_matrix,
    example_tree, TreeNode, TreasureState, alpha_beta, tree_from_dict,
    tree_search, treasure_search, astar, maze_search, weighted_search,
    bfs_unit_grid,
)


def reference_distance(grid, start, goal):
    """Independent Dijkstra with finalized set, used only as a test oracle."""
    queue, finalized = [(0, start)], set()
    while queue:
        cost, node = heappop(queue)
        if node in finalized:
            continue
        finalized.add(node)
        if node == goal:
            return cost
        r, c = node
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c - 1), (r, c + 1)):
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr][nc] is not None and (nr, nc) not in finalized):
                heappush(queue, (cost + grid[nr][nc], (nr, nc)))
    return -1


class AlgorithmTests(unittest.TestCase):
    def test_bfs_baseline_and_weighted_counterexample(self):
        self.assertEqual(bfs_unit_grid([[1,1],[1,1]],(0,0),(1,1))['cost'],2)
        self.assertEqual(bfs_unit_grid([[1,None,1]],(0,0),(0,2))['path'],[])
        grid=[[1,9,1],[1,1,1]]
        result=bfs_unit_grid(grid,(0,0),(0,2))
        self.assertEqual((result['steps'],result['cost']),(2,10))
        self.assertEqual(astar(grid,(0,0),(0,2)).cost,4)

    def test_official_examples(self):
        path = maze_search((0, 0), (3, 4), MAZE).path
        self.assertEqual(len(path) - 1, 7)
        self.assertEqual(weighted_search((0, 0), (4, 4), WEIGHTED).cost, 7)
        self.assertEqual(alpha_beta(example_tree()), 5)
        distances = distance_matrix()
        for depth, target, value in ((2, "T2", -1), (4, "T3", 3)):
            for pruning in (False, True):
                for order in ("natural", "gain_desc", "gain_asc"):
                    result = treasure_search(distances, VALUES, depth, pruning, order)
                    self.assertEqual((result.best_target, result.best_value), (target, value))

    def test_path_and_edge_cases(self):
        self.assertEqual(astar([[1]], (0, 0), (0, 0)).path, [(0, 0)])
        self.assertEqual(astar([[1, None, 1]], (0, 0), (0, 2)).cost, -1)
        self.assertEqual(astar([[None]], (0, 0), (0, 0)).path, [])
        self.assertEqual(astar([[1, 0]], (0, 0), (0, 1)).cost, 0)
        self.assertEqual(astar([[0, 0], [0, 0]], (0, 0), (1, 1)).cost, 0)
        for invalid in ([], [[1], [1, 2]], [[-1]], [[float("inf")]]):
            with self.assertRaises(ValueError):
                astar(invalid, (0, 0), (0, 0))
        with self.assertRaises(ValueError):
            astar([[1]], (-1, 0), (0, 0))

    def test_random_grids_against_independent_dijkstra(self):
        rng = random.Random(20261109)
        for _ in range(150):
            grid = [[None if rng.random() < .2 else rng.randrange(0, 10)
                     for _ in range(7)] for _ in range(6)]
            grid[0][0], grid[-1][-1] = 1, rng.choice([0, 1, 3])
            expected = reference_distance(grid, (0, 0), (5, 6))
            for deeper in (False, True):
                result = astar(grid, (0, 0), (5, 6), prefer_deeper=deeper)
                self.assertEqual(result.cost, expected)
                if result.path:
                    self.assertEqual(result.path[0], (0, 0))
                    self.assertEqual(result.path[-1], (5, 6))
                    self.assertEqual(sum(grid[r][c] for r, c in result.path[1:]), expected)
                    for a, b in zip(result.path, result.path[1:]):
                        self.assertEqual(abs(a[0]-b[0])+abs(a[1]-b[1]), 1)

    def test_distance_matrix_independently(self):
        distances = distance_matrix()
        for a, p in POSITIONS.items():
            for b, q in POSITIONS.items():
                self.assertEqual(distances[a][b], reference_distance(TREASURE_GRID, p, q))

    def test_random_trees_against_plain_recursive_minimax(self):
        rng = random.Random(42)

        def build(depth):
            if depth == 0 or rng.random() < .15:
                return TreeNode(rng.randint(-20, 20))
            return TreeNode(children=[build(depth-1) for _ in range(rng.randint(1, 4))])

        def oracle(node, maximize=True):
            if not node.children:
                return node.value
            return (max if maximize else min)(oracle(c, not maximize) for c in node.children)

        for _ in range(100):
            root = build(4)
            for reverse in (False, True):
                self.assertEqual(tree_search(root, True, reverse).best_value, oracle(root))
        self.assertEqual(alpha_beta(tree_from_dict({"children": [{"value": 2}, {"value": -1}]})), 2)

    def test_treasure_against_exhaustive_enumeration(self):
        distances = distance_matrix()

        def oracle(a, b, remaining, score, player, depth):
            if depth == 0 or not remaining:
                return score
            outcomes = []
            for t in remaining:
                gain = VALUES[t] - distances[a if player == "A" else b][t]
                outcomes.append(oracle(t if player == "A" else a,
                                       b if player == "A" else t,
                                       tuple(x for x in remaining if x != t),
                                       score + (gain if player == "A" else -gain),
                                       "B" if player == "A" else "A", depth-1))
            return (max if player == "A" else min)(outcomes)

        for depth in range(6):
            expected = oracle("A", "B", tuple(VALUES), 0, "A", depth)
            for pruning, cache in ((False, False), (True, False), (False, True)):
                result = treasure_search(distances, VALUES, depth, pruning, cache=cache)
                self.assertEqual(result.best_value, expected)
        state = TreasureState(remaining=())
        self.assertEqual(treasure_search(distances, VALUES, 4, initial=state).best_value, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
