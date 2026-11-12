"""实验二统一入口：四题算法、样例、基准实验和拓展验证。

从 practice2 运行 python ./code/run_experiments.py，默认执行全部实验。
--mode baseline 仅运行四题基准；--mode probes 仅运行拓展。
仅使用标准库。导入本模块不会运行实验或写文件。
"""

import argparse
import csv
import json
import platform
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from heapq import heappop, heappush
from itertools import count
from math import inf, isfinite
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter



# ======================================================================
# 一、共用 A* 与迷宫/加权接口
# ======================================================================

@dataclass
class SearchResult:
    path: list
    cost: float
    expanded: int
    generated: int
    stale_pops: int
    seconds: float


def astar(grid, start, goal, heuristic="manhattan", prefer_deeper=False):
    """Numeric nonnegative costs; None means blocked; start entry is free.

    expanded counts valid pops INCLUDING the goal. generated counts all heap
    pushes, including the start and improvements of already discovered nodes.
    """
    started = perf_counter()
    start, goal = tuple(start), tuple(goal)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid must be a nonempty rectangular matrix")
    height, width = len(grid), len(grid[0])
    for row in grid:
        for value in row:
            if value is not None and (not isinstance(value, (int, float))
                                      or not isfinite(value) or value < 0):
                raise ValueError("terrain costs must be finite nonnegative numbers")
    for r, c in (start, goal):
        if not (0 <= r < height and 0 <= c < width):
            raise ValueError("endpoint outside grid")
    if heuristic not in ("zero", "manhattan"):
        raise ValueError("unknown heuristic")
    if grid[start[0]][start[1]] is None or grid[goal[0]][goal[1]] is None:
        return SearchResult([], -1, 0, 0, 0, perf_counter() - started)

    # Every route with Manhattan distance D has at least D-1 non-goal
    # entries plus the final goal entry. This also handles a zero-cost E.
    non_goal = [v for r, row in enumerate(grid) for c, v in enumerate(row)
                if v is not None and (r, c) != goal]
    min_cost = min(non_goal, default=0)
    goal_cost = grid[goal[0]][goal[1]]

    def h(node):
        distance = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
        if heuristic == "zero" or distance == 0:
            return 0
        return (distance - 1) * min_cost + goal_cost

    serial = count()
    heap = [(h(start), 0, next(serial), 0, start)]
    best = {start: 0}
    parent = {}
    expanded, generated, stale = 0, 1, 0
    while heap:
        _, _, _, cost, node = heappop(heap)
        if cost != best[node]:
            stale += 1
            continue
        expanded += 1
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            return SearchResult(path[::-1], cost, expanded, generated, stale,
                                perf_counter() - started)
        r, c = node
        # Fixed order makes equal-cost results reproducible.
        for nr, nc in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)):
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            terrain = grid[nr][nc]
            if terrain is None:
                continue
            neighbour = (nr, nc)
            candidate = cost + terrain
            if candidate < best.get(neighbour, inf):
                best[neighbour] = candidate
                parent[neighbour] = node
                tie = -candidate if prefer_deeper else 0
                heappush(heap, (candidate + h(neighbour), tie, next(serial),
                                candidate, neighbour))
                generated += 1
    return SearchResult([], -1, expanded, generated, stale, perf_counter() - started)


def maze_search(start, goal, grid, **options):
    if any(cell not in ("S", "E", ".", "X") for row in grid for cell in row):
        raise ValueError("maze cells must be S, E, . or X")
    return astar([[None if c == "X" else 1 for c in row] for row in grid],
                 start, goal, **options)


def weighted_search(start, goal, grid, **options):
    def cost(cell):
        if cell == "X" or cell is None:
            return None
        if cell == "E":
            return 0
        if cell == "S":
            return 1
        return cell
    return astar([[cost(c) for c in row] for row in grid], start, goal, **options)


def maze_path(start, goal, grid):
    return maze_search(start, goal, grid).path


def weighted_cost(start, goal, grid):
    return weighted_search(start, goal, grid).cost


# ======================================================================
# 二、树搜索与宝藏博弈
# ======================================================================

@dataclass
class TreeNode:
    value: int | None = None
    children: list = field(default_factory=list)


def tree_from_dict(data, maximizing=True):
    if "children" not in data or not data["children"]:
        if not isinstance(data.get("value"), (int, float)):
            raise ValueError("leaf must have a numeric value")
        return TreeNode(value=data["value"])
    if data.get("is_maximizing_player", maximizing) != maximizing:
        raise ValueError("players must alternate, starting with MAX")
    return TreeNode(children=[tree_from_dict(child, not maximizing)
                              for child in data["children"]])


@dataclass
class GameResult:
    best_target: str | None
    best_value: float
    visited: int = 0
    leaves: int = 0
    cutoffs: int = 0
    cache_hits: int = 0
    seconds: float = 0


def tree_search(root, pruning=True, reverse=False):
    started = perf_counter()
    result = GameResult(None, 0)

    def visit(node, maximizing, alpha, beta):
        result.visited += 1
        if not node.children:
            if node.value is None:
                raise ValueError("leaf needs a value")
            result.leaves += 1
            return node.value
        best = -inf if maximizing else inf
        children = list(reversed(node.children)) if reverse else node.children
        for index, child in enumerate(children):
            value = visit(child, not maximizing, alpha, beta)
            best = max(best, value) if maximizing else min(best, value)
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if pruning and alpha >= beta and index + 1 < len(children):
                result.cutoffs += 1
                break
        return best

    result.best_value = visit(root, True, -inf, inf)
    result.seconds = perf_counter() - started
    return result


def alpha_beta(root):
    return tree_search(root).best_value


@dataclass(frozen=True)
class TreasureState:
    a_position: str = "A"
    b_position: str = "B"
    remaining: tuple = ("T1", "T2", "T3", "T4")
    score_a: int = 0
    score_b: int = 0
    player: str = "A"


def treasure_search(distances, values, depth, pruning=True, order="natural",
                    cache=False, initial=None):
    """Depth is number of individual turns. Cache stores exact minimax values.

    Never cache alpha-beta cutoff values as if they were exact.
    Root ties always use lexical target order, independent of traversal order.
    """
    if depth < 0 or order not in ("natural", "gain_desc", "gain_asc"):
        raise ValueError("invalid depth or order")
    if pruning and cache:
        raise ValueError("exact cache is provided for plain minimax only")
    started = perf_counter()
    result = GameResult(None, 0)
    memo = {}
    initial = initial if initial is not None else TreasureState()

    def moves(state):
        position = state.a_position if state.player == "A" else state.b_position
        reachable = [t for t in state.remaining if distances[position][t] >= 0]
        if order == "natural":
            return sorted(reachable)
        sign = -1 if order == "gain_desc" else 1
        return sorted(reachable, key=lambda t: (sign * (values[t] - distances[position][t]), t))

    def advance(state, target):
        a_turn = state.player == "A"
        position = state.a_position if a_turn else state.b_position
        gain = values[target] - distances[position][target]
        return TreasureState(target if a_turn else state.a_position,
                             state.b_position if a_turn else target,
                             tuple(t for t in state.remaining if t != target),
                             state.score_a + (gain if a_turn else 0),
                             state.score_b + (0 if a_turn else gain),
                             "B" if a_turn else "A")

    def visit(state, remaining_depth, alpha, beta):
        result.visited += 1
        # Future utility depends on positions, remaining targets and turn;
        # store the increment relative to current utility for more reuse.
        utility = state.score_a - state.score_b
        key = (state.a_position, state.b_position, state.remaining,
               state.player, remaining_depth)
        if cache and key in memo:
            result.cache_hits += 1
            return utility + memo[key]
        targets = moves(state) if remaining_depth > 0 else []
        if not targets:
            result.leaves += 1
            return utility
        maximizing = state.player == "A"
        best = -inf if maximizing else inf
        for index, target in enumerate(targets):
            value = visit(advance(state, target), remaining_depth - 1, alpha, beta)
            best = max(best, value) if maximizing else min(best, value)
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if pruning and alpha >= beta and index + 1 < len(targets):
                result.cutoffs += 1
                break
        if cache:
            memo[key] = best - utility
        return best

    # Full windows for each root action make all root values exact and allow
    # stable tie-breaking. Bounds still propagate inside each action subtree.
    result.visited = 1
    targets = moves(initial) if depth > 0 else []
    if not targets:
        result.leaves = 1
        result.best_value = initial.score_a - initial.score_b
    else:
        root_values = {t: visit(advance(initial, t), depth - 1, -inf, inf)
                       for t in targets}
        choose = max if initial.player == "A" else min
        result.best_value = choose(root_values.values())
        result.best_target = min(t for t, v in root_values.items()
                                 if v == result.best_value)
    result.seconds = perf_counter() - started
    return result


# ======================================================================
# 三、四题输入与距离矩阵
# ======================================================================

MAZE = [list(row) for row in ("S..X.", ".X...", ".X.X.", "...XE", "XX...")]
WEIGHTED = [["S", 1, 1, 8, 1], [1, 8, 3, 1, 1], [1, 8, 1, 8, 1],
            [1, 1, 1, 8, 1], [8, 8, 1, 1, "E"]]
TREASURE_GRID = [
    [1, 1, 1, None, 2, 1, 1],
    [1, None, 6, None, 1, None, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [None, 1, None, 8, None, 1, None],
    [1, 1, 1, 1, 1, 1, 1],
    [1, None, 9, None, 1, None, 1],
    [1, 1, 1, None, 1, 1, 1],
]
POSITIONS = {"A": (0, 0), "B": (6, 6), "T1": (0, 6), "T2": (2, 2),
             "T3": (4, 4), "T4": (6, 0)}
VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}


def example_tree():
    def pair(a, b):
        return TreeNode(children=[TreeNode(a), TreeNode(b)])
    return TreeNode(children=[TreeNode(children=[pair(3, 5), pair(6, 9)]),
                              TreeNode(children=[pair(6, 9), pair(1, 2)])])


def distance_matrix():
    return {a: {b: astar(TREASURE_GRID, p, q).cost for b, q in POSITIONS.items()}
            for a, p in POSITIONS.items()}


# ======================================================================
# 四、基准对照与数据导出
# ======================================================================

def benchmark(call, repeats):
    call()  # warm-up; timings exclude file output and matrix preprocessing
    samples = [call() for _ in range(repeats)]
    result = asdict(samples[0])
    result["seconds"] = median(s.seconds for s in samples)
    result["median_us"] = result["seconds"] * 1e6
    return result


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def run_baseline(repeats=101, output_dir=None):
    """Run all four questions and comparisons; export baseline data."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    base = Path("./")
    output = Path(output_dir) if output_dir is not None else Path("./results")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "实验结果记录.md" if output_dir is not None else base / "实验结果记录.md"
    distances = distance_matrix()
    searches, trees, games = [], [], []
    for name, fn, start, goal, grid in (
        ("maze", maze_search, (0, 0), (3, 4), MAZE),
        ("weighted", weighted_search, (0, 0), (4, 4), WEIGHTED),
        ("open_30", astar, (0, 0), (29, 29), [[1]*30 for _ in range(30)]),
        ("unreachable", astar, (0, 0), (2, 2), [[1, 1, 1], [None]*3, [1, 1, 1]]),
        ("zero_goal", astar, (0, 0), (0, 1), [[1, 0]]),
    ):
        for heuristic, deeper in (("zero", False), ("manhattan", False), ("manhattan", True)):
            result = benchmark(lambda: fn(start=start, goal=goal, grid=grid,
                                         heuristic=heuristic, prefer_deeper=deeper), repeats)
            searches.append(dict(case=name, heuristic=heuristic, prefer_deeper=deeper, **result))
    root = example_tree()
    for pruning in (False, True):
        for reverse in (False, True):
            trees.append(dict(pruning=pruning, reverse=reverse,
                              **benchmark(lambda: tree_search(root, pruning, reverse), repeats)))
    for depth in (2, 4):
        for pruning, cache in ((False, False), (True, False), (False, True)):
            for order in ("natural", "gain_desc", "gain_asc"):
                games.append(dict(depth=depth, pruning=pruning, cache=cache, order=order,
                                  **benchmark(lambda: treasure_search(distances, VALUES, depth,
                                                                      pruning, order, cache), repeats)))

    # Exact root-action values, independently solve each resulting subtree.
    action_values = {}
    for depth in (2, 4):
        action_values[depth] = {}
        for target in VALUES:
            gain = VALUES[target] - distances["A"][target]
            state = TreasureState(a_position=target,
                                  remaining=tuple(t for t in VALUES if t != target),
                                  score_a=gain, player="B")
            action_values[depth][target] = treasure_search(
                distances, VALUES, depth-1, False, initial=state).best_value

    environment = {"run_time": datetime.now().astimezone().isoformat(),
                   "python": platform.python_version(), "platform": platform.platform(),
                   "repeats": repeats, "seed_grids": 20261109, "seed_trees": 42}
    raw = dict(environment=environment, distances=distances, astar=searches,
               trees=trees, treasure=games, root_action_values=action_values)
    (output / "results.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, rows in (("astar.csv", searches), ("tree.csv", trees), ("treasure.csv", games)):
        fields = [key for key in rows[0] if key != "path"]
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    names = list(distances)
    maze = searches[1]
    weighted = searches[4]
    terrain = [[1 if v == "S" else 0 if v == "E" else v for v in row] for row in WEIGHTED]
    costs = [terrain[r][c] for r, c in weighted["path"][1:]]
    cumulative = []
    total = 0
    for point, cost in zip(weighted["path"][1:], costs):
        total += cost
        cumulative.append([point, cost, total])
    document = f"""# 实验二流程与结果记录

本文件由 `code/run_experiments.py` 自动生成，作为实验报告的数据底稿。
人工过程记录见 `实验流程文档.md`；完整原始数据见 `results/results.json`。

## 1. 环境与复现

- 实际运行时间：{environment['run_time']}
- Python：{environment['python']}；系统：{environment['platform']}
- 工作目录：`practice2`；运行 `python ./code/run_experiments.py`。
- 每项先预热 1 次，再运行 {repeats} 次，报告耗时中位数（微秒）。
- 仅使用 Python 标准库；计时不含文件输出、距离矩阵预处理。
- 小规模问题的耗时容易受系统调度影响，应以节点数作为主要比较指标。

## 2. 实现与统计口径

A* 状态为 `(row, col)`，g 为进入各格的累计代价，起点不收费；f=g+h。
堆元素包含 f、平局优先级、插入序号、g、坐标。仅当 g 严格改善时更新父节点；
弹出过期条目时跳过，允许更优路径重新进入队列。四邻接顺序固定为上、右、下、左。
expanded 包含成功弹出的终点；generated 是入堆总次数（含起点和重复改善）；stale_pops 为过期弹出数。
默认 f 相同按插入顺序访问；优化版本优先 g 较大的节点。

博弈 visited 包括根、内部节点、叶节点和缓存命中调用；leaves 是实际终止评价次数；
cutoffs 仅计确实跳过至少一个后续孩子的事件，不能等同于跳过的节点数。
双方累计收益分别保存，终止评价 U=Score_A−Score_B。
H 表示单方行动次数，四回合结束。无剩余宝藏或深度耗尽时立即评价。
若当前无可达宝藏则终止，该规则仅为额外输入的约定，原题不会触发。

## 3. 题目一：迷宫寻路

- 得到路径：`{maze['path']}`
- 步数/代价：{maze['cost']}，符合预期 7；相同长度路径均合法。
- 父指针沿目标回溯后反转得到完整路径。

{table(['启发式', '平局优先较大g', '扩展', '入堆', '耗时中位数/μs'], [[x['heuristic'], x['prefer_deeper'], x['expanded'], x['generated'], f"{x['median_us']:.2f}"] for x in searches if x['case']=='maze'])}

曼哈顿距离利用目标方向信息，但障碍物、同 f 值节点及平局策略会影响具体收益。
不能从小地图推断 A* 总能显著减少扩展数。

## 4. 题目二：加权地形

- 路径：`{weighted['path']}`
- 最小代价：{weighted['cost']}，符合预期 7；起点不收费，E 的进入代价为 0。

{table(['进入坐标', '本步代价', '累计代价'], cumulative)}

若 D 为到目标的曼哈顿距离，直接使用 h=D 在目标相邻格就可能高估（实际只需 0），
因此不满足可采纳性。实现使用 h=0（D=0 时），否则 h=(D−1)×m+c_goal，
m 为所有可通行非终点格的最小代价。本例 m=1、c_goal=0，即 max(D−1,0)。
任意路径至少含 D−1 个非终点进入步骤及最后一步，因此它是下界；
对每条普通边 h 的下降不超过 m，对进入终点的边恰为 c_goal，亦满足一致性。
若地图存在零代价格，m 可能为 0，启发能力随之减弱。

{table(['启发式', '平局优先较大g', '扩展', '入堆', '耗时中位数/μs'], [[x['heuristic'], x['prefer_deeper'], x['expanded'], x['generated'], f"{x['median_us']:.2f}"] for x in searches if x['case']=='weighted'])}

## 5. 题目三：博弈树

以 JSON 给出的完整树为准，左右 MIN 节点分别包含两个 MAX 子节点。
叶值回传为左侧 max(3,5)=5、max(6,9)=9，得到 min(5,9)=5；
右侧 max(6,9)=9、max(1,2)=2，得到 min(9,2)=2；根得 max(5,2)=5。
默认从左到右；reverse=True 是用于对照的反向遍历。
例如左侧第二个 MAX 节点已有 beta=5，访问叶子 6 后 alpha≥beta，可以跳过叶子 9。

{table(['Alpha-beta', '反向顺序', '值', '访问节点', '叶评价', '剪枝事件', '耗时/μs'], [[x['pruning'], x['reverse'], x['best_value'], x['visited'], x['leaves'], x['cutoffs'], f"{x['median_us']:.2f}"] for x in trees])}

## 6. 题目四：宝藏分配

A、B、所有宝藏所在格的进入代价为 1，不能套用题目二的终点零代价。
对 6 个位置的所有有序对分别运行 A*，得到下表。一般的进入权值图距离可不对称；
本题端点代价都为 1，因此矩阵对称。经过其他宝藏不收集，另一玩家不阻挡。

{table(['起点/终点'] + names, [[a] + [distances[a][b] for b in names] for a in names])}

状态字段：`(a_position, b_position, remaining, score_a, score_b, player)`。
选择目标后仅更新行动玩家位置，移除该宝藏并累加 Vi−d；允许负收益，不能跳过必需回合。
根节点每个候选用完整 alpha/beta 窗口单独求精确值，同值按 T1→T4 的顺序选取，
从而保证不同遍历顺序的首步结果可直接比较。这会减少跨根候选的剪枝机会，
以下 Alpha-beta 数据仅对应此实现，不能作为所有 Alpha-beta 实现的固定节点数。

{table(['深度', '首选T1时的U', '首选T2时的U', '首选T3时的U', '首选T4时的U'], [[d] + [action_values[d][t] for t in VALUES] for d in (2,4)])}

H=2 只考虑 A、B 各一次行动，T2 的短期收益最优；H=4 纳入后续位置与剩余宝藏，
此时应选择 T3。候选首步的完整回传值见上表，而非只比较第一步 Gain。
gain_desc / gain_asc 分别按当前行动玩家的即时 Gain 从高到低/从低到高遍历，
这只是排序启发，不保证与最终博弈值顺序相同。

{table(['H', 'Alpha-beta', '缓存', '顺序', '首步', 'U', '访问', '叶评价', '剪枝', '缓存命中', '耗时/μs'], [[x['depth'], x['pruning'], x['cache'], x['order'], x['best_target'], x['best_value'], x['visited'], x['leaves'], x['cutoffs'], x['cache_hits'], f"{x['median_us']:.2f}"] for x in games])}

缓存仅用于普通 Minimax，保存扣除当前 U 后的未来增量；键为双方位置、剩余宝藏、当前玩家和剩余深度。
未将 Alpha-beta 截断得到的界当作精确值存储。缓存是否收益须看实际命中数，不能预设有效。

## 7. 自设对照与拓展

{table(['场景', '启发式', '较大g优先', '最优代价', '扩展', '入堆', '耗时/μs'], [[x['case'], x['heuristic'], x['prefer_deeper'], x['cost'], x['expanded'], x['generated'], f"{x['median_us']:.2f}"] for x in searches if x['case'] not in ('maze','weighted')])}

30×30 开放网格中大量节点具有同一 f，曼哈顿距离配合 FIFO 平局规则仍可能扩展大量节点；
优先较大 g 可以沿最优路径更快到达目标，展示了平局规则的重要性。
此优化只在 f 相同时改变顺序，保留原启发式下的最优性保证；不保证对每张图都更快。
无解案例确认返回空路径和 -1；零代价终点案例确认修正后的启发式仍给出代价 0。

## 8. 报告编写提示

报告应区分正确性、节点开销和墙钟耗时；将输出数字与具体状态、路径及剪枝边界结合。
引用以上完整表格或筛选有代表性的对照，不能只展示截图。
将算法局限写清：最小地形代价太小时启发函数弱、子节点排序只反映即时收益、
小树缓存可能几乎无命中、四宝藏场景不足以说明大规模性能。
本文件提供结果与分析素材，不替代个人总结或最终实验报告。
"""
    report_path.write_text(document, encoding="utf-8")
    print("Distance matrix:")
    print(table(["from/to"] + names, [[a] + [distances[a][b] for b in names] for a in names]))
    print("Maze:", maze["path"], "cost =", maze["cost"])
    print("Weighted:", weighted["path"], "cost =", weighted["cost"])
    print("Tree value:", trees[0]["best_value"])
    for row in games:
        if not row["cache"] and row["order"] == "natural":
            print({key: row[key] for key in ("depth", "pruning", "best_target", "best_value", "visited", "cutoffs")})
    print("Results saved to:", output)
    print("Report data:", report_path)
    return raw


# ======================================================================
# 五、共享窗口与缓存拓展
# ======================================================================

def shared_window_probe(distances, depth, order):
    counts = dict(visited=0, leaves=0, cutoffs=0)

    def visit(a, b, remaining, utility, a_turn, left, alpha, beta):
        counts["visited"] += 1
        if left == 0 or not remaining:
            counts["leaves"] += 1
            return utility, []
        position = a if a_turn else b
        targets = sorted(remaining)
        if order != "natural":
            sign = -1 if order == "gain_desc" else 1
            targets.sort(key=lambda t: (sign * (VALUES[t]-distances[position][t]), t))
        best = -inf if a_turn else inf
        line = []
        for index, target in enumerate(targets):
            gain = VALUES[target]-distances[position][target]
            value, continuation = visit(
                target if a_turn else a, b if a_turn else target,
                tuple(t for t in remaining if t != target),
                utility + (gain if a_turn else -gain), not a_turn,
                left-1, alpha, beta)
            if (a_turn and value > best) or (not a_turn and value < best):
                best, line = value, [target] + continuation
            if a_turn:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if alpha >= beta and index+1 < len(targets):
                counts["cutoffs"] += 1
                break
        return best, line

    value, line = visit("A", "B", tuple(VALUES), 0, True, depth, -inf, inf)
    return dict(depth=depth, order=order, best_target=line[0], best_value=value, **counts)


def exact_line(distances, depth):
    """Full minimax: every returned branch value is exact."""
    def visit(a, b, remaining, utility, a_turn, left):
        if left == 0 or not remaining:
            return utility, []
        options = []
        for target in sorted(remaining):
            gain = VALUES[target]-distances[a if a_turn else b][target]
            value, tail = visit(target if a_turn else a, b if a_turn else target,
                                tuple(t for t in remaining if t != target),
                                utility + (gain if a_turn else -gain),
                                not a_turn, left-1)
            options.append((value, target, tail))
        best = (max if a_turn else min)(o[0] for o in options)
        _, target, tail = next(o for o in options if o[0] == best)
        return best, [target] + tail
    value, line = visit("A", "B", tuple(VALUES), 0, True, depth)
    a, b, score_a, score_b = "A", "B", 0, 0
    steps = []
    for turn, target in enumerate(line):
        a_turn = turn % 2 == 0
        position = a if a_turn else b
        distance = distances[position][target]
        gain = VALUES[target]-distance
        if a_turn:
            score_a, a = score_a+gain, target
        else:
            score_b, b = score_b+gain, target
        steps.append(dict(player="A" if a_turn else "B", source=position,
                          target=target, distance=distance, gain=gain,
                          score_a=score_a, score_b=score_b, utility=score_a-score_b))
    return dict(depth=depth, best_value=value, steps=steps)


def run_probes(output_dir=None):
    """Run shared-window, horizon and multi-treasure cache comparisons."""
    distances = distance_matrix()
    windows = []
    for depth in (2, 4):
        for order in ("natural", "gain_desc", "gain_asc"):
            row = shared_window_probe(distances, depth, order)
            baseline = treasure_search(distances, VALUES, depth, True, order)
            assert (row["best_target"], row["best_value"]) == (baseline.best_target, baseline.best_value)
            row["separate_windows_visited"] = baseline.visited
            windows.append(row)

    # Small explicit expansion: open unit-cost map (Manhattan distances),
    # 5/6/8 treasures, full n-ply horizon. Independent of original map.
    locations = {"A": (0, 0), "B": (8, 8), "T1": (0, 8), "T2": (2, 2),
                 "T3": (4, 4), "T4": (8, 0), "T5": (1, 6),
                 "T6": (6, 1), "T7": (7, 5), "T8": (3, 7)}
    cache_probes = []
    for n in (5, 6, 8):
        values = {"T"+str(i): 5+i for i in range(1, n+1)}
        d = {a: {b: abs(p[0]-q[0])+abs(p[1]-q[1]) for b, q in locations.items()}
             for a, p in locations.items()}
        state = TreasureState(remaining=tuple(values))
        plain = treasure_search(d, values, n, False, initial=state)
        cached = treasure_search(d, values, n, False, cache=True, initial=state)
        assert (plain.best_target, plain.best_value) == (cached.best_target, cached.best_value)
        cache_probes.append(dict(treasures=n, depth=n, best_target=plain.best_target,
                                 best_value=plain.best_value, plain_visited=plain.visited,
                                 cached_visited=cached.visited, cache_hits=cached.cache_hits))
    horizons = []
    for depth in range(1, 5):
        result = treasure_search(distances, VALUES, depth, False)
        horizons.append(dict(depth=depth, best_target=result.best_target,
                             best_value=result.best_value))
    data = dict(shared_root_windows=windows, horizons=horizons,
                exact_lines=[exact_line(distances, d) for d in (2,4)],
                expanded_cache=cache_probes, expanded_locations=locations,
                expanded_values_rule="Vi=5+i; open unit-cost map; H=n")
    output = Path(output_dir) if output_dir is not None else Path("./results")
    destination = output / "opportunity_probe.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def bfs_unit_grid(grid, start, goal):
    """Fewest-step baseline. Numeric weights are ignored, deliberately.

    It is an optimal-cost baseline only when every traversable cell costs 1.
    Return the actual entry cost of its path as well as its step count.
    """
    queue, parent = deque([start]), {start: None}
    expanded = 0
    while queue:
        node = queue.popleft()
        expanded += 1
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return dict(cost=sum(grid[r][c] for r, c in path[1:]),
                        steps=len(path)-1, expanded=expanded, path=path)
        r, c = node
        for nr, nc in ((r-1,c), (r,c+1), (r+1,c), (r,c-1)):
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr][nc] is not None and (nr,nc) not in parent):
                parent[nr,nc] = node
                queue.append((nr,nc))
    return dict(cost=-1, steps=-1, expanded=expanded, path=[])


def timed_call(call, repeats=7):
    """External whole-call timing; exclude input generation and file output."""
    result = call()
    times = []
    for _ in range(repeats):
        started = perf_counter()
        call()
        times.append((perf_counter()-started)*1e6)
    return result, median(times)


def run_research(output_dir=None):
    """Undergraduate-level paired ablations, baselines and sensitivity study."""
    # Fixed size and start/goal; do not carve a route or filter failed maps.
    map_rows = []
    for density in (0.0, 0.15, 0.30):
        for seed in range(20):
            rng = random.Random(12000+seed)
            grid = [[None if rng.random() < density else 1 for _ in range(20)]
                    for _ in range(20)]
            grid[0][0] = grid[-1][-1] = 1
            reference, us = timed_call(lambda: bfs_unit_grid(grid,(0,0),(19,19)))
            reachable = reference['cost'] >= 0
            common = dict(density=density, seed=12000+seed, reachable=reachable)
            map_rows.append(dict(**common, strategy='BFS', cost=reference['cost'],
                                 expanded=reference['expanded'], median_us=us))
            for heuristic in ('zero','manhattan'):
                for deeper in (False,True):
                    result, us = timed_call(lambda: astar(grid,(0,0),(19,19),
                                                          heuristic, deeper))
                    assert result.cost == reference['cost']
                    map_rows.append(dict(**common,
                        strategy=heuristic+('_larger_g' if deeper else '_FIFO'),
                        cost=result.cost, expanded=result.expanded, median_us=us))
    map_summary = []
    for density in (0.0,0.15,0.30):
        for reachable in (True,False):
            for strategy in ('BFS','zero_FIFO','zero_larger_g',
                             'manhattan_FIFO','manhattan_larger_g'):
                rows=[r for r in map_rows if (r['density'],r['reachable'],r['strategy'])
                      ==(density,reachable,strategy)]
                if rows:
                    map_summary.append(dict(density=density,reachable=reachable,
                        strategy=strategy,n=len(rows),
                        median_expanded=median(r['expanded'] for r in rows),
                        expanded_q1=quantiles([r['expanded'] for r in rows],method='inclusive')[0],
                        expanded_q3=quantiles([r['expanded'] for r in rows],method='inclusive')[2],
                        median_us=median(r['median_us'] for r in rows)))

    # Same full ternary tree, different visiting directions; no oracle ordering.
    tree_rows=[]
    for seed in range(20):
        rng=random.Random(22000+seed)
        def build(left):
            return (TreeNode(rng.randint(-20,20)) if left==0 else
                    TreeNode(children=[build(left-1) for _ in range(3)]))
        root=build(4)
        plain=tree_search(root,False)
        forward=tree_search(root,True,False)
        reverse=tree_search(root,True,True)
        assert plain.best_value==forward.best_value==reverse.best_value
        tree_rows.append(dict(seed=22000+seed,value=plain.best_value,
            minimax=plain.visited,forward=forward.visited,reverse=reverse.visited))
    tree_summary=dict(n=20,branching=3,depth=4,
        median_minimax=median(r['minimax'] for r in tree_rows),
        median_forward=median(r['forward'] for r in tree_rows),
        median_reverse=median(r['reverse'] for r in tree_rows),
        reverse_better=sum(r['reverse']<r['forward'] for r in tree_rows),
        equal=sum(r['reverse']==r['forward'] for r in tree_rows),
        reverse_worse=sum(r['reverse']>r['forward'] for r in tree_rows))

    # Policy probe: one greedy FIRST move; optimal play by both thereafter.
    # Do not label this as a full match between two greedy policies.
    distances=distance_matrix()
    policy_rows=[]
    for seed in range(40):
        rng=random.Random(32000+seed)
        values={t:rng.randint(2,15) for t in VALUES}
        optimal=treasure_search(distances,values,4,False)
        target=min(values,key=lambda t: (-(values[t]-distances['A'][t]),t))
        state=TreasureState(a_position=target,
            remaining=tuple(t for t in values if t!=target),
            score_a=values[target]-distances['A'][target],player='B')
        fixed=treasure_search(distances,values,3,False,initial=state)
        loss=optimal.best_value-fixed.best_value
        assert loss>=0
        policy_rows.append(dict(seed=32000+seed,values=values,
            greedy_target=target,optimal_target=optimal.best_target,
            optimal_value=optimal.best_value,greedy_first_value=fixed.best_value,
            loss=loss))
    policy_summary=dict(n=40,positive_loss=sum(r['loss']>0 for r in policy_rows),
        zero_loss=sum(r['loss']==0 for r in policy_rows),
        mean_loss=sum(r['loss'] for r in policy_rows)/40,
        max_loss=max(r['loss'] for r in policy_rows))
    # Fewest steps is not minimum cost: the short path crosses expensive terrain.
    weighted=[[1,9,1],[1,1,1]]
    bfs=bfs_unit_grid(weighted,(0,0),(0,2))
    optimal=astar(weighted,(0,0),(0,2))
    assert bfs['steps']==2 and bfs['cost']==10 and optimal.cost==4
    data=dict(protocol=dict(map_size=[20,20],densities=[0,0.15,0.30],
        map_seeds=list(range(12000,12020)),tree_seeds=list(range(22000,22020)),
        policy_seeds=list(range(32000,32040)),time_repeats=7,
        timing='external whole-call median after one warm-up; input creation excluded'),
        maps=map_rows,map_summary=map_summary,trees=tree_rows,tree_summary=tree_summary,
        policies=policy_rows,policy_summary=policy_summary,
        weighted_counterexample=dict(grid=weighted,bfs=bfs,
                                      astar=asdict(optimal)))
    output=Path(output_dir) if output_dir is not None else Path('./results')
    output.mkdir(parents=True,exist_ok=True)
    (output/'research.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,rows in (('map_ablation',map_rows),('tree_order',tree_rows),('greedy_first',policy_rows)):
        with (output/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print('Controlled research saved to:',output)
    print('Trees:',tree_summary)
    print('Greedy-first:',policy_summary)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "baseline", "probes", "research"), default="all",
                        help="all: 全部实验；baseline: 四题基准；probes: 附加探索；research: 控制变量研究")
    parser.add_argument("--repeats", type=int, default=101, help="基准计时重复次数")
    parser.add_argument("--output-dir", type=Path,
                        help="输出到指定目录，便于验证时保留原始报告数据")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.mode in ("all", "baseline"):
        run_baseline(args.repeats, args.output_dir)
    if args.mode in ("all", "probes"):
        run_probes(args.output_dir)
    if args.mode in ("all", "research"):
        run_research(args.output_dir)


if __name__ == "__main__":
    main()
