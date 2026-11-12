# 实验二：搜索算法

实现A*、Minimax与Alpha-beta，完成四道搜索算法题目。

`./results/`保存原始实验数据。验证复现时应该使用`--output-dir ./results_check`，避免覆盖原始计时。

## 1. 环境要求

Python 3.10 或更高版本。算法实验和自动测试仅使用 Python 标准库，无需额外安装依赖。

## 2. 快速运行

将本目录内容作为代码仓库根目录，在仓库根目录执行以下命令。仓库根目录应包含`code`和`results`文件夹。

```powershell
python -m unittest discover -s ./code -v
python ./code/run_experiments.py --output-dir ./results_check
```

第一条命令运行七组正确性测试；第二条命令运行全部实验，将复现结果写入`./results_check/`。

主要预期结果：

| 题目 | 预期结果 |
|---|---|
| 迷宫寻路 | 最短路径为 7 次移动，包含起终点共 8 个坐标 |
| 加权地形 | 最小总代价为 7 |
| 极大极小博弈树 | 根节点最佳得分为 5 |
| 宝藏分配，H=2 | 最优首步 T2，收益差 −1 |
| 宝藏分配，H=4 | 最优首步 T3，收益差 3 |

迷宫允许返回任意一条同长度最短路径。随机地图、随机博弈树及边界场景也会在测试中核验。

## 3. 实验命令与参数

### 3.1 选择实验范围

```powershell
# 四题基准与拓展实验
python ./code/run_experiments.py

# 四题基准、启发式与访问顺序对照
python ./code/run_experiments.py --mode baseline

# 共享根窗口、深度决策、最优行动序列和多宝藏缓存实验
python ./code/run_experiments.py --mode probes

# 控制变量研究：多地图、树访问顺序、贪心首步与 BFS 基线
python ./code/run_experiments.py --mode research
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | `all` | `all` 执行全部实验；`baseline` 执行基准；`probes` 执行附加探索；`research` 执行控制变量研究 |
| `--repeats` | `101` | 基准计时重复次数，必须为正整数；预热一次后取中位数 |
| `--output-dir` | `./results` | 数据输出目录；显式指定时，自动结果记录也写入该目录 |

调整计时次数或单独保存一次运行结果：

```powershell
python ./code/run_experiments.py --repeats 501
python ./code/run_experiments.py --output-dir ./results_check
python ./code/run_experiments.py --help
```

默认运行会更新`./results/`中的数据并生成`./实验结果记录.md`；显式指定输出目录时，该记录也写入指定目录。`probes`只记录节点统计；基准计时为内部预热后101次中位数，`research`为完整调用预热后七次中位数。计时均排除文件输出和输入生成；基准还排除距离矩阵预处理及地图字符适配。不同口径分别解释。

`research` 额外生成 `./results/research.json`、`map_ablation.csv`、`tree_order.csv` 和 `greedy_first.csv`。研究协议固定随机种子、地图尺寸、起终点和树结构；地图按可达与无解分组，时间为完整算法调用的中位数。该组实验不改变四道题的标准答案。

## 4. 文件说明

| 文件或目录 | 作用 |
|---|---|
| `./code/run_experiments.py` | A*、树搜索、宝藏博弈、题目输入、基准与拓展实验的实现及命令行入口 |
| `./code/test_algorithms.py` | 样例、边界、随机地图、随机博弈树及穷举对照测试 |
| `./results/results.json` | 环境信息、路径、距离矩阵、节点统计、基准计时及首步价值 |
| `./results/astar.csv` | A* 对照实验数据 |
| `./results/tree.csv` | 博弈树搜索对照数据 |
| `./results/treasure.csv` | 宝藏分配基准实验数据 |
| `./results/opportunity_probe.json` | 根窗口共享、深度对照、行动序列及缓存拓展数据 |
| `./results/research.json` | 多地图启发式消融、树访问顺序和贪心首步研究结果 |
| `./results/map_ablation.csv` | 每张地图五种策略的代价、扩展数与完整调用时间 |
| `./results/tree_order.csv` | 每棵树的 Minimax 与正反序 Alpha-beta 配对结果 |
| `./results/greedy_first.csv` | 每组宝藏价值的最优首步、贪心首步和价值损失 |

## 5. 算法接口与规则

在 `code` 目录启动 Python 后，可直接调用算法：

```python
from run_experiments import MAZE, maze_search

result = maze_search((0, 0), (3, 4), MAZE)
print(result.path)
print(result.cost, result.expanded)
```

| 接口 | 功能 |
|---|---|
| `astar` | 在非负数值代价网格上寻路，`None` 表示障碍 |
| `maze_path` / `maze_search` | 返回迷宫最短路径 / 含统计信息的结果 |
| `weighted_cost` / `weighted_search` | 返回加权最小代价 / 含统计信息的结果 |
| `alpha_beta` / `tree_search` | 返回树的最佳得分 / 含统计信息的搜索结果 |
| `distance_matrix` | 计算题目四中六个关键位置的距离矩阵 |
| `treasure_search` | 按深度、访问顺序及剪枝或缓存设置求解宝藏博弈 |

A* 起点不收费，每步支付进入后继格的权值。题目二的 E 进入代价为 0，题目四的 A、B 和宝藏格进入代价均为 1。无解时路径为空、代价为 −1。

宝藏搜索的 H 是单方行动次数，评价值为双方累计收益差。共享根窗口为单独的拓展策略，其结果与主实验分别记录；精确缓存用于普通 Minimax，不存储 Alpha-beta 截断产生的界值。

研究中的 BFS 只作为单位代价地图的基线，不用于加权地形的最优代价求解；加权反例专门说明按步数最短不等于按代价最优。
