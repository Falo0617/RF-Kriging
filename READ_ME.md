# RF-Kriging
### 数据集的载入：

若使用txt类型文档（如中国气象数据网下载的数据集），把数据集放入 `Guangzhou_data` 文件夹并打开其中的 `data_combine.py` ，修改最下方的相关文件名，运行并生成parquet文件。随后打开主文件夹中的 `run_prepare_guangdong.py` 运行并得到 `guangdong_merged_cache.parquet` 作为预处理过的parquet文件。

若使用csv类型文件（如gsod下载的数据集），打开主文件夹中的run_prepare_gsod.py，修改  `DEFAULT_INPUT_DIR = `后的具体数据集文件夹名称，运行并生成预处理好的parquet文件。

预处理parquet文件之后打开主文件夹中的 `run_kriging.py` ，修改 `p.add_argument('--cache', type=str, default=` 后的路径为你预处理好的parquet文件，随后便可运行。

------------------

### 具体参数的修改

在 `run_kriging.py` 中的 `def parse_args():` 函数规定了相关的参数，

| 参数名           | 含义                                                       | 默认值                                |
| ---------------- | ---------------------------------------------------------- | ------------------------------------- |
| `--cache`        | 预处理后的 Parquet 缓存文件路径（支持完整或相对路径）      | `'gsod_merged_cache.parquet'`         |
| `--cache-only`   | 仅从缓存读取数据，若文件不存在则直接退出（不进行后续建模） | `True`                                |
| `--lat`          | 目标预测点的纬度（十进制度数）                             | `30.66`                               |
| `--lon`          | 目标预测点的经度（十进制度数）                             | `104.06`                              |
| `--radius`       | 搜索气象站点的半径范围（单位：公里）                       | `1000.0`                              |
| `--max-stations` | 半径内最多选取的气象站数量                                 | `100`                                 |
| `--grid-res`     | 克里金插值使用的网格分辨率（距离单位）                     | `10`                                  |
| `--n-splits`     | 数据子集划分数量（用于交叉验证或训练分割）                 | `50`                                  |
| `--out-prefix`   | 输出文件前缀（用于保存结果）                               | `'rfok'`                              |
| `--ok-neighbors` | 普通克里金（OK）留一法交叉验证时使用的最近邻站点数 `k`     | `10`                                  |
| `--ok-cache`     | OK‑LOO 结果缓存文件路径（`.npz` 格式）                     | `'ok_loo.npz'`                        |
| `--no-compare`   | 禁用 OK 与 RF‑Kriging 的结果比较（若添加此标志则比较关闭） | 未指定时比较启用（`do_compare=True`） |

--------------------------------
### 表图查看

在运行 `run_kriging.py` 后，python终端会直接输出相关结果，同时 `monitor` 文件夹会自动保存各训练轮次的具体数据并生成最终结果的相关图表。

此外，还可以运行主文件夹中的 `plot_compare_ok_rfok_per_fold.py` 生成可视化的RF-Kriging与OK的比较图表（主要是折线图和矩阵图）。还有 `plot_rf_kriging_comparison.py` 来生成RF-Kriging与OK的比较图表（主要是矩阵图）。