# GeoStudio 灵活自动建模与计算

本项目提供八变量坝体模型的参数化建模、GeoStudio 原生重网格、十阶段求解和五响应训练数据提取。建模、网格和求解三个阶段既可独立执行，也可用 `pipeline` 串联；数据提取保持为独立命令。后续阶段只读取实际 GSZ，不依赖状态文件，因此可以在任意阶段打开 GeoStudio 人工修改后继续运行。

项目包含 Initial DoE 知识边校准及完整知识引导贝叶斯优化循环：可行 LHS、逐响应留一验证、局部虚拟导数符号约束、真实计算与断点恢复。

## 环境

- Windows、GeoStudio 2025.2.0 或更高版本；
- SIGMA/W 许可证；
- Python `>=3.12.1,<3.13`；
- `uv`。

```powershell
uv sync
.\scripts\install_geostudio_api.ps1
```

第二个命令严格安装本机 GeoStudio `API\requirements.txt` 指向的官方 `gsi` wheel，不从 PyPI 猜测版本。

## CSV 输入

CSV 使用英文表头：

| 字段 | 单位/范围 |
|---|---|
| `design_id` | Windows 合法文件名且在 CSV 中唯一 |
| `upstream_slope` | H/V，1.3–1.7 |
| `downstream_slope` | H/V，1.3–1.7 |
| `crest_width` | m，6–12 |
| `modulus_zone_height` | m，5–20 |
| `secondary_top_width` | m，10–30 |
| `secondary_height` | m，20–40 |
| `secondary_bottom_width` | m，25–70 |
| `secondary_right_slope_distance` | m，2–15；右腰到下游接触参考线的最短垂直距离 |

次堆石区右腰强制与下游坝坡平行。参考线通过次堆石区底高程处的真实下游接触点，并采用当前下游坡比；右腰再按给定垂直距离向坝内平移。左下角角度不再是 CSV 设计变量，而是由上下宽度、高度和下游坡比反算；派生角度仍必须严格位于 45°–150°。该角度、参考点及垂直距离对应的水平偏移会写入 `parameters.json` 和 `build_report.json`。

上游垫层、过渡层的分界线按模板上部坡线直延至坝基，自动消除坡脚向坝内折返的尖角；同材料、同施工阶段的垫层分区会合并。坝基共享点同步调整，保留层厚、外轮廓及边界条件引用。该修正在建模时应用，不覆盖原始模板 XML。

上游风化层（材料 10、11）及其坝基共享顶点整体跟随上游坡面变形，避免高坡比下部分顶点受距离阈值限制而静止、造成分区自交。风化层面积及同高程水平厚度保持不变，远端坝基边界固定。

各变量分别合法并不保证组合几何一定可行。派生角度超限，或耦合后发生越界、重叠、短边、碎片时，该方案会报告建模失败，批次默认继续处理下一方案。

旧 CSV 需要显式迁移：删除 `secondary_left_bottom_angle`，并将旧水平净距换算为新垂直距离：

```text
secondary_right_slope_distance =
    secondary_bottom_clearance / sqrt(1 + downstream_slope^2)
```

例如下游坡比为 1.5、旧水平净距为 30 m 时，新垂直距离为约 16.641 m。程序会拒绝旧 CSV 并给出迁移提示，避免静默改变已有实验含义；旧 GSZ 模板旁的 `parameters.json` 则会在读取时自动完成相同换算。

## 20 组初始 LHS 试验

仅开展初始试验时，依次使用 `sample`、`pipeline`、`extract`，无需设置造价或响应限值，也不会开始自适应优化。

```powershell
uv run python -m geostudio_automation sample `
  --template templates/geo_studio_model_sample `
  --count 20 --seed 2026 `
  --output runs/initial_doe/designs.csv

uv run python -m geostudio_automation pipeline `
  --template templates/geo_studio_model_sample `
  --designs runs/initial_doe/designs.csv `
  --output runs/initial_doe/models

uv run python -m geostudio_automation extract `
  --project-dir runs/initial_doe/models `
  --glob "*/*.gsz" `
  --cvar-alpha 0.995 `
  --output runs/initial_doe/extracted
```

`sample` 保留每个变量的 20 个等宽分层各一个点，通过层内重采样和维度配对交换修复不可行组合；所有样本必须通过几何及 XML 预检才会写出 CSV。它不调用 GeoStudio，也不缩小变量范围。`designs.sampling.json` 保存种子、实际范围、模板 XML 哈希、归一化样本和检查状态；已有输出不会被覆盖。

几何预检不保证求解收敛。计算后检查 `models/manifest.csv` 和 `extracted/extraction_report.json`，最终 `dataset.csv` 应有 20 行成功数据。独立流水线不会自动补点；失败方案应先排查、重算，必要时在原 LHS 分层内替换并重新验证，不能简单删除后宣称仍有完整的 20 点 LHS。

优化器只将三个位移响应作为约束，原始 `extract` 仍保留两项应力结果供审计；应力不参与优化代理模型、知识校准或可行性判定。

## 独立阶段

建模整个 CSV：

```powershell
uv run python -m geostudio_automation build `
  --template templates/geo_studio_model_sample `
  --designs examples/designs.csv `
  --output runs
```

只建模指定方案；`--design-id` 可重复：

```powershell
uv run python -m geostudio_automation build `
  --template templates/geo_studio_model_sample `
  --designs examples/designs.csv `
  --output runs `
  --design-id baseline_feasible
```

对人工修改过的项目直接重网格，不需要 `build_report.json`：

```powershell
uv run python -m geostudio_automation mesh `
  --project .\manual\edited.gsz
```

批量发现目录中的项目。目录默认只匹配当前层 `*.gsz`；下面的模式显式匹配一层方案目录：

```powershell
uv run python -m geostudio_automation mesh `
  --project-dir runs `
  --glob "*/*.gsz"
```

求解只使用 GSZ 当前网格。XML/PLY 校验不通过时会拒绝求解，不会隐式重新建模或重网格：

```powershell
uv run python -m geostudio_automation solve `
  --project runs/baseline_feasible/baseline_feasible.gsz
```

从已求解方案批量提取三个变形响应和两个面积加权主应力 CVaR 响应：

```powershell
uv run python -m geostudio_automation extract `
  --project-dir runs `
  --glob "*/*.gsz" `
  --output runs/extracted
```

默认查询“正常蓄水位应力分布”的最终保存结果步号 `3`，并计算数值最高 0.5% 物理面积的 CVaR（`--cvar-alpha 0.995`，输出列名包含 `cvar99p5`）。当前模型中，步号 `0` 是该分析开始时继承自上一分析的初始状态，步号 `1` 和 `2` 未保存，步号 `3` 才是正常蓄水位计算完成后的结果。可用 `--step` 显式选择其他已保存的非负结果步号。每个 GSZ 必须有同目录 `parameters.json` 和状态为 `solved`、且不早于 GSZ 的求解报告。若修改过模型或材料，需要重新求解后再提取。

使用 Initial DoE 数据对静态知识边执行单层局部 ALE 校准：

```powershell
uv run python -m geostudio_automation calibrate-kg `
  --dataset runs/extracted/dataset.csv `
  --output runs/kg_calibration
```

校准对五个响应各拟合一次 GP，不重复删除样本或重拟合。只有局部方向与专家先验一致且影响强度达到阈值的边才会激活；稳定反向边会熔断但不会翻转专家方向。已有输出需要显式传 `--force` 才能替换。

`mesh` 和 `solve` 默认原位更新。单文件可用 `--output-project copy.gsz`，多文件可用 `--output-dir copied`；目标已存在时还需显式传 `--overwrite-output`。

## 便捷流水线

完整执行建模、重网格和十阶段求解：

```powershell
uv run python -m geostudio_automation pipeline `
  --template templates/geo_studio_model_sample `
  --designs examples/designs.csv `
  --output runs
```

只执行建模和网格：

```powershell
uv run python -m geostudio_automation pipeline `
  --template templates/geo_studio_model_sample `
  --designs examples/designs.csv `
  --output runs `
  --stop-after mesh
```

已有 GSZ 默认不会被建模阶段覆盖；明确重建时使用 `--force`。批次默认隔离单方案失败，使用 `--fail-fast` 可在首次失败时停止。

## 输出

生成方案位于 `runs/<design_id>/`：

- `<design_id>.gsz`：当前工作项目，可在 GeoStudio 中人工修改；
- `parameters.json`：输入参数、哈希、派生坐标和软件版本；
- `build_report.json`、`mesh_report.json`、`solve_report.json`：相互独立的阶段报告；
- `build_report.log`、`mesh_report.log`、`solve_report.log`：阶段日志。

`extract --output <目录>` 另写：

- `dataset.csv`：成功方案的设计变量和五个响应；
- `extraction_audit.csv`：各方案的网格规模、有效面积、VaR、原始极值和失败原因；
- `extraction_report.json`：批次配置和成功/失败汇总；
- `per_run/<design_id>/response_metrics.json`：逐方案完整提取记录和缓存。

`calibrate-kg --output <目录>` 另写：

- `kg_weights.json`：静态先验、连续有效权重和激活掩码；
- `kg_edge_calibration.csv`：40 条知识边的方向、强度、状态和审计指标；
- `calibration_report.json`：代理拟合状态、激活统计和警告。

几何预览保存在 `runs/model_picture/<design_id>.png`，`pipeline` 另写 `runs/manifest.csv`。任意目录中的 GSZ 使用独立命令时，报告名会带项目文件名前缀，避免同目录多项目相互覆盖。

命令退出码：全部成功为 `0`，至少一个方案失败为 `1`，命令参数、输入或环境预检失败为 `2`。

## 知识引导贝叶斯优化

`examples/optimization.json` 已配置三个位移约束：坝顶最大沉降 `<0.080 m`、坝体最大沉降 `<0.16 m`、坝体最大水平位移 `<0.075 m`。`initial_dataset` 指向已完成的 20 组试算数据，优化开始时复用真实结果，只对后续候选启动 GeoStudio。运行前须核实 `material_unit_costs` 是否是本工程的实际单价。配置中的路径相对于配置文件所在目录解析。

`material_unit_costs` 的键是模板材料 ID，值是元/m³单价；**只有列出的材料计入造价**，未列出的材料不计价。价格必须为正数，排除某种材料时不要填零，直接不列该材料。

填写格式是一个 JSON 对象：`"材料ID": 单位体积价格`。ID 必须用英文双引号包围，不能用材料名称或设计变量序号代替；价格填写正数，可以有小数，不要附加 `元/m³` 等单位文字。各项用英文逗号分隔，最后一项后不要加逗号；JSON 不支持 `//` 注释。

例如，若造价目标只计入主堆石区、次堆石区、排水和坝体增模区，价格填写格式如下。**下列价格仅用于展示格式，必须核实并改成实际单价。**

```json
{
  "material_unit_costs": {
    "4": 100.0,
    "5": 70.0,
    "6": 120.0,
    "12": 150.0
  }
}
```

例如 `"4": 100.0` 表示材料 ID 4（主堆石区）的单位体积价格为 100 元/m³。需要计入面板、垫层或其他材料时，按下表添加对应 ID 和真实单价。程序不会自动补齐未列出的材料价格；未列出的材料即使随设计变化，也不会贡献造价。

| ID | 材料 | ID | 材料 |
|---|---|---|---|
| 1 | 面板 | 2 | 垫层 |
| 3 | 过渡 | 4 | 主堆石区 |
| 5 | 次堆石区 | 6 | 排水 |
| 7 | 压坡 | 8 | 帷幕 |
| 9 | 地基 | 10 | 全强风化 |
| 11 | 全风化 | 12 | 坝体增模区 |
| 13 | 坝顶 | | |

造价由当前重构几何的实际分区面积直接求和：`C(x) = Σ A_material(x) × unit_cost_material`，单位为元/米坝长。材料字典中存在但当前几何未使用的材料面积为零。三个位移约束沿用 `extract` 的列名和米单位；恰好等于限值时判为不合格。

填写参数后运行：

```powershell
uv run python -m geostudio_automation optimize `
  --config examples/optimization.json `
  --output runs/knowledge_bo
```

恢复同一个实验：

```powershell
uv run python -m geostudio_automation optimize `
  --config examples/optimization.json `
  --output runs/knowledge_bo --resume
```

恢复会核对配置、初始数据文件哈希、模板内容、变量范围、专家先验矩阵、模式和评估器指纹，以及已有成功结果的文件哈希。改变参数、模式或修改结果文件时，应另用输出目录。中断前完成的有效评估会被复用；未完成评估若已开始网格阶段，其预算不会退还。输出目录有进程锁，同一实验不会被两个进程同时写入。

使用 `--mode plain` 关闭知识校准和导数注入，保留相同的采样、精度验证、预算和采集准则。配置含 initial_dataset 时，20 个初始结果从已有数据导入并计入总预算；不同模式使用独立输出目录。本次没有自动启动额外真实对照组。

### 先运行模拟验证

以下命令使用真实模板的几何、分区造价和优化代码，但三个位移响应来自确定性合成函数，**不调用 GeoStudio 求解**：

```powershell
uv run python -m geostudio_automation optimize `
  --config examples/optimization.simulation.json `
  --output runs/bo_simulation --simulate
```

该示例的单价和限值只用于软件验证，不是工程建议。它设有 `simulation_only: true`，不带 `--simulate` 时拒绝运行。为了快速检查，示例执行 20 个初始样本和 3 次自适应评估，使用 64 个候选、零次额外 GP 超参数重启；正式配置默认执行 20+40 次，使用 512 个候选、3 次额外重启。

### 算法与预算

1. 配置 initial_dataset 时先验证参数哈希、几何、XML 和逐维拉丁分层并导入初始结果；否则在八维边界内生成 LHS。通过维度配对交换和层内重采样修复几何，保持每维每层恰好一点。默认最多 10 次初始化、每次 2,000 次修复；失败明确退出，不静默缩小设计空间。实际计算前再次验证 XML。失败初始样本在原来的八维分层单元内补点。
2. 三个位移响应分别使用 `Constant × Matérn(ν=2.5) + WhiteKernel` GP。输入按固定设计边界归一化；每次训练单独计算响应均值和标准差。留一法逐折重新拟合标准化参数和核超参数，合并全部留一预测计算 R²。正式示例配置将知识门槛设为 R²≥0.75，各响应单独启停；常量响应及无效 R² 不启用知识。
3. 仅使用纯数据 GP 校准现有 ALE 知识边。每次新样本后重新计算；反向、弱效应或不稳定边熔断，不翻转专家方向。没有响应通过精度门控时，先补充高不确定性样本。
4. 每轮先筛选 512 个几何可行、未评估的 Sobol 候选，直接计算造价。对已激活知识边，每个响应保留权重最大的两个变量，用纯数据 GP 计算候选处导数违反先验方向的概率 `q=P(sign × derivative<0 | 真实数据)`。候选需处于已有样本的经验覆盖范围，且位于预测约束边界 2 个标准差内，或属于纯数据采集得分最高的 25%。优先选择 `0.1≤q≤0.5` 的位置；每选一处，利用当前软符号后验重新计算其余候选的导数概率，并避开同一变量 0.15 归一化距离内的重复点。每个响应最多 20 条；`q>0.5` 的区域仅记录方向分歧，达到 `local_conflict_probability` 时标为局部冲突，不强行注入。上述位置阈值是本实验的预设规则，不是论文通用阈值。
5. 在所选位置注入 `Phi(sign × derivative / softness)^weight` 的软符号似然；导数按先验导数标准差缩放，`softness` 默认 0.1。权重不是导数数值。条件导数后验通过白化 Laplace 近似推断，核超参数不受虚拟观测重新拟合。数值失败会记录并退回纯数据预测。没有真实可行解时最大化联合可行概率；已有可行解时最大化 `(best_feasible_cost - cost)+ × P(feasible)`。响应 GP 相互独立，可行概率在对数域累计。每第 5 次自适应评估以及所有改进得分为零时，选择纯数据模型的不确定性探索点；完全常量时以距已有样本的最小距离填充空间。

可先对已有检查点离线重放一个选点决策，**不会启动 GeoStudio 求解**：

```powershell
uv run python -m scripts.replay_virtual_derivatives `
  --config examples/optimization.json `
  --checkpoint runs/knowledge_bo_displacement/checkpoint.json `
  --samples 20 --output runs/virtual_replay_20
```

输出 `replay_summary.json` 和 `iterations/samples_0020/virtual_derivatives.json`，可检查虚拟点、每次选择时的 `q`、方向分歧与下一推荐方案。新的放点逻辑使用新的检查点版本；既有优化实验请保留原目录，新实验使用新输出目录。

留一验证衡量当前样本上的泛化表现，不等同于整个可行域的精度证明。算法不会宣称找到全局最优解或保证知识必定加速收敛；`plain` 模式和统一日志用于后续检验。

默认总预算为 60 次昂贵评估：已有的 20 个有效初始样本计入预算，再最多进行 40 次自适应评估。每次开始原生网格前先持久化扣减；网格、求解或提取失败不会返还预算。初始失败重试会减少剩余自适应次数。纯几何/XML 检查不消耗昂贵预算，另受有界补样次数限制。一个“评估”包含现有十阶段求解，并非只指单个分析阶段。

### 优化输出与程序接口

- `resolved_config.json`、`checkpoint.json`：冻结配置、指纹、初始分层与逐次状态。
- `initial_designs.csv`：最初修复后的 LHS；`initial_designs_final.csv`：替换失败点后真正成功的初始设计。
- `dataset.csv`：成功评估的八变量、三个位移响应与 `cost_per_m`；优化内部的知识校准只读取这三列。
- `optimization_history.csv`：各次评估的阶段、预算、失败原因与最佳可行造价。
- `iterations/samples_NNNN/`：逐响应留一结果、预测 CSV、校准矩阵与边审计、逐次尝试的 `virtual_derivatives_XXXX.json`、最新虚拟导数状态及候选选择记录。
- `runs/eval_NNNN/`：逐设计原有阶段报告、提取文件与带结果文件哈希的 `evaluation.json`。
- `optimization_report.json`、`best_design.json`、`convergence.png`：结果汇总和曲线。没有真实可行解时最佳设计为空，另列标准化违约量最小的已评估设计，不把代理预测当作可行解。

`optimize` 在预算正常结束且存在真实可行解时返回 `0`，未找到可行解或采样受阻返回 `1`，输入、环境或恢复校验错误返回 `2`。

程序入口为 `ExperimentConfig.from_json(...)` 和 `run_optimization(config, output_directory, ...)`。后者允许传入 `Evaluator` 实现：提供稳定的 `fingerprint`、`evaluate(design, directory, charge)` 和 `recover(design, directory)`；昂贵操作前必须调用一次 `charge()`，成功结果必须包含三个有限位移响应值。`GeoStudioEvaluator` 是正式后端，`SimulatedEvaluator` 是示例后端。`calibrate_kg_edges` 新增可选的 `response_models`、`response_enabled`，复用纯数据拟合并校验其训练集，原命令和默认行为保持兼容。

方法参考：[虚拟导数符号观测与迭代选位](https://proceedings.mlr.press/v9/riihimaki10a.html)；[约束贝叶斯优化](https://proceedings.mlr.press/v32/gardner14.html)。本实现使用 Laplace 近似，前者论文使用 EP，二者推断方法不同。

## 测试

```powershell
uv run python -m unittest discover -s tests -v
```

真实 GeoStudio 集成测试默认跳过：

```powershell
$env:GEOSTUDIO_INTEGRATION = "1"
uv run python -m unittest discover -s tests -p "test_integration.py" -v
```
