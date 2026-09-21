# SigmaX 最优控制问题梯度法求解工具箱

一个自研的**直接法梯度求解器**，用于求解时间最优的含 $\sigma_x$ 驱动的量子比特重置问题（Lindblad 主方程 + 庞特里亚金极值原理）。它把「**待求的最优控制问题**」抽象成一个接口，把「**求解算法**」做成通用可复用组件——同一个求解器无需任何修改即可求解另一个满足该接口的问题。

> **分工**：本文件是**实操指南**（怎么装、怎么跑、怎么改）；算法与物理的原理、推导见 [`descriptor/main.tex`](descriptor/main.tex)（用 `xelatex` 编译成 PDF）。

---

## 1. 环境要求

- **MATLAB R2024a 及以上**（用到 `trapz`、`getframe`、`VideoWriter`、类/包）。无第三方依赖。
- 可选 **CasADi**：仅当你想用 `_source_original/code_Gradient_AD` 里的**牛顿法（二阶）**版本时才需要；梯度工具箱本身**不需要** CasADi。
- 所有 `.m` 文件注释为**英文/ASCII**，在任何系统语言/编码的 MATLAB 下都能直接运行（中文说明集中在 descriptor 与 README）。

---

## 2. 快速开始

把 `src` 加入 MATLAB 路径，然后：

```matlab
addpath('...\SigmaX_OCP\src');          % 只加 src（它是 +ocp 等包的父目录）

% 1) 建模型（要解决的 OCP）
model = ocp.QubitReset('tau', 0.15, 'N', 100, 'gamma', 10, 'lambdaMax', [3; 2]);

% 2) 配置求解器
s = solver.GradientSolver(model);
s.direction  = 'gradient';              % 方向优化方法
s.lineSearch = 'armijo';                % 步长准则
s.maxIter    = 2000; s.tolGrad = 1e-5;

% 3) 求解
res = s.solve();
fprintf('J = %.6e, converged = %d (%s)\n', res.J, res.converged, res.reason);
```

也可直接运行示例脚本：

```matlab
run('demo/demo_gradient.m')     % 梯度法求解 + 结果绘图
run('demo/demo_prolong.m')      % 同伦延拓求解（tau 或 kappa）
run('demo/demo_animation.m')    % 边求解边录制动图 GIF/MP4
run('demo/demo_example1.m')     % 换一个完全不同的可替换问题
```

---

## 3. 项目结构

```
SigmaX_OCP/
  src/
    +ocp/           AbstractOCP.m      抽象接口（可替换问题的契约）
                    QubitReset.m       σx 量子比特重置问题（核心）
                    ControlExample.m   另一个可替换问题（descriptor 例1）
    +solver/        GradientSolver.m   直接法梯度求解器
                    LineSearch.m       非精确步长搜索
    +vis/           Visualizer.m       迭代四联图 + 动画（GIF/MP4）
    +continuation/  HomotopyContinuation.m  同伦延拓外壳
  demo/             demo_*.m           4 个示例脚本
  tests/            validate_suite.m   自检
  descriptor/       main.tex/main.pdf  原理文档（xelatex 编译）
```

只需要把 `src`（它下面才是 `+ocp` 等包）加入路径。**不要**把 `src/+ocp` 之类的包目录本身加到路径，那样会破坏包的命名。

---

## 4. 问题接口（`ocp.AbstractOCP`）

任何最优控制问题只需继承 `AbstractOCP` 并实现下列方法，即可被同一求解器求解：

| 方法 | 含义 |
|---|---|
| `timeGrid(N)` | 时间格点 `1×N`（`[0,tau]`） |
| `initialState()` | 初态 `nState×1` |
| `initialControl(tg)` | 初始控制的猜测 `nControl×N` |
| `costGradient(tg,U)` | 成本 `J`、状态 `X`、协态 `Mu`、梯度 `G` |
| `project(U)` | 把控制投影到可行域 |
| `controlBounds()` | 控制上下界 `[Ulo,Uhi]` |

属性：`nState`、`nControl`、`name`、`tau`、`homotopyValue`、`homotopyName`。

> **可替换性验证**：`ocp.ControlExample`（descriptor 例1）是一个标量状态/控制、带运行成本、与量子比特完全无关的问题，但 `solver.GradientSolver` **零修改**即可求解它。`demo_example1.m` 演示了这一点。

---

## 5. 求解器参数（`solver.GradientSolver`）

```matlab
s = solver.GradientSolver(model);
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `direction` | `'gradient'` | 方向优化：`'gradient'`/`'adagrad'`/`'rmsprop'`/`'momentum'`(经典动量)/`'nestorov'`/`'adam'` |
| `lineSearch` | `'armijo'` | 步长准则：`'simple'`/`'armijo'`/`'wolfe'` |
| `interpolation` | `'bisection'` | 步长折半方式 |
| `useProjection` | `true` | 每步把控制投影回可行域（比罚函数稳定） |
| `kktActiveSet` | `false` | 投影梯度/KKT 活动集：在到达边界的控制分量处把向外推的梯度置零 |
| `maxIter` | `1000` | 最大迭代 |
| `minIter` | `1` | 最小迭代 |
| `tolGrad` | `1e-4` | 梯度范数收敛阈值 |
| `tolRel` | `1e-6` | 成本相对变化阈值 |
| `beta1`/`beta2` | `0.9`/`0.999` | 动量/二阶矩衰减（momentum/adam 用） |
| `rho`/`sigma` | `0.1`/`0.4` | Armijo/Wolfe 参数 |
| `alpha0`/`alphaMin`/`alphaMax` | `0.5`/`1e-8`/`1` | 初始/最小/最大步长 |
| `N` | `100` | 时间离散点数（默认网格） |
| `observer` | 空 | 可选回调 `@(iter,J,X,Mu,U,tg) ...`，每轮迭代调用（用于可视化） |
| `verbosity` | `1` | 打印进度 |

### 5.1 求解结果 `res`

```matlab
res.U            % nControl×N  最优控制
res.J            % 最优成本
res.X, res.Mu    % nState×N    末次状态/协态
res.G            % 最优控制处的梯度
res.hist.J       % 每轮成本
res.hist.gnorm   % 每轮梯度范数
res.converged    % 是否收敛
res.reason       % 收敛原因
res.iterations   % 实际迭代数
```

### 5.2 选择方向优化（跨越鞍点/高原区）

纯梯度法收敛慢，在高原区易停滞。历史累计方向能显著改善：

```matlab
s.direction = 'adam';           % 收敛最快（见第 8 节实测）
s.direction = 'momentum';       % 经典动量
s.direction = 'nestorov';       % Nesterov 加速
s.direction = 'adagrad';        % 自适应当量
s.direction = 'rmsprop';
```

---

## 6. 同伦延拓（prolong）单独使用

`continuation.HomotopyContinuation` 是**独立于具体问题**的外壳：它沿一条参数路径延拓求解，每一步以上一步的解作为热启动，从而解决**初值敏感 / 梯度消失**。两种延拓族由 `model.homotopyName` 选择：

```matlab
% 方式一：时域延拓（tau 从短到长）
model = ocp.QubitReset('tau', 1.0, 'N', 80, 'homotopyName', 'tau');
c = continuation.HomotopyContinuation(model, linspace(0.09, 1.0, 12));
c.solver.direction = 'gradient'; c.solver.maxIter = 800;

% 方式二：成本权重延拓（kappa 从 0 到 1，含正则化运行项）
model = ocp.QubitReset('tau', 0.15, 'N', 80, 'homotopyName', 'kappa');
c = continuation.HomotopyContinuation(model, linspace(0.2, 1, 8));

r = c.run();
r.results(end).U, r.results(end).J   % 最终解
r.results(k).J                        % 第 k 步的成本
```

---

## 7. 迭代可视化 / 动画

```matlab
v = vis.Visualizer('animate', true, 'format', 'gif', 'videoFile', 'iteration.gif', 'fps', 12);
tg = model.timeGrid(model.N); v.init(model, tg);
s.observer = @(iter,J,X,Mu,U,tgrid) v.snapshot(iter,J,X,Mu,U,tgrid);
res = s.solve();  v.finish();
```

- `format` 换 `'mp4'` 即导出 MP4（`videoFile='iteration.mp4'`，同原研究代码）。
- 四联图：状态、协态、控制、成本历史（对数轴）；曲线数量自动适配 `nState`/`nControl`。

---

## 8. 自检（可选）

在正常 MATLAB（带图形）中运行：

```matlab
addpath('...\SigmaX_OCP\tests');      % 让 validate_suite 可被调用
validate_suite('...\SigmaX_OCP\src')
```

会检查：①梯度是否为下降方向；②求解器成本是否下降；③Adam 方向是否更优；④同伦延拓是否可运行。

在本机 R2024a 的实测输出：

```
[1] 梯度是下降方向      : qubit=PASS example=PASS
[2] 求解器成本下降       : qubit 0.316→0.175(60步)  example 4.554→4.494
[3] Adam 方向优化        : qubit 0.316→0.053
[4] 同伦延拓运行         : kappa 4 步 J=0.052
tau 延拓                 : 3 步 J=0.073
动画                     : 生成 5 帧、1500×1000 的 GIF
```

> ⚠️ **以上是修复前的记录。** `+ocp/QubitReset.m` 的两处修正
> （`matVecTheta` 中 Gp 的几何因子、`initialState` 忽略 `pInitial`）
> 会改变第 [2] 项：qubit 的 60 步结果应从 `0.175` 变为 **`0.114`**
> （Python 移植版实测 1.1402e-01）。其余各项（example、Adam、延拓、动画）
> 预期不变。请在 R2024a 上重跑 `tests/validate_suite.m` 确认。

---

## 9. 待办 / 备注

- **`_source_original/`**：原始研究代码备份（含 CasADi/牛顿法的 `code_Gradient_AD`），只读参考，非成品。若想复用牛顿法，需自行安装 CasADi 并把 `addpath('casadi\')` 加回。
- **罚函数法**：descriptor 里作为原理讲解；实现上默认用更稳定的**投影梯度法**（`useProjection`）。
- 若要解决**另一个**最优控制问题：继承 `AbstractOCP` 即可，无需改任何求解器代码。（见 `ocp.ControlExample` 与 `demo_example1.m`。）
