# 看懂 Python：从你自己的代码开始

> 这份文档只为一个目标：**让你能独立读懂 `SigmaX_OCP_py/` 里的每一个 `.py` 文件。**
>
> 写法是「**Python 代码 → 逐行拆解 → 等价 MATLAB**」。
> 你的思维是 MATLAB 的，所以每讲一个 Python 语法点，我都会告诉你
> "这在 MATLAB 里对应什么"。
>
> 建议顺序：**从头读，不要跳**。第 1–3 节是地基。

---

# 第 0 节：先拆掉三个心理障碍

## 0.1 "Python 看着像伪代码，没有 end，我怎么知道块到哪结束？"

**缩进就是块边界。**

MATLAB 靠 `end` 关键字标记结束；Python 靠**缩进**。

```matlab
if x > 0
    y = 1;
    z = 2;
end
w = 3;
```

```python
if x > 0:
    y = 1
    z = 2
w = 3
```

规则只有一条：**同一块里的行，左边缩进必须一样多**（惯例是 4 个空格）。
缩进退回到外层，就表示"这个块结束了"。

> ⚠️ **所以读代码时，缩进不是"排版好看"，而是语法本身。**
> 看到某行往左退了一级，就知道上一个块结束了。

## 0.2 "self 是什么？为什么每行都要写？"

`self` **就是 MATLAB 的 `obj`**，只是 Python 要求你**显式写出来**。

```matlab
function printInfo(obj)
    fprintf('%s\n', obj.name);    % ← 显式用 obj
end
```

```python
def print_info(self):
    print(f"{self.name}")          # ← 显式用 self
```

**唯一的区别**：MATLAB 里 `obj` 是你自己随便起的名，Python 里**第一个参数固定是 `self`**（约定俗成，不写会看不懂）。调用时也不用传：`m.print_info()` 就够了。

## 0.3 "那些 @ 开头的东西是什么？"

`@` 叫**装饰器**，作用是"给下面的函数/类附加一些额外行为"。

你先把它当成**"这句话是在说：下面这个函数有特殊身份"**：

| 你看到的 | 它的意思 | MATLAB 里对应 |
|---|---|---|
| `@staticmethod` | 这个函数不需要 `self`，就是个普通函数 | 不依赖 obj 的局部函数 |
| `@abstractmethod` | 这个方法子类必须实现 | `methods (Abstract)` |
| `@dataclass` | 这个类只是用来装数据的，自动生成构造函数 | `struct` |
| `@property` | 这个"属性"其实是算出来的 | 依赖属性（dependent property） |

第 5 节会逐个细讲。

---

# 第 1 节：最核心的一张对照表

**先把这张表记住，80% 的代码你就能读下来了。**

## 1.1 定义与结构

| MATLAB | Python | 备注 |
|---|---|---|
| `function f(x)` | `def f(x):` | 冒号结尾 |
| `end` | *（缩进）* | 没有 end |
| `classdef C < handle` | `class C:` | |
| `classdef (Abstract) C` | `class C(ABC):` | 需要 `from abc import ABC` |
| `obj.a = 1` | `self.a = 1` | 必须显式 self |
| `obj.(name) = v` | `setattr(self, name, v)` | 动态字段名 |
| `%` 注释 | `#` 注释 | |
| `...` 续行 | `\` 或括号内自动续行 | Python 里更常用括号 |
| `;` 语句结尾 | *（换行即结尾）* | 不要写分号 |

## 1.2 数组与下标（**最容易出错的地方**）

| MATLAB | Python / numpy |
|---|---|
| `a(1)` | `a[0]` |
| `a(end)` | `a[-1]` |
| `a(2:5)` | `a[1:5]` ← **注意右端不含 5！** |
| `a(2:end)` | `a[1:]` |
| `A(:,1)` | `A[:, 0]` |
| `A(1,:)` | `A[0, :]` |
| `numel(a)` | `a.size` |
| `size(A,1)` | `A.shape[0]` |
| `zeros(3,4)` | `np.zeros((3, 4))` ← 注意双括号 |
| `zeros(1,n)` | `np.zeros(n)` |

> 🔴 **两条铁律，背下来：**
> 1. **MATLAB 从 1 开始数，Python 从 0 开始数。**
> 2. **Python 的 `a[i:j]` 包含 `i`，不包含 `j`。**
>    `a(1:5)`（MATLAB，5 个元素）↔ `a[0:5]`（Python，5 个元素）。

## 1.3 运算

| MATLAB | Python / numpy |
|---|---|
| `x .* y` | `x * y` |
| `x .^ 2` | `x ** 2` |
| `A * x`（矩阵乘） | `A @ x` ← **完全不同！** |
| `A'`（共轭转置） | `A.conj().T`；实数矩阵用 `A.T` |
| `sum(G.^2, 1)` | `np.sum(G**2, axis=0)` |
| `sqrt(x)` | `np.sqrt(x)` |
| `mod(a,b)` | `a % b` |
| `a == b`（判断） | `a == b` |

> 🔴 **最容易踩的坑**：MATLAB 的 `*` 是**矩阵乘法**，Python 的 `*` 是**逐元素乘法**。
> MATLAB 的 `A*x` 要写成 Python 的 `A @ x`。

## 1.4 求和维度（**MATLAB 和 numpy 差 1**）

| MATLAB | numpy |
|---|---|
| `sum(A, 1)`（沿第 1 维） | `np.sum(A, axis=0)` |
| `sum(A, 2)`（沿第 2 维） | `np.sum(A, axis=1)` |

**规律：numpy 的 `axis` 比 MATLAB 的 `dim` 小 1。** 因为 Python 从 0 编号。

---

# 第 2 节：真正开始逐行读代码

我们从**最简单的函数**开始，一步步加深。

## 2.1 例一：`print_info`（只有打印和一个 if）

`sigmax_ocp/ocp/abstract_ocp.py` 里：

```python
def print_info(self) -> None:
    print(f"OCP  : {self.name}")
    print(f"state dim   = {self.n_state}")
    print(f"control dim = {self.n_control}")
    print(f"horizon tau = {self.tau:.4g}")
    if self.homotopy_name:
        print(f"homotopy    = {self.homotopy_name} (value {self.homotopy_value:.4g})")
```

**逐行拆解：**

**① `def print_info(self) -> None:`**

| 部分 | 含义 |
|---|---|
| `def` | 定义函数，等于 MATLAB 的 `function` |
| `print_info` | 函数名 |
| `(self)` | 第一个参数，等于 MATLAB 的 `obj` |
| `-> None` | **返回类型注解**：说明这个函数不返回东西。纯粹是给人看的，**删掉照样能跑** |
| `:` | 函数体从这里开始（后面要缩进） |

**② 缩进 4 空格** → 这四行都属于 `print_info` 的函数体。

**③ `print(f"OCP  : {self.name}")`**

- `print(...)` = MATLAB 的 `fprintf(...)`，但**自动换行**（不用写 `\n`）。
- `f"..."` 叫 **f-string**：字符串前面加个 `f`，里面就能用 `{变量}` 直接插入值。

对照：

```matlab
fprintf('OCP  : %s\n', obj.name);
```
```python
print(f"OCP  : {self.name}")
```

**④ `f"...{self.tau:.4g}"`**

- `{self.tau:.4g}` 里的 `:.4g` 是**格式说明符**，和 MATLAB 的 `%.4g` **完全一样**。
- 写法规律：`{值:格式}`，冒号后面就是你在 MATLAB 里习惯的那套。

| MATLAB | Python f-string |
|---|---|
| `%d` | `{n:d}` |
| `%s` | `{s}` |
| `%.6e` | `{x:.6e}` |
| `%.4g` | `{x:.4g}` |
| `%4d`（右对齐宽度 4） | `{n:4d}` |

**⑤ `if self.homotopy_name:`**

这是**Python 的一个关键习惯**：`if` 后面跟的不是"比较"，而是"这个值算不算真"。

| 值 | Python 里算真还是假 |
|---|---|
| `""`（空字符串） | **假** |
| `0` | **假** |
| `None` | **假** |
| `[]`（空列表） | **假** |
| 任何非空字符串 / 非零数 | **真** |

所以这行等价于 MATLAB 的：

```matlab
if ~isempty(obj.homotopyName)
```

**看到 `if 某某:` 而里面没有比较运算符，就要反应过来：这是在判断"非空/非零"。**

## 2.2 例二：`time_grid`（带默认参数）

```python
def time_grid(self, N=None):
    if N is None:
        N = self.N
    return np.linspace(0.0, self.tau, int(N))
```

**逐行：**

**① `def time_grid(self, N=None):`**

`N=None` 的意思是"**这个参数可以不传**，不传时它的值是 `None`"。

**这对应 MATLAB 的 `nargin` 检查**：

```matlab
function tg = timeGrid(obj, N)
    if nargin < 2
        N = obj.N;
    end
    tg = linspace(0, obj.tau, N);
end
```

**记住这个模式**：Python 里用 `参数=None` + `if 参数 is None:` 来表达"没传这个参数"。

**② `if N is None:`**

- `is None` 是 Python 判断"是不是空"的**标准写法**（不要写 `== None`）。
- `None` 就是 MATLAB 的"空"（类似 `[]`）。

**③ `return np.linspace(0.0, self.tau, int(N))`**

- `return` = MATLAB 的输出赋值 + 立即返回。
- `np.linspace(...)` = MATLAB 的 `linspace(...)`，但**前面要写 `np.`**。
- `int(N)` = 强制转成整数（防止传进来的是浮点数）。

**关于 `np.` 前缀**：Python 的 numpy 库要先 `import numpy as np`，然后所有函数都写成 `np.xxx`。
MATLAB 的内置函数（`linspace`、`zeros`、`sqrt`）在 Python 里**大部分都在 numpy 里**，所以都要加 `np.`。

## 2.3 例三：`project`（第一次遇到 numpy 数组操作）

```python
def project(self, U):
    lo, hi = self.control_bounds()
    return np.clip(U, lo[:, None], hi[:, None])
```

**逐行：**

**① `lo, hi = self.control_bounds()`** —— **这是 Python 很常用的"一行接多个返回值"**。

MATLAB：
```matlab
[lo, hi] = obj.controlBounds();
```
Python：
```python
lo, hi = self.control_bounds()
```

`control_bounds()` 返回的是 `(下界数组, 上界数组)` 这样一个**元组**，
左边写几个变量，就按顺序接几个值。**这是 Python 最常用的写法之一，一定要认出来。**

**② `np.clip(U, 下界, 上界)`** —— 把 `U` 裁剪到 `[下界, 上界]` 区间内。

MATLAB 里是：
```matlab
max(lo, min(hi, U))
```
Python 一个 `np.clip` 就干完了同样的事。

**③ `lo[:, None]`** —— ⚠️ **这是 numpy 里最需要理解的一个语法。**

- `lo` 的形状是 `(2,)` —— 一维，2 个元素。
- `U` 的形状是 `(2, N)` —— 二维。
- 要把它们放在一起运算，`lo` 得"站起来"变成 `(2, 1)`，才能和 `(2, N)` 对上。

`lo[:, None]` 的意思就是"**在第二个位置加一个新维度**"，把 `(2,)` 变成 `(2, 1)`。

**类比 MATLAB**：MATLAB 里 `lo` 本来就是 `(2,1)` 列向量，所以不用管。
numpy 里一维数组没有"行/列"之分，需要时得手动加维度。

| 写法 | 形状变化 | 记忆 |
|---|---|---|
| `x[:, None]` | `(N,)` → `(N, 1)` | 竖起来 |
| `x[None, :]` | `(N,)` → `(1, N)` | 躺下去 |

> 🔴 **读代码时看到 `[:, None]` 或 `[None, :]`，就理解成"为了和另一个数组对齐形状而调整维度"。**

---

# 第 3 节：numpy 数组 vs MATLAB 矩阵

这一节单独讲，因为项目里到处都是数组操作。

## 3.1 "数组"就是矩阵，但有一维的

| 概念 | MATLAB | numpy |
|---|---|---|
| 2 维矩阵 | `zeros(3, 4)` → 3×4 | `np.zeros((3, 4))` → `(3, 4)` |
| 1 维向量 | `zeros(1, 4)`（其实是 1×4 矩阵） | `np.zeros(4)` → **`(4,)`，没有行/列之分** |
| 标量 | `5` | `5`（或者 `np.float64(5)`） |

**关键差异**：MATLAB 里所有东西至少是 2 维的（`zeros(1,4)` 是 1×4 矩阵）；
numpy 有**真正的 1 维数组**，形状写作 `(4,)`，既有别于 `(1,4)` 也有别于 `(4,1)`。

**读代码时的信号**：
- 看到 `.shape` 打印出 `(60,)` → 一维，60 个元素
- 看到 `(3, 60)` → 3 行 60 列的二维数组

## 3.2 切片：`[行切片, 列切片]`

```python
X[:, 0]     # 第 0 列（所有行）      ≈ MATLAB 的 X(:, 1)
X[0, :]     # 第 0 行（所有列）      ≈ MATLAB 的 X(1, :)
X[:, -1]    # 最后一列               ≈ MATLAB 的 X(:, end)
X[1:]       # 从第 1 个到末尾        ≈ MATLAB 的 X(2:end)
X[:5]       # 前 5 个（0..4）        ≈ MATLAB 的 X(1:5)
X[k+1]      # 第 k+1 个（0-based）   ≈ MATLAB 的 X(k+2)：⚠️ 小心！
```

> 🔴 **重写代码时 90% 的下标 bug 都出在这里。**
> 看到 `X[:, -1]` 要知道它是"最后一列"（对应 `X(:, end)`），
> 而不是"从倒数第一列开始"。

## 3.3 布尔索引（很常用，一定要认出来）

```python
small = Lam < 1e-9        # small 是一个和 Lam 同形状的"真/假"数组
tx[small] = 0.0           # 把所有 small 为真的位置赋 0
```

MATLAB 里一模一样：
```matlab
small = Lambda < 1e-9;
tx(small) = 0;
```

**Python 只是把圆括号换成了方括号。** 读代码时看到 `数组[布尔数组] = 值`，
就是"按条件批量赋值"。

## 3.4 `np.where`：向量化的 if-else

```python
G = np.where(np.isnan(G), 0.0, G)
```

意思：`G` 里是 NaN 的位置换成 `0.0`，否则保持原值。

MATLAB 对照：
```matlab
G(isnan(G)) = 0;
```

**`np.where(条件, 真时取的值, 假时取的值)`** —— 记住这个三参数形式，项目里出现很多次。

## 3.5 `np.einsum`：一眼看懂的技巧（项目里只出现 2 次）

`_gradient_at` 里有这么一行：

```python
Hth = np.einsum("ik,ik->k", Mu, p0th) + np.einsum("ik,ijk,jk->k", Mu, Pth, X)
```

**读法**：把箭头 `->` 左边看成"输入的维度标签"，右边是"输出的标签"。
**出现在左边、但没出现在右边的字母，就是要被求和掉的维度。**

- `"ik,ik->k"`：输入是 `i,k` 和 `i,k`，输出是 `k`。
  所以 `i` 被求和掉了 → 结果是 `result[k] = Σ_i Mu[i,k] * p0th[i,k]`。
  （MATLAB：`sum(Mu(:,k) .* p0th(:,k))`）

- `"ik,ijk,jk->k"`：`i` 和 `j` 都被求和掉 → 对 `i` 和 `j` 双重求和。

**你不需要会写 einsum，但要能读出"它在对哪些维度求和"。**

---

# 第 4 节：类和 self

项目里所有核心代码都是类。这一节讲清楚 Python 的类长什么样。

## 4.1 一个类的最小结构

```python
class AbstractOCP(ABC):
    n_state = 0                      # ← 类属性（不需要实例就存在）

    def __init__(self):              # ← 构造函数
        self.tau = 1.0               # ← 实例属性（每个对象各自一份）

    def print_info(self):            # ← 方法
        print(self.name)
```

对照 MATLAB：

```matlab
classdef (Abstract) AbstractOCP < handle
    properties (Abstract)
        n_state
    end
    properties
        tau = 1
    end
    methods
        function obj = AbstractOCP()
            obj.tau = 1;
        end
        function printInfo(obj)
            fprintf('%s\n', obj.name);
        end
    end
end
```

**关键对照：**

| MATLAB | Python |
|---|---|
| `classdef A < handle` | `class A:` |
| `properties` 块 | 在 `__init__` 里写 `self.xxx = ...` |
| `properties` 带默认值 | 可以写成类属性直接赋默认值 |
| `function obj = AbstractOCP()` | `def __init__(self):` |
| `obj.tau = 1` | `self.tau = 1.0` |
| `AbstractOCP(参数)` | `AbstractOCP(参数)`（调用方式一样） |

**`__init__` 是什么**：名字两边各有两个下划线，叫"构造函数"。
创建对象时自动调用，`m = QubitReset(tau=0.15)` 就等于 MATLAB 的 `m = ocp.QubitReset('tau',0.15)`。

**为什么 Python 要在 `__init__` 里一个个写 `self.xxx = ...`**：
因为 Python 没有 MATLAB 那种"在 properties 块里声明并给默认值"的机制，
必须在构造时显式赋值。

## 4.2 继承

```python
class QubitReset(AbstractOCP):        # ← 继承，括号里写父类
    def __init__(self, **kwargs):
        super().__init__()            # ← 先调用父类构造函数
        self.gamma = 10.0
```

对照 MATLAB：`classdef QubitReset < ocp.AbstractOCP`。

**`super().__init__()` 必须写**（在 Python 里）：
它负责把父类的属性也初始化好。这是 MATLAB 里自动完成的，Python 要手动调。

## 4.3 `**kwargs`：接收任意关键字参数

```python
def __init__(self, **kwargs):
    super().__init__()
    self.set(**kwargs)
```

`**kwargs` 的意思是"**把所有额外的关键字参数收集成一个字典**"。

用法：
```python
m = QubitReset(tau=0.15, N=100, gamma=10)
# 这时 kwargs = {"tau": 0.15, "N": 100, "gamma": 10}
```

对照 MATLAB 的：
```matlab
function obj = QubitReset(varargin)
```
调用时 `ocp.QubitReset('tau', 0.15, 'N', 100, 'gamma', 10)`。

**区别**：MATLAB 用"名-值对"（字符串 + 值），Python 用**关键字参数**（`名字=值`）。
Python 的写法更安全（名字写错会报错）。

## 4.4 `set` 方法：动态设属性

```python
def set(self, **kwargs):
    for key, value in kwargs.items():
        if not hasattr(self, key):
            raise AttributeError(f"...没有属性 '{key}'")
        setattr(self, key, value)
    return self
```

**逐行：**

| Python | 含义 | MATLAB |
|---|---|---|
| `kwargs.items()` | 把字典变成"键值对"序列 | — |
| `for key, value in ...` | 每个键值对拆成两个变量 | — |
| `hasattr(self, key)` | 检查属性是否存在 | `isprop(obj, name)` |
| `raise AttributeError(...)` | 抛错 | `error(...)` |
| `setattr(self, key, value)` | 按名字设属性 | `obj.(name) = value` |
| `return self` | 返回自身（支持链式调用） | — |

---

# 第 5 节：装饰器（那些 @）

## 5.1 `@staticmethod` —— "这函数不需要 self"

```python
@staticmethod
def trapezoid(tg, y):
    return float(np.trapezoid(y, tg, axis=-1))
```

**看到 `@staticmethod`，你就知道**：调用它时不用先创建对象，而且函数体内**不会出现 `self`**。

MATLAB 里对应"不依赖 obj 的辅助函数"（MATLAB 的 `methods (Static)`）。

## 5.2 `@abstractmethod` —— "子类必须实现"

```python
from abc import ABC, abstractmethod

class AbstractOCP(ABC):
    @abstractmethod
    def time_grid(self, N=None):
        """返回时间格点"""
```

**含义**：这个类不能直接实例化；子类**必须**实现所有带 `@abstractmethod` 的方法，否则创建对象时报错。

MATLAB 对照：`methods (Abstract)`。

**看代码时的意义**：你在 `abstract_ocp.py` 里看到这些只有文档字符串、没有实现的函数，
就知道"真正的内容在 `qubit_reset.py` / `control_example.py` 里"。

## 5.3 `@dataclass` —— "这个类只是装数据的"

```python
from dataclasses import dataclass

@dataclass
class History:
    J: list = field(default_factory=list)
    gnorm: list = field(default_factory=list)
```

**含义**：Python 自动帮你生成构造函数等代码。等价于手写：

```python
class History:
    def __init__(self, J=None, gnorm=None):
        self.J = [] if J is None else J
        self.gnorm = [] if gnorm is None else gnorm
```

**MATLAB 对照**：`struct`。

**看到 `@dataclass` 就理解成"这是个结构体"**，然后看它的字段列表就够了。

> `field(default_factory=list)` 的意思是"默认值是一个新的空列表"。
> 为什么要这么绕？因为直接写 `J: list = []` 会让所有实例**共享同一个列表**（Python 的坑）。
> 你只要记住：**看到 `default_factory=list` 就是"默认是空列表"**。

---

# 第 6 节：其他会卡住你的语法点

## 6.1 `for` 循环

```python
for k in range(N - 1):          # k = 0, 1, ..., N-2
for k in range(1, N + 1):       # k = 1, 2, ..., N
for k in range(N - 1, 0, -1):   # k = N-1, ..., 1（倒着数）
for i in range(ns):             # i = 0, ..., ns-1
for key, value in d.items():    # 遍历字典
for st in res.results:          # 遍历列表
```

**`range(a, b)` 生成 `a` 到 `b-1`（右端不含！）。**

MATLAB 对照表：

| MATLAB | Python |
|---|---|
| `for k = 1:N` | `for k in range(1, N+1)` |
| `for k = 1:N-1` | `for k in range(N-1)`（等价 `range(0, N-1)`） |
| `for k = N:-1:2` | `for k in range(N-1, 0, -1)` |
| `for i = 1:numel(a)` | `for i in range(len(a))` |

> 🔴 **后向循环是重写代码时最容易错的地方。** 记忆法：
> **把 MATLAB 的下标整体减 1**，`N:-1:2` → `range(N-1, 1-1, -1)` = `range(N-1, 0, -1)`。

## 6.2 `for ... else`（Python 特有）

```python
for it in range(1, max_iter + 1):
    ...
    if 收敛:
        break
else:
    reason = "max iteration"     # ← 只有循环没被 break 时才会执行
```

**`else` 挂在 `for` 上**，意思是"循环**正常跑完**（没被 `break` 打断）时执行这里"。

## 6.3 列表 / 字典 / 元组

| 类型 | 写法 | MATLAB 对应 |
|---|---|---|
| 列表 list | `[1, 2, 3]` | 一维数组 / cell |
| 元组 tuple | `(1, 2, 3)` | 不可改的列表 |
| 字典 dict | `{"a": 1, "b": 2}` | `containers.Map` 或 struct |

常用操作：

```python
hist.J.append(x)      # 往列表末尾加一个元素   ≈ hist.J(end+1) = x
len(hist.J)           # 长度                   ≈ numel(hist.J)
hist.J[-1]            # 最后一个元素           ≈ hist.J(end)
hist.J[0]             # 第一个元素             ≈ hist.J(1)
```

## 6.4 推导式（看起来吓人，其实很简单）

```python
self._ax = [self._fig.add_subplot(2, 2, p + 1) for p in range(4)]
```

**读法**：把方括号里的内容理解成"一个循环，每次产出一个元素放进列表"。
等价于：

```python
self._ax = []
for p in range(4):
    self._ax.append(self._fig.add_subplot(2, 2, p + 1))
```

**以后看到 `[表达式 for 变量 in 序列]`，就在脑子里展开成上面那个 for 循环。**

## 6.5 `lambda`（就是 MATLAB 的匿名函数）

```python
X[:, k + 1] = self._rk4(lambda x, A=A, b=b: A @ x + b, X[:, k], dt)
```

`lambda x: A @ x + b` 等价于 MATLAB 的 `@(x) A*x + b`。

**`lambda 参数: 表达式`** —— 冒号左边是参数，右边是返回值（只能写一个表达式）。

> 这里写的是 `lambda x, A=A, b=b:`，多出来的 `A=A, b=b` 是"把当前值固定进去"，
> 避免循环变量在后面被改掉。看到这种写法不用慌，理解成"把需要的值一起带上"即可。

## 6.6 `with` 语句

```python
with np.errstate(invalid="ignore", divide="ignore"):
    re = 0.5 * self._spec(2 * la) * (1.0 / np.tanh(la) + 1.0) * np.cos(th) ** 2
```

**含义**："在这个块内，临时改一下某个设置，出了块自动恢复。"

这里的意思是"在这个块里，**不要对除零/NaN 报警告**"。
MATLAB 里对应的是 `warning('off', ...)` ... `warning('on', ...)`，但 Python 的 `with` 会自动恢复。

## 6.7 `try` / `except`

```python
try:
    res = fetch(url)
except Exception as e:
    print(f"失败: {e}")
```

MATLAB 对照：`try ... catch ... end`。

## 6.8 `if __name__ == "__main__":`

demo 文件末尾都有这个：

```python
if __name__ == "__main__":
    main()
```

**含义**："**只有当这个文件被直接运行时**，才执行 `main()`"。

- 直接运行 `python demo_gradient.py` → `__name__` 是 `"__main__"` → 执行
- 被别人 `import demo_gradient` → `__name__` 是模块名 → 不执行

**作用**：让文件既能当脚本跑，又能被别的代码引用而不产生副作用。

## 6.9 下划线开头的名字

```python
self._m = None        # 单下划线开头
self.__x = 1          # 双下划线开头
def _compute_direction(self): ...
```

**单下划线 = "这是内部的，请别从外面直接用它"**（只是约定，不是强制）。

MATLAB 对照：`properties (Access = private)` 和 `methods (Access = private)`。

**读代码时的意义**：看到 `_` 开头的，就知道它是内部实现细节，可以最后再读。

---

# 第 7 节：实践 —— 用这套方法读一个新的函数

现在试试看你能不能独立读懂下面这个函数（它在 `gradient_solver.py` 里）：

```python
def _apply_active_set(self, U, G):
    lo, hi = self.model.control_bounds()
    at_up = U >= hi[:, None] - 1e-12
    at_lo = U <= lo[:, None] + 1e-12
    outward = (at_up & (G < -1e-12)) | (at_lo & (G > 1e-12))
    G = G.copy()
    G[outward] = 0.0
    return G
```

**自己先读一遍，然后对照下面的拆解：**

**① `def _apply_active_set(self, U, G):`**
一个私有方法（`_` 开头），需要 `self`，接收两个参数 `U` 和 `G`。

**② `lo, hi = self.model.control_bounds()`**
调用模型的 `control_bounds()`，它返回 `(下界, 上界)` 两个值，用 `lo, hi` 接住。

**③ `at_up = U >= hi[:, None] - 1e-12`**
- `hi[:, None]` → 把上界数组的形状从 `(2,)` 变成 `(2, 1)`
- `U >= ...` → 逐元素比较，`at_up` 是一个**布尔数组**
- `- 1e-12` 是一个很小的容差（浮点误差下"刚好等于"可能算不出来）
- 这一行的意思：**哪些位置的控制已经贴着上界了**

**④ `at_lo = U <= lo[:, None] + 1e-12`**
同理，哪些位置贴着下界。

**⑤ `outward = (at_up & (G < -1e-12)) | (at_lo & (G > 1e-12))`**
- `&` 是"与"（MATLAB 的 `&`），`|` 是"或"（MATLAB 的 `|`）
- 整句意思：**贴在上界、且梯度还想把它往外推** 的位置，或者 **贴在下界、且梯度还想往外推** 的位置

**⑥ `G = G.copy()`**
**复制一份！** 因为下一行要**原地修改** `G`，不复制就会改到传进来的那个数组（影响调用方）。

**⑦ `G[outward] = 0.0`**
布尔索引赋值：把 `outward` 为真的位置全部设为 `0`。等价 MATLAB 的 `G(outward) = 0;`

**⑧ `return G`**
返回处理后的梯度。

**这个函数在做什么（一句话）**：把"顶在边界上、且梯度还在往外推"的那些分量的梯度置零
（这就是 KKT 活动集处理）。

---

# 第 8 节：你现在应该做的事

## 8.1 立刻做：把项目按难度顺序读一遍

| 顺序 | 文件 | 难度 | 你会遇到的新语法 |
|---|---|---|---|
| 1 | `sigmax_ocp/ocp/abstract_ocp.py` | ★ | `def`、`ABC`、f-string、`if 非空` |
| 2 | `sigmax_ocp/ocp/control_example.py` | ★★ | numpy 数组、RK4 循环、`[None, :]` |
| 3 | `sigmax_ocp/solver/line_search.py` | ★★ | `@dataclass`、`while`、三参数 `np.where` |
| 4 | `sigmax_ocp/solver/gradient_solver.py` | ★★★ | 元组解包、`for...else`、字典 `.items()` |
| 5 | `sigmax_ocp/ocp/qubit_reset.py` | ★★★★ | 布尔索引、`np.errstate`、`einsum`、三维数组 |
| 6 | `sigmax_ocp/continuation/*.py` | ★★ | 列表、`@dataclass` 套用 |
| 7 | `sigmax_ocp/vis/visualizer.py` | ★★★ | 列表推导式、`with`、延迟 `import` |

**不要按字母顺序读**，按上面这个难度顺序读。

## 8.2 三个能立刻验证自己"看懂了"的方法

**方法一：把 Python 翻译回 MATLAB。**
随便挑一个函数，用纸笔写成 MATLAB 的形式。写不出来的地方，就是你没懂的地方。

**方法二：给函数写一句话注释。**
在 `gradient_solver.py` 的每个函数上方，用自己的话写一句话说明它干什么。
如果你写不出来，说明那段代码你还没读懂。

**方法三：改坏它，看会怎样。**
比如把 `_apply_active_set` 里的 `G.copy()` 删掉，然后想清楚"为什么这样会出问题"。
**能预测改动后果，就是真懂了。**

## 8.3 遇到看不懂的语法时，这样查

1. **先在本文档的对照表里找**（绝大多数语法点都收录了）
2. **再看该行上下文的注释**（项目代码里注释写得比较密）
3. **实在不行，把那一行单独拿出来跑一下**：

```python
a = [1, 2, 3]
print(a[-1])        # 猜猜输出什么？→ 3
print(a[0:2])       # → [1, 2]（右端不含）
x = 5
print(f"{x:.2f}")   # → 5.00
```

**Python 最好的学习方法就是"改一行、跑一下、看输出"。**
你现在有可运行的环境，这一步成本极低 —— 一定要用起来。

---

# 附：一张"我卡住了"急救表

| 你看到的 | 它的意思 | 回头看第几节 |
|---|---|---|
| `self.xxx` | MATLAB 的 `obj.xxx` | 0.2 / 4.1 |
| `def f(self, x):` | MATLAB 的 `function ... (obj, x)` | 0.2 / 2.1 |
| 没有 `end` | 缩进就是块边界 | 0.1 |
| `f"...{x:.4g}"` | MATLAB 的 `fprintf('%...%.4g', x)` | 2.1 ④ |
| `if x:` | `if ~isempty(x)` / `if x ~= 0` | 2.1 ⑤ |
| `x[:, None]` | 为了对齐形状而加维度 | 2.3 ③ |
| `A @ x` | MATLAB 的 `A * x`（矩阵乘） | 1.3 |
| `x * y` | MATLAB 的 `x .* y`（逐元素） | 1.3 |
| `np.sum(G, axis=0)` | MATLAB 的 `sum(G, 1)` | 1.4 |
| `a[0]` | MATLAB 的 `a(1)` | 1.2 |
| `a[1:5]` | MATLAB 的 `a(2:5)` | 1.2 |
| `a[-1]` | MATLAB 的 `a(end)` | 1.2 |
| `for k in range(N-1)` | MATLAB 的 `for k = 1:N-1` | 6.1 |
| `range(N-1, 0, -1)` | MATLAB 的 `for k = N:-1:2` | 6.1 |
| `@staticmethod` | 不需要 self 的辅助函数 | 5.1 |
| `@dataclass` | 相当于 struct | 5.3 |
| `[f(x) for x in seq]` | 一个循环，生成列表 | 6.4 |
| `lambda x: ...` | MATLAB 的 `@(x) ...` | 6.5 |
| `with ...:` | 临时设置，出块自动恢复 | 6.6 |
| `if __name__ == "__main__":` | 只在直接运行时执行 | 6.8 |
| `_name` | 内部使用，别从外面碰 | 6.9 |
| `lo, hi = f()` | MATLAB 的 `[lo, hi] = f()` | 2.3 ① |
| `np.where(cond, a, b)` | 向量化的 if-else | 3.4 |
| `lo[:, None]` / `x[None, :]` | 加一个新维度 | 2.3 ③ |
| `np.einsum("ik,ik->k", ...)` | 对被省略的字母求和 | 3.5 |
