---
title: [BoTorch]Monte Carlo獲得関数とその最適化
tags: ベイズ最適化 ガウス過程 BoTorch GPyTorch
author: narrowlyapplicable
slide: false
---
# Outline
1. Monte Carlo 獲得関数
2. 勾配法による最大化
3. 貪欲法による逐次最適化（Sequentialモード）

# Intro

[前回](https://qiita.com/narrowlyapplicable/items/d8e9be53f73d6fa5e4d3)に引き続き、[BoTorch](https://botorch.org/)の解説記事です。

BoTorchの最大の特徴は、獲得関数の最大化を勾配法で統一している点にあります。  
ベイズ最適化（Bayesian Optimization; BO）では通常、候補点の中から獲得関数を最大にする点を選び、次の評価対象とします。
$$x^{*} = \arg\max _{\mathbf{x}}{\mathcal{L}(\mathbf{x})}$$
しかしこの最大化は必ずしも容易ではなく、複雑な獲得関数（Entropy SearchやKnowledge Gradientなど）を扱う場合や、複数の候補点を選ぶ（Parallel Selection）場合などにおいては、一般に勾配法による最適化は実行困難となります。この最大化自体がBlack-Box最適化問題となり、CMA-ESなど別のBlack-Box最適化手法に頼ることになります。  

BoTorchはこうした煩雑な処理を避け、獲得関数最大化を通常の勾配法に統一しています。これにより、GpyOptなどの先行ライブラリに比べて、多彩な獲得関数の使用やParallel Selectionの気軽な実行が可能となっています。  
これを可能にしているのが、Monte Carlo獲得関数（以下、「MC獲得関数」）です。本記事では、このMC獲得関数の解説を目的とします。  

前回同様、ガウス過程やベイズ最適化の基礎は説明しません。他の資料を参照してください。

# 1. Monte Carlo獲得関数
## 1.1. 元論文
BoTorchのドキュメントにはMC獲得関数およびその最大化に関する参考文献がいくつか示されていますが、主要なものは下記の2つです。

- 元論文１：[The Reparameterization Trick for Acquisition Functions](https://arxiv.org/abs/1712.00424)
- 元論文２：[Maximizing acquisition functions for Bayesian optimization](https://proceedings.neurips.cc/paper/2018/hash/498f2c21688f6451d9f5fd09d53edda7-Abstract.html)

元論文１でMC獲得関数が初めて提案されています。この時点で獲得関数の最大化が目的に挙げられていますが、元論文２で獲得関数最大化自体が扱われています。

これ以前にも、EI（Expected Improvement）やKG（Knowledge Gradient）などの特定の獲得関数において、Parallel Selectionを可能とする改良案（qEI, qKG）は提案されていました。

- 先行論文１：[A Multi-points Criterion for Deterministic Parallel Global Optimization based on Gaussian Processes](https://hal.archives-ouvertes.fr/hal-00260579)
- 先行論文２：[The Parallel Knowledge Gradient Method for Batch Bayesian Optimization](https://arxiv.org/abs/1606.04414)
- 先行論文３：[Parallel Bayesian Global Optimization of Expensive Functions](https://arxiv.org/abs/1602.05149)

しかし上記の元論文×２で提案されたMC獲得関数は、より広範な（よく知られたもののうち大半の）獲得関数に対して一貫した手法を提供しました。BoTorchはこのMC獲得関数を核とし、多彩な獲得関数を扱いやすい形で提供しています。

## 1.2. 定義
### 1.2.1. 必要な表記と諸概念
問題設定 

- 観測済みのデータ $\mathcal{D} := \\{ ({\mathbf{x}}_i, y_i) \\} _{i=1}^N, {\mathbf{x}}_i \in {\mathbb{R}}^{d}$ から、最適なq個の候補点 $\mathbf{X}\in\mathbb{R}^{q \times d}$ を決定したい。
- 代理モデル（surrogate model）としてガウス過程（GP） $p(f|\mathcal{D})$ を使い、そのハイパーパラメータをデータに適合させた結果を $\mathcal{M}(\mathbf{X}):=(\mu(\mathbf{X}), \mathbf{\Sigma}(\mathbf{X}))$ と書く。$\mathbf{X}$の下での事後分布は $\mathcal{N}(\mathbf{y}|\mathbf{\mu}, \mathbf{\Sigma})$。

utility関数

- 大半の獲得関数は、何らかの関数（utility関数）$l$の期待値として表すことができます。
  - 獲得関数 $\mathcal{L}(\mathbf{X})$ のutility関数を $l(\mathbf{y})$ とすると、$$\mathcal{L}(\mathbf{X}) = \mathbb{E}_{\mathbf{y}}[l(\mathbf{y})] = \int{l(\mathbf{y})p(\mathbf{y}|\mathbf{X}, \mathcal{D})d\mathbf{y}}$$ と書き換えます。
  - 獲得関数がパラメータ$\alpha$を持つ場合、utilityもそれに対応し $\mathcal{L}(\mathbf{X};\alpha) = \mathbb{E}_{\mathbf{y}}[l(\mathbf{y};\alpha)]$ となります。
  - 代表的な獲得関数に対するutilityの一覧は、元論文２の表に示されています。
   ![utility.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/1617118/dc817638-68ae-66c5-01f2-fb2a31b42b15.png)

    出典：[Maximizing acquisition functions for Bayesian optimization](https://proceedings.neurips.cc/paper/2018/hash/498f2c21688f6451d9f5fd09d53edda7-Abstract.html)

### 1.2.2. Monte Carlo近似
ガウス過程の事後分布 $p(\mathbf{y}|\mathbf{X}, \mathcal{D})$ はパラメータ既知の正規分布なので、サンプル$\mathbf{y}^k\sim p(\mathbf{y}|\mathbf{X}, \mathcal{D})$を生成してMonte Carlo近似できます。

$$\mathcal{L}(\mathbf{X}) \approx \mathcal{L}_m(\mathbf{X}) := \frac{1}{m}\sum _{k=1}^m{l(\mathbf{y^k})}$$

勾配法で最適な入力$\mathbf{X}$を決めるには、獲得関数の勾配が必要です。
上記の近似から獲得関数の勾配を求めるには、微分と期待値（積分）の交換 $$\nabla\mathcal{L} = \nabla\mathbb{E}[l(\mathbf{y})]=\mathbb{E}[\nabla l(\mathbf{y})]$$ が成り立つ必要があります。

- 交換 $\nabla\mathbb{E}[l(\mathbf{y})]=\mathbb{E}[\nabla l(\mathbf{y})]$ の成立条件
  - 被積分関数 $l$ が連続
  - $l^{\prime}$ がほとんど至る所で（そうでない点の測度が0で）存在し、積分可能

この条件はGPのカーネル関数に依存しますが、[2回微分可能なカーネルを使用すれば成立することが示されており](https://arxiv.org/abs/1602.05149)、通常用いるMaternカーネルなどでは問題になりません。
（リンク先の論文の§4.1.で交換可能性が検討されています。）

上記の交換が成り立てば、獲得関数の勾配もMonte Carlo近似できます。

$$\nabla\mathcal{L}(\mathbf{X}) \approx \nabla\mathcal{L} _m(\mathbf{X}):= \frac{1}{m}\sum _{k=1}^m{\nabla l(\mathbf{y^k})}$$

以上より、*勾配を計算できる獲得関数の近似* $\mathcal{L}_m(\mathbf{X})$ が得られました。これが**Monte Carlo獲得関数**です。

## 1.3. BoTorchにおける実装例
EIを例に、BoTorchにおけるMC獲得関数の実装を確認しておきます。EIのMC獲得関数版（qEI）は[`qExpectedImprovement`](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/acquisition/monte_carlo.py#L93)として実装されています。このうち実際の計算を担うのは`forward()`メソッドです。

```py:monte_carlo.py
    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        ### (中略)
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior) ## 1. 事後分布からサンプリング
        obj = self.objective(samples, X=X) ## 2. 指定した場合のみ、サンプルを変換
        obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0) ## 3. utility計算
        q_ei = obj.max(dim=-1)[0].mean(dim=0) ## 4. qEIを計算
        return q_ei
```

- 冒頭のデコレータ×２は最適化の際に使用するもので、獲得関数の計算自体には影響しません。
  - 最初の`concatenate_pending_points`デコレータは、Sequentialモード（後述）で候補点を逐次的に決めていく際、決定済みの候補点を入力に追加する（しかし最適化対象からは外す）ために使用するものです。
  - 2番目の`t_batch_mode_transform()`は、初期値を変えて複数回の最適化を独立に実行する（t-batch動作）のため、入力データを変換するデコレータです。

- `forward(X)`において、指定した入力点（複数可）`X`に対するqEIを計算します。
  1. `X`における事後分布 $p(\mathbf{y}|\mathbf{X})$ から、準モンテカルロ法によるサンプリング
     - `posterior = self.model.posterior(X)`で、与えたGPモデル`model`の事後分布を取得し、`samples = self.sampler(posterior)`によりサンプリング実行
     - `self.sampler`は何も指定しなければ`SobolQMCNormalSampler`で512個のサンプルを生成
       - このサンプリングでは、後述するre-parametrizationを使用しています。
  2. 取得サンプル$\{\mathbf{y}^k\}$ を、指定した`objective`で変形
     - デフォルトでは`squeeze(-1)`するだけの`IdentityMCObjective`
     - 出力が多変数の場合、出力に重み付けするために使用する？
  3. qEIのutilityを計算
     - $ReLU(\mathbf{y} - \alpha)$ を計算
         - `obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)`
  4. qEIを計算
     - 候補点$\mathbf{X}$に関する最大値 $\max(ReLU(\mathbf{y} - \alpha))$ を取る
         - `obj.max(dim=-1)[0]`
     - サンプル平均 = 期待値のMC近似を求め $\mathcal{L} _m(\mathbf{X}) = \mathbb{E} _{\mathbf{y}}[\max(ReLU(\mathbf{y} - \alpha))]$ を得る

# 2. 勾配法による最適化
## 2.1. re-parametrizationによる勾配評価
MC獲得関数の勾配

```math
\nabla{l(\mathbf{y})}=\frac{\partial l(\mathbf{y})}{\partial{\mathbf{y}}} \frac
{\partial{\mathbf{y}}}{\partial \mathcal{M}(\mathbf{X})} \frac{\partial \mathcal{M}(\mathbf{X})}{\partial \mathbf{X}}
```

は連鎖律の途中に事後分布 $p(\mathbf{y}|\mathbf{X})=p(\mathbf{y}|\mathcal{M}(\mathbf{X}))$ からのサンプリングを挟むため、勾配の評価にはre-parametrization（再パラメータ化）が必要になります。
このre-parametrizationはVAEなどと同様です。

決定論的な関数 $\phi$ により、事後分布からのサンプル $\mathbf{y}^k \sim p(\mathbf{y}|\theta)$ を、各次元独立な標準正規分布からのサンプル $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ で置き換えます。

$$\mathbb{E}_{\mathbf{y}}[l(\mathbf{y})]=\mathbb{E} _{\mathbf{z}}[l(\phi(\mathbf{z}))],\quad\mathbf{y}^k=\phi(\mathbf{z},\theta)$$

ガウス過程であれば事後分布は正規分布となるので、 $\mathbf{\Sigma}$ のコレスキー分解 $\mathbf{L}$ を用いて $\phi(\mathbf{z},\mathcal{M}(\mathbf{X})):=\boldsymbol{\mu}+\mathbf{L}\mathbf{z}$ とすれば、

$$\nabla\mathcal{L} _m(\mathbf{X})=\mathbb{E} _{\mathbf{z}}[\nabla l(\phi(\mathbf{z}))]$$

として計算できます。

このre-parametrizationはBoTorchモデルの事後分布が備える`rsample`メソッドによって実現されます。
通常事後分布からのサンプリングには[SobolQMCNormalSampler](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/sampling/samplers.py#L226)が用いられますが、これは

1. まず標準正規分布に従うサンプルを生成し、
2. そのサンプルを`posterior.rsample()`によってre-parametrizationする

という処理になっています。
[forward部分の実装](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/sampling/samplers.py#L82)を見ると、

```py:samplers.py
    def forward(self, posterior: Posterior) -> Tensor:
        ## (中略)
        base_sample_shape = self._get_base_sample_shape(posterior=posterior)
        self._construct_base_samples(posterior=posterior, shape=base_sample_shape)
        samples = posterior.rsample(
            sample_shape=self.sample_shape, base_samples=self.base_samples
        )
        return 
```

- `_construct_base_samples()`で $\mathcal{N}(\mathbf{0}, \mathbf{I})$ からサンプル生成
- `posterior.rsample()`で、与えられた事後分布に合わせてre-parametrization

という流れが確認できます。

この`rsample()`は内部的には[GPyTorchのmultivariate_normal](https://docs.gpytorch.ai/en/latest/_modules/gpytorch/distributions/multivariate_normal.html)が持つ[rsample()メソッド](https://docs.gpytorch.ai/en/latest/_modules/gpytorch/distributions/multivariate_normal.html)を実行しています（もちろん型や配列長を合わせるなどの処理は入っていますが）。これはまさにreparametrizationのためのメソッドです。
前処理・後処理が多く入っていますが、該当部分だけを抜き出すと下記のようになります。

```py:multivariate_normal.py
    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        covar = self.lazy_covariance_matrix
            covar_root = covar.root_decomposition().root
            ## (中略)
            # Now reparameterize those base samples
            ## (中略)
            res = covar_root.matmul(base_samples) + self.loc.unsqueeze(-1)
            ## (中略)
       return res
```

- 共分散行列を `covar.root_decomposition().root` によって分解
- $\boldsymbol{\mu} + \mathbf{L}\mathbf{z}$ を、`covar_root.matmul(base_samples) + self.loc.unsqueeze(-1)` として計算

以上の処理によって、MC獲得関数の勾配をPyTorchの誤差逆伝播で得ることができます。

## 2.2. optimize_acqf()による最適化
勾配が分かれば、候補点 $\mathbf{X}$ を最適化することができます。
ここではBoTorchの獲得関数最適化用関数である[optimize_acqf()](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/optim/optimize.py#L49)を確認しておきます。

`optimize_acqf()`は、処理の核となる`gen_candidates_scipy()`により候補点を作成します。

```py:optimize.py
    for i, batched_ics_ in enumerate(batched_ics):
        # optimize using random restart optimization
        batch_candidates_curr, batch_acq_values_curr = gen_candidates_scipy(
            initial_conditions=batched_ics_,
            acquisition_function=acq_function,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options={k: v for k, v in options.items() if k not in INIT_OPTION_KEYS},
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
        )
        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        logger.info(f"Generated candidate batch {i+1} of {len(batched_ics)}.")
    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)

    ## (中略)

    if return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]
    ## (中略)
    return batch_candidates, batch_acq_values
```
- for文はいわゆる"t-batch"で、初期値を変えて複数回の最適化を実行しています。
  - デフォルトでは `return_best_only=True` であり、"t-batch"で得た候補点 `batch_candidates` から獲得関数値 `batch_acq_values` が最大のものを返します。

処理の核となる [`gen_candidates_scipy()`](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/generation/gen.py#L31) は、その名の通りscipy.optimizingを使って獲得関数を最大化 or 最小化する候補点を返します。
制約条件や初期値処理が入っているものの、中核部分はscipyのminimizeをそのまま使っています。

```py:gen.py
    res = minimize(
        f,
        x0,
        method=options.get("method", "SLSQP" if constraints else "L-BFGS-B"),
        jac=True,
        bounds=bounds,
        constraints=constraints,
        callback=options.get("callback", None),
        options={k: v for k, v in options.items() if k not in ["method", "callback"]},
    )
```

- 引数`jac`をTrueとしたため、第一引数として与える（最小化する）関数は、値とともに勾配を返す必要があります。
- 第一引数として与える `f` は直前で定義されており、最小化するlossは獲得関数×(-1)とし、その勾配はpytorchの自動微分で求めています。その実装からnanに関する処理を除くと、下記のようになります。

```py:gen.py
    def f(x):
        ## (中略)
        X = (
            torch.from_numpy(x)
            .to(initial_conditions)
            .view(shapeX)
            .contiguous()
            .requires_grad_(True)
        )
        X_fix = fix_features(X, fixed_features=fixed_features)
        loss = -acquisition_function(X_fix).sum()
        # compute gradient w.r.t. the inputs (does not accumulate in leaves)
        gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
        ## (中略)
        fval = loss.item()
        return fval, gradf

```

- `X`のうち固定する引数を`fixed_features`で固定した後、`loss`として指定した獲得関数×(-1)を計算しています。
- その後PyTorchの自動微分により、入力`X`に関する獲得関数の勾配を計算しています。MC獲得関数であれば、先ほどのre-parametrizationによってこの計算が可能となります。


# 3. 貪欲法による逐次最適化

§2. までで、複数の候補点を同時に最適化する手法を紹介しました。
しかし元論文２では、複数の候補点 $\mathbf{X}=(\mathbf{x}^1, ..., \mathbf{x}^p)$ を決定する場合において、（同時に最適化するより）1点ずつ逐次的に決定する方が良い性能を示すことが示唆されています。これは大半の獲得関数が持つ劣モジュラ性（submodularity）により、貪欲法（greedy algorithm）で最適解近傍へ到達することが保証されているためです。
BoTorchでは、貪欲法によって1点ずつ（逐次的に）候補点を決定していく手法が`Sequential`モードとして実装されています。

## 3.1. 劣モジュラ最適化
獲得関数を最大化するような（複数の）候補点を選ぶことは、組合せ最適化の一種とみなすことができます。
すなわち、有限集合 $\mathcal X$ の部分集合 $S \subset \mathcal X$ に対して実数値を取る関数 $a:2^{\mathcal X} \rightarrow \mathbb R$ を最大化する $S$ を選ぶ組合せ最適化問題を解いていることになります。このように、部分集合に実数値を割り当てる関数を **集合関数** と呼びます。

劣モジュラ性は、関数の凸性を集合関数（set function）に拡張した概念と見做せます。
具体的な定義は下記のようになります。

- 部分集合 $S,T \subset \mathcal X$ に対して、集合関数$a$が $$a(S)+a(T) \geq a(S \cup T) + a(S \cap T)$$ を満たすとき、$a$は **劣モジュラ関数（submodular function）** であるという。

劣モジュラ関数の詳細については、MLPシリーズで入門書籍が出ているので参照してください。

- [『劣モジュラ最適化と機械学習』](https://www.kspub.co.jp/book/detail/1529090.html)
  - この本の§3.3.では、相互情報量を獲得関数に用いた場合の候補点最適化が説明されています。

劣モジュラ関数の最大化問題はNP困難ですが、**貪欲法**（greedy algorithm）により良い近似解に到達できるという性質があります。

- 貪欲法 : 複数の候補点を得たい場合に、1点ずつ逐次的に決定していく手法。具体的には下記の通り。
  1.  $S=\emptyset$ とする。
  2.  $f(S \cup \\{x\\})$ を最大化する $x\in \mathcal X$ を選び、$S$ に追加する。
  3. ステップ2.を繰り返し、$|S|=q$ となったら停止する。

すなわち獲得関数が劣モジュラであれば、候補点 $\mathbf{X}=(\mathbf{x} _1 , \cdots, \mathbf{x} _q)$ を1点ずつ最適化していけば良いことになります。
既に決定した $k<q$ 個の候補点を $\mathbf{X} _{<k} = (\mathbf{x} _1 ,\cdots, \mathbf{x} _k)$ とすると、$k+1$点目$x\in\mathcal{X}$は $\mathcal{L} (\mathbf{X} _{<k} \cup \\{x\\})$ を最大化するように選んでいきます。

## 3.2. 候補点の逐次最適化
[元論文2](https://proceedings.neurips.cc/paper/2018/hash/498f2c21688f6451d9f5fd09d53edda7-Abstract.html)では、 $\mathcal{L}(\mathbf{X})=\mathbb{E}[\max{\hat{l}(\mathbf{y})}]$ の形で書ける獲得関数を"myopic maximal" (MM) と呼び、こうした獲得関数がいくつかの条件の下で劣モジュラ関数であることを示しています（証明略）。
代表的な獲得関数の中ではEI, PI, UCBなどがMMであり、したがってこれらの獲得関数を最適化する場合、前説に示した貪欲法で最適解付近に到達できることがわかります。

```math
x_{k+1} = \arg\max_x{\mathcal{L}(\mathbf{X}_{<k} \cup \{x\})}.
```

論文では比較実験も行われており、貪欲法による逐次最適化の方がより少ない評価回数でより最適な点に到達できると主張されています。

BoTorchで逐次最適化を実行するには、[optimize_acqf()](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/optim/optimize.py#L49)で`sequential=True`とします。
このとき内部では`q=1`とした`optimize_acqf()`が繰り返し実行され、逐次的に候補点を決定していきます。

```py:optimize.py
    if sequential and q > 1:
        ## (中略)
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending # 保留済みの点を除けておく
        for i in range(q):                      # 1点ずつ逐次決定
            candidate, acq_value = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=1,                            # 1点のみのoptmizeになっている
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                batch_initial_conditions=None,
                return_best_only=True,
                sequential=False,
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(         # 新規に決定した点を保留に追加
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

```

- `optimize_acqf(q=1)`が再帰的に呼ばれ、候補点を1点ずつ決定しています。
- 決定した候補点は`X_pending`として保留され、以降は最適化されません。しかし獲得関数値の算出時には`forward()`に入力されます。
  - §1.3.に示した通り、獲得関数の`forward()`には必ず[concatenate_pending_pointsデコレータ](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/utils/transforms.py#L248)が付いています。

  ```py:monte_carlo.py
      @concatenate_pending_points
      @t_batch_mode_transform()
      def forward(self, X: Tensor) -> Tensor:
  ```

  - この[concatenate_pending_pointsデコレータ](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/utils/transforms.py#L248)によって、`X_pending`（保留された点）は獲得関数への入力に結合されます。このデコレータによって保留された点は、獲得関数値の計算には反映されるものの、勾配は計算されず最適化対象からは外れるようになっています。

以上のように、BoTorchでは貪欲法によって複数候補点の逐次最適化を実行しています。


- 注意点として、逐次最適化の途中で保留した点 $\mathbf{X} _{<k}$ に対応する出力 $(\mathbf{y} _1, \cdots, \mathbf{y} _k)$ は、原則として仮定も予測もしていません。
現状のガウス過程モデルから $\mathbf{y}$ を予測し欠測を埋める手法（fantasize）もありますが、貪欲法ではその必要はありません。

  - 例外として、エントロピー探索系の獲得関数のMES (Max-value Entropy Search)では、複数候補点の決定においてfantasize()を使用しています。
     - [BoTorchのfantasize](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/models/model.py#L132)は、保留した各点 $\mathbf{x}_{j}$ における事後分布 $p(\mathbf{y}|\mathbf{X})$から$\mathbf{y}$をサンプリングし、それらを加えたGPを作成しています。

# 終わりに
BoTorchの基本を解説した前回に続き、BoTorchの中核を成すMonte Carlo獲得関数とその使用法について説明しました。私の勘違いや理解の甘い点などがありましたら、コメント等いただければ幸いです。~~マサカリは優しめにお願いします~~
さらに続けて、BoTorchに実装されている新しい獲得関数や関係する手法について記事を書ければと考えております。


