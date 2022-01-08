# Outline
1. Monte Carlo 獲得関数
2. 勾配法による最大化
3. 貪欲法による逐次最適化（Sequentialモード）

# Intro

[前回](https://qiita.com/narrowlyapplicable/items/d8e9be53f73d6fa5e4d3)に引き続き、[BoTorch](https://botorch.org/)の解説記事です。

BoTorchの最大の特徴は、獲得関数の最大化を勾配法で統一している点にあります。  
ベイズ最適化（Bayesian Optimization; BO）では通常、候補点の中から獲得関数を最大にする点を選び、次の評価対象とします。
$$x^{*} = \argmax_{\mathbb{x}}{\mathcal{L}(\mathbb{x})}$$
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
~~先行論文はあまりちゃんと読めてないです…~~

## 1.2. 定義
### 1.2.1. 必要な表記と諸概念
問題設定

- 観測済みのデータ $\mathcal{D} := \{ (\mathbb{x}_i, y_i) \}_{i=1}^N, \mathbb{x}_i\in\mathbb{R}^{d}$ から、最適なq個の候補点 $\mathbf{X}\in\mathbb{R}^{q\times d}$ を決定したい。
- 代理モデル（surrogate model）としてガウス過程（GP） $p(f|\mathcal{D})$ を使い、そのハイパーパラメータをデータに適合させた結果を $\mathcal{M}(\mathbf{X}):=(\mu(\mathbf{X}), \mathbf{\Sigma}(\mathbf{X}))$ と書く。$\mathbf{X}$の下での事後分布は $\mathcal{N}(\mathbf{y}|\mathbf{\mu}, \mathbf{\Sigma})$。

utility関数

- 大半の獲得関数は、何らかの関数（utility関数）$l$の期待値として表すことができます。
  - 獲得関数 $\mathcal{L}(\mathbf{X})$ のutility関数を $l(\mathbf{y})$ とすると、$$\mathcal{L}(\mathbf{X}) = \mathbb{E}_{\mathbf{y}}[l(\mathbf{y})] = \int{l(\mathbf{y})p(\mathbf{y}|\mathbf{X}, \mathcal{D})d\mathbf{y}}$$ と書き換えます。
  - 獲得関数がパラメータ$\alpha$を持つ場合、utilityもそれに対応し $\mathcal{L}(\mathbf{X};\alpha) = \mathbb{E}_{\mathbf{y}}[l(\mathbf{y};\alpha)]$ となります。
  - 代表的な獲得関数に対するutilityの一覧は、元論文２の表に示されています。![utility.png](./graph/utility.png) 
    出典：[Maximizing acquisition functions for Bayesian optimization](https://proceedings.neurips.cc/paper/2018/hash/498f2c21688f6451d9f5fd09d53edda7-Abstract.html)

### 1.2.2. Monte Carlo近似
ガウス過程の事後分布 $p(\mathbf{y}|\mathbf{X}, \mathcal{D})$ はパラメータ既知の正規分布なので、サンプル$\mathbf{y}^k\sim p(\mathbf{y}|\mathbf{X}, \mathcal{D})$を生成してMonte Carlo近似できます。
$$\mathcal{L}(\mathbf{X}) \approx \mathcal{L}_m(\mathbf{X}) := \frac{1}{m}\sum_{k=1}^m{l(\mathbf{y^k})}$$

この近似を用いて獲得関数の勾配を求めるには、微分と期待値（積分）の交換が成り立つ必要があります。

- 交換 $\nabla\mathbb{E}[l(\mathbf{y})]=\mathbb{E}[\nabla l(\mathbf{y})]$ の成立条件
  - 被積分関数 $l$ が連続
  - $l^{\prime}$ がほとんど至る所で（そうでない点の測度が0で）存在し、積分可能

この条件はGPのカーネル関数に依存しますが、[2回微分可能なカーネルを使用すれば成立することが示されており](https://arxiv.org/abs/1602.05149)、通常用いるMaternカーネルなどでは問題になりません。
（リンク先の論文の§4.1.で交換可能性が検討されています。）

### 1.2.3. re-parametrization


## 1.3. 非連続性への対応

# 2. 勾配法による最適化
勾配が分かれば、候補点 $\mathbf{X}$ を最適化することができます。
ここではBoTorchでの実装を確認しておきます。

# 3. 貪欲法による逐次最適化
## 3.1. 貪欲法

- 貪欲法 (greedy) : 複数の候補点を得たい場合に、1点ずつ逐次的に決定していく手法のこと。

[memo]
- [元論文2](https://proceedings.neurips.cc/paper/2018/hash/498f2c21688f6451d9f5fd09d53edda7-Abstract.html)では、貪欲法による逐次最適化の方が精度に優れる可能性が指摘されている。
  - 獲得関数の多くが劣モジュラ関数となるため、貪欲法で最適化することで最適解付近に到達できることが示されています。
  - 劣モジュラ関数と最適化については、MLPシリーズの書籍を参照。
    - [『劣モジュラ最適化と機械学習』](https://www.kspub.co.jp/book/detail/1529090.html)

- BoTorchでは[`optimize_Acqf()`](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/optim/optimize.py#L49)で`sequential=True`とすれば実行できる。
  - 内部では`optimize_acqf(q=1)`が再起的に呼ばれ、候補となるDesign Pointを1点ずつ決定していく。
  - 決定したDesign Pointは`X_pending`として保留され、以降は最適化されない。しかし獲得関数値の算出時には`forward()`に入力される。
    - `X_pending`は[concatenate_pending_pointsデコレータ](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/utils/transforms.py#L248)によって、獲得関数への入力に結合されます。
    - 獲得関数の計算時には、デコレータを除いて変化はありません。たとえば[qExpectedImprovementの場合](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/acquisition/monte_carlo.py#L144)

```py:acquisition/monte_carlo.py
    def forward(self, X: Tensor) -> Tensor:
        ### (中略)
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        return q_ei

```
  - これは貪欲法$$x_j = \argmax_{x}{\mathcal{L}(\mathbf{X}_{<j} \cup \{x\})}$$の実装

- MES (Max-value Entropy Search)ではfantasize()を使用している。
  - [BoTorchのfantasize](https://github.com/pytorch/botorch/blob/v0.6.0/botorch/models/model.py#L132)は、固定した暫定Design Pointにおける事後分布からサンプリングし、各サンプルを加えたGPを作成している。

# 終わりに
BoTorchの基本を解説した前回に続き、BoTorchの中核を成すMonte Carlo獲得関数とその使用法について説明しました。私の勘違いや理解の甘い点などがありましたら、コメント等いただければ幸いです。
さらに続けて、BoTorchに実装されている新しい獲得関数や関係する手法について記事を書ければと考えております。

# 参考文献
- 論文
  - [The Reparameterization Trick for Acquisition Functions](https://arxiv.org/abs/1712.00424)
  - [Maximizing acquisition functions for Bayesian optimization](https://proceedings.neurips.cc/paper/2018/hash/498f2c21688f6451d9f5fd09d53edda7-Abstract.html)
  - [Parallel Bayesian Global Optimization of Expensive Functions](https://arxiv.org/abs/1602.05149)
- 書籍
  - [劣モジュラ最適化と機械学習](https://www.kspub.co.jp/book/detail/1529090.html)
  - [ガウス過程と機械学習](https://www.kspub.co.jp/book/detail/1529267.html)
  - [Bayesian Optimization and Data Science](https://www.amazon.co.jp/Bayesian-Optimization-Data-Science-SpringerBriefs/dp/3030244938)
