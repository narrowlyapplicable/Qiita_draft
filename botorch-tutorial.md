---
title: ベイズ最適化ツールBoTorch入門
tags: ベイズ最適化 BoTorch GPyTorch ガウス過程
author: narrowlyapplicable
slide: false
---
# Intro
Optunaの記事2本に続いて、ガウス過程によるベイズ最適化ツール[BoTorch](https://botorch.org/)を扱います。

BoTorchはFacebookが開発を主導するベイズ最適化用Pythonライブラリです。ガウス過程部分にはPyTorchを利用した実装である[GPyTorch](https://gpytorch.ai/)を利用していますが、獲得関数や候補点提案などに関する最新の手法をサポートしており、最小限の労力で最新のベイズ最適化を実行できます。特に獲得関数の工夫により、複数点の同時提案や多目的最適化などに対応できる点が特徴です。
通常は同じFacebook主導のパッケージである[Ax](https://ax.dev/)から利用することが想定されていますが、ガウス過程や周辺手法を隠蔽せず細かく調整する場合はBoTorchを直接使用することになります。（Optunaからも一部機能が利用できるようになっています。）

本記事では、BoTorchに実装されている最新手法を学ぶ準備として、BoTorchの基本的な使い方と、使われているアルゴリズムの基本を紹介します。
なおベイズ最適化それ自体やガウス過程については説明しません。以下の資料などを参照してください。

- 佐藤一誠先生のスライド：[ベイズ的最適化(Bayesian Optimization)の入門とその応用](https://www.slideshare.net/issei_sato/bayesian-optimization)
- MLPシリーズの[『ガウス過程と機械学習』](https://www.kspub.co.jp/book/detail/1529267.html)

[2022/01/15追加]

- 松井孝太先生のスライド：[機械学習による統計的実験計画](https://drive.google.com/drive/folders/15uk8GHRd1Xy46zA2EE1yznoshqQ2Ylyp)
    - 研究用途で使うライブラリとしてBoTorchが推薦されています。

# インストール
公式ではconda経由でのインストールが推奨されています（2021.06.30時点）。

```cmd
conda install botorch -c pytorch -c gpytorch
```

チャンネル指定はなくてもconda-forge等からインストールできますが、私が使っているWindows環境ではconflictを起こすことがありました。
またPyTorch, GPyTorchの双方が必要になるので、できるだけ公式チャンネルからインストールした方が無難でしょう。

# Get Start（基本的な使い方）
[公式トップページ](https://botorch.org/)にある"Get Start"からはじめましょう。この例には、BoTorchの最も基本的な利用方法が示されています。まずその全体像を示し、後述の章で関連部分について解説します。

まずBoTorchをインポートし、乱数で仮想データを作成します。

```py:get_start
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

train_X = torch.rand(10, 2)
Y = 1 - torch.norm(train_X - 0.5, dim=-1, keepdim=True)
Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
train_Y = standardize(Y)
```

続いてガウス仮定モデルを定義し、周辺尤度最大化によってカーネルパラメータを推定します。

```py:get_start
gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)
```

- `SingleTaskGP`モデルを作成し、周辺尤度`ExactMarginalLogLikelihood`を対象とした推定を`fit_gpytorch_model`で実行しています。  

作成したGPモデルに、UCB獲得関数を与えています。
BoTorchの利点を生かすため、UCBに代えてモンテカルロ獲得関数を用いることがあります。こちらについては後述の章に述べることとします。

```py:get_start
from botorch.acquisition import UpperConfidenceBound

UCB = UpperConfidenceBound(gp, beta=0.1)
```

最後に、獲得関数に関する最適化を行い、最良の候補点を取得します。

```py:get_start
from botorch.optim import optimize_acqf

bounds = torch.stack([torch.zeros(2), torch.ones(2)])
candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)
candidate  # tensor([0.4887, 0.5063])
```

- この最適化を勾配法で実施できるのがBoTorchの大きな特徴です。
  - この例は目的変数が1つ & 候補点1点の単純なケースのため、特段の工夫なしでも実行できます。ただし多目的や候補点を複数提案する場合（parallel selection）などでは、獲得関数の勾配が計算困難となるため、モンテカルロ獲得関数を使用する必要があります。

この例では、獲得関数をUCB（Upper Confidence Bound）とし、カーネルパラメータを点推定で求めています。また目的変数は1つ、同時に提案する点は1点のみと、BoTorchの強みをあまり用いていない単純な例となっています。
この単純な例をもとにBoTorchの概要を説明しつつ、その強みを生かした拡張を紹介していきます。

# 解説
## 1. ガウス過程
### 1.1. GPモデル
まず初めに、`botorch.models`のモジュールから使用するGPモデルを選択します。

```py:get_start
gp = SingleTaskGP(train_X, train_Y)
```

GPyTorchのガウス過程モデルも用意されていますが、通常はそれをラップしたBoTorchのモデルを使用します。ここではもっとも一般的な[`SingleTaskGP`](https://botorch.org/api/acquisition.html?highlight=singletaskgp)を使用します。

BoTorchには他にも多彩なモデルが用意されています。その一部を下表に示しておきます。

|||
|:---|:---|
|`SingleTaskGP`|基本モデル|
|`FixedNoiseGP`|分散一定のノイズ付加|
|`HeteroskedasticSingleTaskGP`|分散不均一のノイズ付加|
|`MixedSingleTaskGP`|カテゴリ変数が混在|
|`MultiTaskGP`|マルチタスクベイズ最適化|


### 1.2. カーネル
`SingleTaskGP`をはじめとした通常のガウス過程モデルでは、デフォルトでMatern 5/2カーネル + ARD（関連度自動決定）が採用されています。

```py:gp_regression.py
        if covar_module is None:
            self.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=transformed_X.shape[-1],
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
```

#### Maternカーネル
[Maternカーネル](https://docs.gpytorch.ai/en/stable/kernels.html#gpytorch.kernels.MaternKernel)

```math
\begin{align}
k_\nu(\mathbf{x}, \mathbf{x´}) &= \frac{2^{1-\nu}}{\Gamma(\nu)} (\sqrt{2\nu}d)^{\nu} K_{\nu} (\sqrt{2\nu}d) \\
\mathbf{d} &= (\mathbf{x} - \mathbf{x´})^T  \Theta^{-2} (\mathbf{x} - \mathbf{x´})
\end{align}
```

はベイズ最適化において一般的に用いられるカーネルです。（$K_\nu$は修正ベッセル関数）
パラメータ$\nu$によって関数の滑らかさが定まり、$\theta$によってスケールが調整されます。

$\nu$については、通常計算のしやすさから$\nu=3/2, 5/2$などが使用され、BoTorchデフォルトでは$\nu=5/2$が使用されます。

$$k_{5/2}(\mathbf{x}, \mathbf{x´}) = \left(1 + \sqrt{5} \mathbf{d} + \frac{5}{3} \mathbf{d}^2 \right) \exp(-\sqrt{2\nu}\mathbf{d})$$

Gaussianカーネルよりは滑らかさ（微分可能性）に関する過程が弱く、それほど滑らかでない場合も想定していることになります。

残る$\Theta$はlengthscale parameterと呼ばれ、データから推定します。
GPyTorchでは、$\Theta$を対角行列の形に制限することが多いようです。[GPyTorchのドキュメント](https://docs.gpytorch.ai/en/v1.4.2/kernels.html)によれば

- Default: $\Theta$は単位行列
- Single lengthscale: $\Theta$は単位行列の定数倍
- ARD: $\Theta$は各要素別々の単位行列

の3通りの選択肢があり、BoTorchではARDをデフォルトとしています。

#### ARD
ARD（関連度自動決定）はlengthscale parameter $\Theta$を各次元独立（すなわち各要素が異なる対角行列）に設定することで、入力変数の各成分が与える影響の大小がデータから推定されるようにするテクニックです。
出力に与える影響が小さい成分については、対応する$\Theta$の対角成分が小さく推定されるため、結果として変数選択が行われることになります。
ARDを有効にするには、カーネルの`ard_num_dims`引数に成分の数（入力の次元）を与える必要があります。`SingleTaskGP`など大半のGPモデルでは、デフォルトでARDが有効に設定されています。

#### Scaleカーネル
[ScaleKernel](https://docs.gpytorch.ai/en/stable/kernels.html#gpytorch.kernels.ScaleKernel)は、その名の通り出力変数のスケーリングを担当します。実装上はカーネルが定める共分散行列$\mathbf{K}_{orig}$に対して

$$\mathbf{K}{scaled} = \theta_{scale} \mathbf{K}_{orig} $$

と共分散行列を修正します。この $\theta_{scale}$ はoutputscale parameterと呼び、データから推定します。

####　観測ノイズについて
通常ガウス過程では、観測ノイズの存在を考慮しカーネルに$\sigma^2 \mathbf{I}$を加えます（White Noise Kernel）。

$$\mathbf{K}{noise} = \mathbf{K}_{scaled} + \sigma^2 \mathbf{I} .$$

しかし上記のデフォルト設定には、観測ノイズに対応するWhiteNoiseKernelが含まれていません。GPyTorchではWhiteNoiseKernelが廃止されており、観測ノイズは後述する`likelihood`で処理されています。

- 関連する[issue](https://github.com/cornellius-gp/gpytorch/issues/1128)

### 1.3. 事前分布・尤度

#### 事前分布
誤差項分散などのパラメータには、決め打ちの事前分布が設定されています。例えば`SingleTaskGP`であれば、前掲の通りカーネルパラメータにはいずれもガンマ事前分布が

```
lengthscale_prior=GammaPrior(3.0, 6.0),
outputscale_prior=GammaPrior(2.0, 0.15),
```

と設定されていましたし、誤差項分散についても

```py:gp_regression.py
        if likelihood is None:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
```

のように（2,3行目）、ガンマ事前分布`GammaPrior(1.1, 0.05)`が割り当てられています。この操作は`SingleTaskGP`の引数`likelihood`を指定しなかった場合のみ行われるので、事前分布を指定したい場合は`likelihood`ごと与える必要があります。BoTorchの目的はガウス過程モデリングではなくベイズ最適化にあるので、経験的にベイズ最適化に適した設定を採用し、変更することはあまり想定していないようです。（この辺りはOptunaと共通の思想を感じます）

#### 尤度
`likelihood`はその名の通り尤度を指定しますが、通常は上記の通りガウス分布を使用した尤度`GaussianLikelihood`が設定されています。この設定では、ノイズの分散もカーネルパラメータと共にデータから推定されます。

一定の観測ノイズを仮定する場合は、`FixedNoiseGaussianLikelihood`のインスタンスを作り、`SingleTaskGP`の`likelihood`に与えることができます。
しかしより単純に、ノイズを仮定したモデル`FixedNoiseGP`を利用することができます。`FixedNoiseGP`では、デフォルトの`likelihood`に`FixedNoiseGaussianLikelihood`が使用されています。

```py:gp_regression.py
        likelihood = FixedNoiseGaussianLikelihood(
            noise=train_Yvar, batch_shape=self._aug_batch_shape
        )
```

現実的には、BoTorch側で用意されているモデルの中から適したものを選び、`likelihood`を細かく指定することは想定されていないようです。

### 1.4. カーネルパラメータ最適化
ガウス過程の予測分布はカーネル指定によって一意に定まり、いわゆる「学習」の工程は存在しませんが、ハイパーパラメータに相当するカーネルのパラメータを決定する必要があります。
通常のカーネルが持つパラメータは、滑らかさを決めるScale Parameter $\Theta$ と、ScaleKernelが持つoutputscale parameter $\theta_{scale}$ でした。これらをデータから推定します。BoTorchでは`fit_gpytorch_model`が用意されており、勾配法によるカーメルパラメータ推定を実行してくれます。

```py:get_start
gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)
```

`fit_gpytorch_model`では最適化する目的関数を指定する必要があるため、先ほど指定した`likelihood`をに基づく対数周辺尤度を与えています。ここでは通常の対数周辺尤度 `ExactMarginalLogLikelihood`を用いていますが、`gpytorch.mlls`モジュールには他にも変分近似用のELBO`VariationalELBO`やその改善版なども用意されています。

なお簡易的な実行においては上記のように点推定しますが、正確なモデリングを重視する場合はMCMCを用いることが推奨されます。しかし[GPy](https://gpy.readthedocs.io/en/latest/)等とは異なり、BoTorchやGPyTorchには独自のMCMC機能が無いようです（`botorch.sampling`モジュールは、後述するモンテカルロ獲得関数用）。[GpyTorchのドキュメント](https://docs.gpytorch.ai/en/v1.1.1/examples/01_Exact_GPs/GP_Regression_Fully_Bayesian.html)にはPyroを用いた推定方法が示されており、この例のように外部ツールを動員する必要がありそうです。この点はGPyOpt等に比べてデメリットと言えます。

## 2. 獲得関数
BoTorchの獲得関数は２種類あり、解析的獲得関数（Analytic Acquisition Function）と、モンテカルロ獲得関数（Monte-Carlo Acquisition Function）に大別できます。

### 2.1. 解析的獲得関数
獲得関数の値を（近似せず）解析的に算出する実装です。BoTorch上では`Analytic Acquisition Function`と呼ばれています。EI, PI, UCB, 多目的用のEHVIなど基本的な獲得関数は用意されています。
後述のモンテカルロ獲得関数と異なり近似に頼らず獲得関数値を算出しますが、計算困難に陥ることが多く、用途には制限が付きます。例えば、候補点を複数取得する（parallel selection）場合には、後述のモンテカルロ獲得関数を使う必要があります。

```py
from botorch.acquisition import UpperConfidenceBound
UCB = UpperConfidenceBound(gp, beta=0.1)
```

### 2.2. MC（モンテカルロ）獲得関数

このうちモンテカルロ獲得関数（あるいは準モンテカルロ獲得関数）は、深層学習でよく知られた再パラメータ化（reparameterization trick）を応用し、解析的な計算なしで獲得関数値やその勾配の導出を可能にするものです。一部ドキュメントでは「バッチ獲得関数」とも呼ばれているようです。
論文はこちら。

- [The reparameterization trick for acquisition functions](https://arxiv.org/abs/1712.00424)

このモンテカルロ獲得関数の利点により、*多数の候補点を同時にor逐次的に提案*したり（docsでは"batch acquisition functions"と呼称）、*独立でない多次元出力を同時に扱うモデル*を作成することが可能になります。この点こそ、**BoTorchの最大の特徴**です。

[2022/01/14追記]
[MC獲得関数の解析記事](https://qiita.com/narrowlyapplicable/items/3c2c80e05e16fa935cf1)を書きました。詳細はこちらをご覧ください。

`botorch.acquisition`では、元の獲得関数名に`q`がついているものがモンテカルロ獲得関数です。
主要な獲得関数（EI, UCB, EHVIなど）は一通り実装されており、解析的獲得関数と同じように使用できます。

```py
from botorch.acquisition import qUpperConfidenceBound
qUCB = qUpperConfidenceBound(gp, beta=0.1)
```

モンテカルロ獲得関数の計算に際しては、reparametrization trickと準モンテカルロ法を使用しています。
準モンテカルロ部分のサンプラーは、陽に指定することもできます。

```py
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition import qUpperConfidenceBound

sampler = SobolQMCNormalSampler(1024)
qUCB = qUpperConfidenceBound(model, 0.1, sampler)
```

- [公式ドキュメントの該当ページ](https://botorch.org/docs/samplers)

## 3. 最適化
最適な候補点を決めるため、獲得関数の最大化を行います。
通常ガウス過程によるベイズ最適化では、この最適化を勾配法で実行することが困難になる（勾配の計算が困難になる）ことがあり、CMA-ES（進化計算）などを用いて局所解を探索していました。しかしBoTorchはモンテカルロ獲得関数を前提として、勾配法による最大化を行います。

この最大化=候補点の提案は、`botorch.optim.optimize_acqf`関数によって実行されます。（内部的には`scipy.optimize.minimize`を使用しています。）
`botorch.optim.optimize_acqf`の主要な引数はそれぞれ以下の通りです。

- acq_function：最大化すべき獲得関数
- bounds：探索する入力変数の定義域
- q：提案したい候補点のバッチ数
- num_restart：並列実行するオプティマイザの数
  - 獲得関数形状は局所解を多く含むことがあるため、初期値を変えた複数のオプティマイザを使用します。
- raw_samples：初期値探索に用いる乱数の数
- sequential：候補点を逐次的に作成するか否かを決めるbool型フラグ

#### q=1
提案する候補点が1つであれば、獲得関数を解析的に扱うことができます。

```py
UCB = UpperConfidenceBound(gp, beta=0.1)

candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)
```

提案される候補点数は`optimize_acqf`の引数`q`で指定します。したがって、通常のUCB獲得関数を用いる場合は`q=1`である必要があり、`q>1`ではエラーとなります。

#### q>1
提案する候補点が複数の場合は、モンテカルロ獲得関数を使用する必要があります。（解析的獲得関数を使用すると、`optimize_acqf`がエラーを返します）

```py
qUCB = qUpperConfidenceBound(gp, beta=0.1)

candidate, acq_value = optimize_acqf(
    qUCB, bounds=bounds, q=6, num_restarts=5, raw_samples=20,
)
```

逐次的に候補点を作成する場合（非同期候補生成）には、`sequential`引数をTrueにします。

```py
qUCB = qUpperConfidenceBound(gp, beta=0.1)

candidate, acq_value = optimize_acqf(
    qUCB, bounds=bounds, q=6, num_restarts=5, raw_samples=20, sequential=True,
)
```

[BoTorch論文](https://arxiv.org/abs/1910.06403)の§F.2では、逐次的に決定した方が良い結果を得る可能性が示唆されています。
> In practice, the sequential greedy approach often performs well, and may even outperform the joint optimization approach, since it involves a sequence of small, simpler optimization problems, rather than a larger and complex one that is harder to solve.

上記の例では、同時に6個の候補点を取得しました。
データ取得を複数並列で実行できる環境においては、このような形のベイズ最適化（バッチベイズ最適化）が役に立つことがあります。BoTorchでは"batch"という言葉をこの意味で使うことがあります。

- [公式ドキュメントの該当ページ](https://botorch.org/docs/batching)

上記ページにおいて"t-batch"と"q-batch"という二通りの概念が登場しますが、このうち"q-batch"が同時に提案する候補点を指します。"t-batch"は後述の獲得関数最適化において、オプティマイザの並列実行のために用いるものです。
（下記引用：開発者チームの方から頂いたツイート）

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">t-batches are mostly used to make it easy to parallelize random restart optimization, which is critical for effectively optimizing acquisition functions.</p>&mdash; eytan bakshy (@eytan) <a href="https://twitter.com/eytan/status/1410104035328225285?ref_src=twsrc%5Etfw">June 30, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> 

# 終わりに
ごく粗くではありますが、BoTorchの基本機能について解説しました。内容についてご指摘などありましたらコメント頂ければ幸いです。
~~モンテカルロ獲得関数の詳細や、（ここでは紹介しなかった）BoTorchに実装されている諸手法については、後日改めて説明記事をかければと思っています。~~ 
[モンテカルロ獲得関数の解説記事](https://qiita.com/narrowlyapplicable/items/3c2c80e05e16fa935cf1)を書きました！まだ続きます。


