---
title: OptunaのTPESamplerを読む
tags: Optuna hyperparameter 機械学習
author: narrowlyapplicable
slide: false
---
# Outline
1. TPEのアルゴリズム概略
2. データの分割
3. カーネル密度推定（KDE）の実行
4. 候補点サンプリングおよび評価
5. 実行時の流れ
6. おまけ：TPESamplerの引数の意味

# イントロ
[Optuna](https://optuna.readthedocs.io/en/stable/)の最適化部分は[optuna.samplers](https://optuna.readthedocs.io/en/stable/reference/samplers.html)に実装されていますが、その中でも基幹となるのが`TPESampler`です。ガウス過程ベースのアルゴリズムより計算量が軽く、カテゴリ変数も扱えるなど非常に便利な点があります。  
しかし…  

```py
class optuna.samplers.TPESampler(consider_prior=True, prior_weight=1.0, consider_magic_clip=True, consider_endpoints=False, n_startup_trials=10, n_ei_candidates=24, gamma=<function default_gamma>, weights=<function default_weights>, seed=None, *, multivariate=False, warn_independent_sampling=True)
```
  
ドキュメントだけだと、引数の意味がイマイチ腑に落ちません。元論文に"Prior"とかあったか？magic_clipって何？…  
ということで、Optunaが何をやっているのかキチンと把握するためには`TPESampler`の実装を読むしかなさそうです。  
  
なお、この記事は[Optuna2.7.0](https://github.com/optuna/optuna/tree/v2.7.0)を元にしています。執筆中にOptuna2.8.0が出てしまいましたが、ここでは一部実装が整理されたようです。特に候補点サンプリングの機能（後述）がsampler.pyからparzen_estimator.pyに移動しています。しかし根本部分は変わっていないので、この記事と対応づけることは容易かと思います。また文中のコードは[Optuna2.7.0](https://github.com/optuna/optuna/tree/v2.7.0)からの引用であり、コード引用部にあるコメントは筆者が付したものです。  

# 0. TPE（Tree-Structured Parzen Estimator）の概略
## 0.1. 元論文
- [Algorithms for Hyper-Parameter Optimization](https://hal.inria.fr/hal-00642998/)  
  TPEの骨子はほぼこの論文のままです。さらにOptunaのドキュメントでは、もう一つこの論文もリファレンスに挙げられています。  
- [Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures](http://proceedings.mlr.press/v28/bergstra13.pdf)  
  この論文の実験パートにおいてTPEの改善が示されており、Optuna実装でも採用されています。しかしどちらの論文にもないテクニックも存在しているような...  

## 0.2. アルゴリズムの概略
ガウス過程が直接$p(y|x)$すなわち「入力変数$x$の下での出力変数$y$の分布」を作るのに対し、TPEはまず$p(x|y)$を作ります（※機械学習モデルのチューニングであれば、入力変数$x$が学習モデルのハイパーパラメータ、出力変数$y$が推定精度に相当します）。その後ベイズルールにより$p(y|x)=p(x|y)p(y) / p(x)$ として、獲得関数の値を計算します。その手順は大きく3段階に分けられます。  
  
### (1). データの分割
   測定したデータ$\{(x_i, y_i)\}_i^k$を、**出力変数$y$の値で2つに分割**します。
   閾値$y^* $ を $\gamma=p(y<y^*)$となるように決め、この閾値を下回るものを$L=\{(x_i^L, y_i^L)\}$, 上回るものを$G=\{(x_i^G, y_i^G)\}$ とします。最小化問題を想定して、Lを上位群、Gを下位群と呼ぶことにします。  
   
### (2). カーネル密度推定（KDE）の実行
   上記の分割を踏まえて、 $p(x|y)$ を作成します。$g(x), l(x)$をそれぞれ分割したデータセット$G, L$から、**カーネル密度推定で求めた入力$x$の分布**として、下式のように定義します。
   
   ```math
   p(x|y) = \left\{
    \begin{array}{ll}
    g(x) & (y \geq y^*) \\
    l(x) & (y \lt y^*)
    \end{array}
    \right.
   ```
   ただしこのカーネル密度推定は $x$ の各次元で独立に行われます。

  
### (3). 候補点サンプリングおよび評価  
   探索空間上全ての点$x$に対して獲得関数を求めることはできないため、サンプリングにより候補点を作成します。ガウス過程によるベイズ最適化では[CMA-ES](https://arxiv.org/abs/1604.00772)などを用いて候補点を決めていましたが、TPEは**$p(x|y)$を作成しているため、直接 $x$ をサンプリングできます**。$p(x|y)$からサンプリングした候補点 $x^+$に対して獲得関数値を計算し、その値が最大となる候補点を最終的な提案とします。  
   獲得関数としてはEI（Expected Improvement）を用います。  

   $$ EI(x^+) = \int{\max(0, y^*-y) p(y|x^+)dy} $$  

   TPEでは、このEIを単純な形に変形できます。
   $$EI(x^+) \propto \left(\gamma + \frac{g(x^+)}{l(x^+)} (1-\gamma) \right)^{-1}$$
   すなわち、**EIを最大化するには $l(x^+) / g(x~^+)$ を最大化する $x^+$ を見つければ良い**ということです。  


# 1. データの分割
TPESamplerの実装はoptuna/samplers/_tpe/下の[sampler.py](https://github.com/optuna/optuna/blob/v2.7.0/optuna/samplers/_tpe/sampler.py)にあります。  

```py:samplers/_tpe/sampler.py
class TPESampler(BaseSampler):
    def __init__(
        self,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = False,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma: Callable[[int], int] = default_gamma,
        weights: Callable[[int], np.ndarray] = default_weights,
        seed: Optional[int] = None,
        *,
        multivariate: bool = False,
        warn_independent_sampling: bool = True,
    ) -> None:
```
## 1.1. 分割比率の決定
データの分割比率（正確には上位群のサンプルサイズ）を決めるのは引数`gamma`で、そのデフォルト値は`default_gamma`関数です。  

```py:samplers/_tpe/sampler.py
def default_gamma(x: int) -> int:
    return min(int(np.ceil(0.1 * x)), 25)
```
**上位群のサンプルサイズを $\min(0.1x, 25)$ と指定しています**。ここで$x$はサンプルサイズを意味しています。つまりデータが少ない時は１割を上位群に割り当てつつ、25個以上にはしない方針が採られています。  
かつてはHyperoptの設定を引き継いでいたようで、`hyperopt_default_gamma`という関数も残されています。  

```py:samplers/_tpe/sampler.py
def hyperopt_default_gamma(x: int) -> int:
    return min(int(np.ceil(0.25 * np.sqrt(x))), 25)
```

$\min(0.25\sqrt{x}, 25)$ と設定しており、こちらの方に馴染みがある方も多いかもしれません。しかし現在のOptunaでは、この設定はデフォルトから外されています。  

## 1.2. 分割実行
データをスコア（出力変数）順に並び替え、上位群／下位群に分割します。この処理は`TPESampler`内の`_split_observation_pairs`関数が担当します。  

```py:samplers/_tpe/sampler.py
    def _split_observation_pairs(
        self, config_vals: List[Optional[float]], loss_vals: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # ...[中略]
        config_values = {k: np.asarray(v, dtype=float) for k, v in config_vals.items()} #観測値の辞書を変換
        loss_values = np.asarray(loss_vals, dtype=[("step", float), ("score", float)]) # lossの辞書を変換

        n_below = self._gamma(len(config_values)) # 上位群のサンプルサイズ決定（default_gamma実行）
        loss_ascending = np.argsort(loss_values) # loss順にargsortしindex取得
        below = config_values[np.sort(loss_ascending[:n_below])] #上位群のindexをソートし、対応する値を取得
        below = np.asarray([v for v in below if v is not None], dtype=float) # numpy化
        above = config_values[np.sort(loss_ascending[n_below:])]　#下位群のindexをソートし、対応する値を取得
        above = np.asarray([v for v in above if v is not None], dtype=float) # numpy化
        return below, above
```
デフォルトでは`self._gamma=default_gamma`であり、ここで上記の`default_gamma`が呼び出されています。

ここでloss順のindex配列`loss_ascending`を直接使わず、sortしてから使用していることに注意が必要です。上位群では`[np.sort(loss_ascending[:n_below])]`としているため、作成された上位群配列は元のindex順＝**データの取得順に再配置されている**ことになります。この理由は、公式ドキュメント[2本目の参照論文]((http://proceedings.mlr.press/v28/bergstra13.pdf))の記述にあります。  
> The first modification was to down-weight trials as they age so that old results do not count for as much as more recent ones. We gave full weight to the most recent 25 trials and applied a linear ramp from 0 to 1.0 to older trials.   

すなわち、カーネル密度推定時に古いデータへの重みを減衰させる処理を行うため、データは取得順に並べ替えられる必要があります。  

# 2. カーネル密度推定（KDE）の実行
データを上位群／下位群に分割したら、双方についてKDEを実行します。**TPEでは通常入力変数の各次元に独立性を仮定する**ため、その分布をモデリングする **KDEは各次元別々に行います**。したがって、実装も1次元変数のみを前提としたカーネル密度推定を実行しています。   
直接的には`suggest_float()`などを実行することで`TPESampler`内で`sample_independent() -> _sample_uniform()等 -> _sample_numerical()`が呼び出され、その内部で`_ParzenEstimator`が作成・実行されます。`_ParzenEstimator`の実装はoptuna/samplers/_tpe/下の[parzen_estimator.py](https://github.com/optuna/optuna/blob/v2.7.0/optuna/samplers/_tpe/parzen_estimator.py)にあります。  

## 2.1. 各カーネルの位置
KDEのカーネルとして、**データセットの各観測点を平均としたガウス分布**を用います。またデフォルト設定`consider_prior=True`では、入力変数の**定義域（指定した探索範囲）の中心にカーネル＝ガウス分布を一つ追加**します。ドキュメントでは"Enhance the stability of Parzen estimator by imposing a Gaussian prior"、すなわち安定性を向上するための**Prior**を追加しているのだとされています。  

この処理は`_ParzenEstimator`の`__init__()`内で呼び出される`_culculate()`が担当します。同関数内では`consider_prior`引数（意味は後述）で処理が分岐しますが、デフォルト`True`の場合を記載します。

```py:samplers/_tpe/parzen_estimator.py
    @classmethod
    def _calculate(
        cls,
        mus: ndarray, # 観測点の値
        low: float, # 定義域のlow側
        high: float, # 定義域のHigh側
        consider_prior: bool, # Priorを使うかどうか（デフォルト設定だとTrueが入る）
        prior_weight: Optional[float], # Priorに与える重み
        consider_magic_clip: bool,# カーネル幅決定時（§2.3）に使用, カーネル幅に"magical_clip"を行うか
        consider_endpoints: bool, # カーネル幅決定時（§2.3）に使用, 定義域を考慮した縮小の可否
        weights_func: Callable[[int], ndarray], # カーネルへの重みを定義する関数（§2.2で説明）
    ) -> Tuple[ndarray, ndarray, ndarray]:
    # ...[中略]
            prior_mu = 0.5 * (low + high) # Priorカーネルの位置＝定義域の中心を決定
            # ...[中略]
            else:  # サンプルサイズが0でなければこちら
                order = numpy.argsort(mus) # musをソートした後の並び順（§2.2で使用）
                ordered_mus = mus[order]
                prior_pos = int(numpy.searchsorted(ordered_mus, prior_mu)) # muの配列におけるPriorのindex決定

                low_sorted_mus_high = numpy.zeros(len(mus) + 3) # 両端とPriorで3つ増加（§2.3で使用）
                sorted_mus = low_sorted_mus_high[1:-1]
                sorted_mus[:prior_pos] = ordered_mus[:prior_pos] # Priorより左側のカーネル位置
                sorted_mus[prior_pos] = prior_mu # Priorの位置
                sorted_mus[prior_pos + 1 :] = ordered_mus[prior_pos:] # Priorより右側のカーネル位置
```
Priorカーネルを割り込ませるために面倒なことになっていますが、それ以外では与えられた観測値`mus`をソートしているだけです。先にソートせずindex配列`order`を作成しているのは、後で重みを与える際（§2.2）に対応付けるためです。

## 2.2. 各カーネルの重み
カーネルに与える重みは`TPESampler`の引数`weights`に与えられた関数で定義されます。デフォルトは同じ[sampler.py](https://github.com/optuna/optuna/tree/v2.7.0/optuna/samplers/_tpe)に定義されている`default_gamma()`を使います。  

```py:samplers/_tpe/sampler.py
def default_weights(x: int) -> np.ndarray:

    if x == 0:
        return np.asarray([])
    elif x < 25:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat], axis=0)
```
§1.2で引用した論文の通り、**新しいデータに1.0の重み`flat`を与え、古いデータに対応するカーネルにはランプ関数状に減衰させた重み`ramp`を与えています**。（上述の通り、データは上位群／下位群ともに取得順を保存した並びになっているため、前半＝古いデータに減衰した重みを割り当てています。）   

この重みを実際に与える部分は前節同様`_ParzenEstimator`の`_culculate()`が担当します。  

```py:samplers/_tpe/parzen_estimator.py
        unsorted_weights = weights_func(mus.size) # 上述のdefault_weights実行
        if consider_prior: # Priorを入れる場合はこちら
            sorted_weights = numpy.zeros_like(sorted_mus)
            sorted_weights[:prior_pos] = unsorted_weights[order[:prior_pos]]
            sorted_weights[prior_pos] = prior_weight
            sorted_weights[prior_pos + 1 :] = unsorted_weights[order[prior_pos:]]
        else:
            sorted_weights = unsorted_weights[order] # Priorを入れない場合はそのまま
        sorted_weights /= sorted_weights.sum() # 総和を1に規格化
```
Priorを割り込ませている他は特別な処理は行わず、default_weightsで作った重み`unsorted_weights`がそのまま与えられています。ただしカーネル位置`mu`がソートされた関係上、並びを`mu`の並び順`order`（前節参照）に対応させています。

## 2.3. 各カーネルのバンド幅
カーネルの幅には、基本的に**隣接する観測値との距離**をそのまま用います。ただし両端のカーネルを除けば隣接する観測値は両側にあるので、その距離のうち大きい方を採用します。
また両端のカーネルについては、定義域の端との距離も取り、隣接カーネルとの距離と比較して大きい方がカーネル幅に採用されます。これは定義域を考慮した縮小と見なせます。

```py:samplers/_tpe/parzen_estimator.py
            prior_sigma = 1.0 * (high - low) # Priorカーネルは定義域全体にまたがる
        ### ...[中略]    
        # low_sorted_mus_highは§2.1の引用部で定義, 両端以外sorted_musと同じ＝観測値が入っている
        if mus.size > 0:
            low_sorted_mus_high[-1] = high # 定義域最大側（両端カーネルでの比較用）
            low_sorted_mus_high[0] = low # 定義域最小側（両端カーネルでの比較用）
            sigma = numpy.maximum( # カーネルバンド幅決定
                low_sorted_mus_high[1:-1] - low_sorted_mus_high[0:-2],
                low_sorted_mus_high[2:] - low_sorted_mus_high[1:-1],
            ) # 両側との距離を取りmaximum（両端カーネルでは定義域終端との距離も取りmaximum）
            if not consider_endpoints and low_sorted_mus_high.size > 2:
                sigma[0] = low_sorted_mus_high[2] - low_sorted_mus_high[1]
                sigma[-1] = low_sorted_mus_high[-2] - low_sorted_mus_high[-3]
```
最後の３行について補足します。`consider_endpoints=False`（デフォルト設定）の場合、両端カーネルの幅は隣接カーネル（両端より一つ内側のカーネル）の幅と同一に設定し、定義域を考慮した縮小は行わないよう変更を加えています。  

さらにOptunaでは"magical_clip"と呼ばれる処理を加えています。

```py:samplers/_tpe/parzen_estimator.py
        maxsigma = 1.0 * (high - low)
        if consider_magic_clip: # デフォルトはこっち
            minsigma = 1.0 * (high - low) / min(100.0, (1.0 + len(sorted_mus)))
        else:
            minsigma = EPS
        sigma = numpy.asarray(numpy.clip(sigma, minsigma, maxsigma))
```
文字通り、決定したカーネル幅`sigma`にclipしています。maxは定義域からはみ出さないようclipする処理ですが、min側は $\frac{high-low}{\min(100, 1+N)}$（Nはサンプルサイズ）と黒魔術めいています。おそらく経験的に決められたものと思われます。

# 3. 候補点サンプリングおよび評価
いよいよ提案部分に移ります。  
`suggest_float()`などを実行すると、`TPESampler`内の`sample_independent() -> _sample_uniform()等 -> _sample_numerical()`が呼び出されます。 `_sample_numerical()`内では候補となる入力変数値を上位群のKDE結果 $p(x|y)=l(x)$ からサンプリングし、得られた各候補値における尤度比を計算します。これはEI関数値が
 $$\left(\gamma + \frac{g(x^+)}{l(x^+)} (1-\gamma) \right)^{-1}$$ 
に比例することから、EI最大の候補点を知るには尤度比 $l(x^+)/g(x^+)$ が分かれば十分であるためです。`_sample_numerical()`は、この尤度比が最大になる点を返します。
実装は以下の通りです。**上位群のKDE結果を混合正規分布と見做してサンプリング**し、得られた**候補点について$g(x), l(x)$双方で対数尤度を計算**しています。最後に対数尤度の差を取り、これを`exp`で目的の尤度比に復元しています。
  
```py:samplers/_tpe/sampler.py
    def _sample_numerical(
        self,
        low: float, # 定義域下限
        high: float, # 定義域上限
        below: np.ndarray, # 上位群（ロスが小さい方）のデータ
        above: np.ndarray, # 下位群（ロスが大きい方）のデータ
        q: Optional[float] = None, 
        is_log: bool = False, # 対数変換の有無
    ) -> float:
        # ...[中略] (logで変数変換された変数への対処)
        size = (self._n_ei_candidates,) # サンプリングする候補点の数＝TPESamplerの引数_n_ei_candidates
        # まず上位群（ロスが小さい方）の処理
        parzen_estimator_below = _ParzenEstimator( # 上位群KDE実行
            mus=below, low=low, high=high, parameters=self._parzen_estimator_parameters
        )
        samples_below = self._sample_from_gmm( # KDE結果を混合正規分布とみて候補点サンプリング
            parzen_estimator=parzen_estimator_below, low=low, high=high, q=q, size=size
        )
        log_likelihoods_below = self._gmm_log_pdf( # EI値計算のため、samples_belowの対数尤度を計算しておく
            samples=samples_below,
            parzen_estimator=parzen_estimator_below,
            low=low,
            high=high,
            q=q,
        )
        # 下位群に関してもKDEと対数尤度計算（ただし候補点は上位群からサンプリングしたsamples_belowを使用）
        parzen_estimator_above = _ParzenEstimator( # 下位群KDE実行
            mus=above, low=low, high=high, parameters=self._parzen_estimator_parameters
        )

        log_likelihoods_above = self._gmm_log_pdf( # 上位群のサンプルsamples_belowの対数尤度を下位群でも計算
            samples=samples_below,
            parzen_estimator=parzen_estimator_above,
            low=low,
            high=high,
            q=q,
        )

        ret = float(
            TPESampler._compare( # 対数尤度の差を取り、最大のものを返す
                samples=samples_below, log_l=log_likelihoods_below, log_g=log_likelihoods_above
            )[0]
        )
        return math.exp(ret) if is_log else ret # expで対数尤度の差->尤度比に復元
```
ここの処理は明瞭で、説明の必要はないでしょう。

# 実行時の流れ
最後に、Optuna使用時において上記の処理がどのように呼び出されるのか、簡単に確認しておきます。ここでは処理が追いやすい ~~（というか私がそれしか使ってない）~~ "Ask-and-Tell"インターフェースを前提とします。

```py:example.py
study = optuna.create_study()
for _ in range(n_trials):
    trial = study.ask()
    param = trial.suggest_uniform("param", low, high)
```
また引数等は全てデフォルトのままとします。  

## TPESamplerの呼び出しまで
`study = optuna.create_study()`はStudyクラスのインスタンスを作成しますが、このインスタンスは内部に`study.sampler`を持ちます。どのsamplerを使うか`create_study`の`sampler`引数で指定できますが、指定しない（デフォルトの`None`のままの）場合、通常`TPESampler`が、多目的最適化の場合は`NSGAIISampler`と指定されます。
この処理は2箇所に分かれていますが、いずれもoptunaフォルダ直下の [study.py](https://github.com/optuna/optuna/blob/v2.7.0/optuna/study.py) にあります。
まず多目的の場合には、`create_study`内で直接指定されます。

```py:study.py
def create_study(
    storage: Optional[Union[str, storages.BaseStorage]] = None,
    sampler: Optional["samplers.BaseSampler"] = None,
    #...[中略]
) -> Study:
   # ... [中略]
   if sampler is None and len(direction_objects) > 1:
      sampler = samplers.NSGAIISampler() # NSGAⅡを指定
   study_name = storage.get_study_name_from_id(study_id)
   study = Study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)
```

そして目的変数が単一の場合は `Study`クラスの`__init__()`で指定されます。

```py:study.py
class Study(BaseStudy):
   # ...[中略]
   def __init__(
        self,
        study_name: str,
        storage: Union[str, storages.BaseStorage],
        sampler: Optional["samplers.BaseSampler"] = None,
        pruner: Optional[pruners.BasePruner] = None,
    ) -> None:
      # ...[中略]
      self.sampler = sampler or samplers.TPESampler() # デフォルト値NoneならTPESamplerを選択
```
Pythonの文法上、`None`（samplerのデフォルト値）と `or` を取れば必ず他方が選ばれるため、事実上`study.sampler = TPEsampler` がデフォルト値ということになります。

## sample.independence()の実行
続く`study.ask()`は新しい`trial`を作成するために実行します。ここで作られるのは `Trial` のインスタンスであり、その実装は[trial/_trial.py](https://github.com/optuna/optuna/blob/v2.7.0/optuna/trial/_trial.py)にあります。
この`trial`を使って新しいパラメータ候補を作成するため`trial.suggest_....()`を実行します。作成するパラメータの種類に応じて`suggest_uniform`, `suggest_int`などがありますが、内部で`_suggest()`を呼び出している点は同じです。

```py:trial/_trial.py
def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
   ### [中略]
            if self._is_fixed_param(name, distribution): # 固定パラメータ
                param_value = storage.get_trial_system_attrs(trial_id)["fixed_params"][name]
            elif distribution.single():
                param_value = distributions._get_single_value(distribution)
            elif self._is_relative_param(name, distribution): # multivariateモード時
                param_value = self.relative_params[name]
            else: # デフォルト（TPEでは各次元の独立性を仮定）
                study = pruners._filter_study(self.study, trial)
                param_value = self.study.sampler.sample_independent(
                    study, trial, name, distribution
                )

            param_value_in_internal_repr = distribution.to_internal_repr(param_value)
            storage.set_trial_param(trial_id, name, param_value_in_internal_repr, distribution)

        return param_value
```
複雑な条件分岐がありますが、特に指定がない場合はelse以下の部分で`self.study.sampler.sample_independent`が呼ばれます。ここで`sampler`のデフォルトはTPESamplerでしたから、上記の処理によってTPESamplerによる提案がなされることになります。

# おまけ：TPESamplerの引数の意味
TPESamplerが持つ引数の意味を確認しておきましょう。

- consider_prior=True  
  Priorカーネル（§2参照）を使用するか否かを指定します。
- prior_weight=1.0  
  Priorカーネルにかける重み（§2.2参照）を指定します。
- consider_magic_clip=True  
  magical_clip（§2.3参照）を使用するか否かを指定します。
- consider_endpoints=False  
  両端カーネル幅を定義域に基づき縮小する（§2.3参照）か否かを指定します。
- n_startup_trials=10  
  サンプルサイズが小さいとき、TPEを使わず候補点をランダムに決めます。その処理を使う閾値となるサンプルサイズを指定します。
- n_ei_candidates=24  
  TPEにおいてEI値を算出する候補点数（§3参照）を指定します。
- gamma=\<function default_gamma>  
  上位群がデータセットに占める比率（§1.1参照）を決める関数を与えます。
- weights=\<function default_weights>  
  KDEカーネルの重み（§2.2参照）を決める関数を与えます。
- seed=None  
  乱数seedを指定します。
- multivariate=False  
  本記事では紹介しませんでしたが、入力変数間の関係を考慮するTPEの改善版（現在experimental feature）が実装されており、その使用可否を指定します。
- warn_independent_sampling=True  
  multivariate = Trueの場合に、入力変数間に独立性を仮定したサンプリングを行った際に警告を発するか指定します。

# 終わりに
TPESamplerの処理を一通り追ってみましたが、ご質問やご指摘などありましたらコメント頂ければ幸いです。
今回は`multivariate=True`としたモードまでは追えませんでしたし、Optunaには多目的ベイズ最適化用のMOTPESamplerもあります。今後この辺りを続編として書いていくつもりです。

