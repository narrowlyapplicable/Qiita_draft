---
title: [BoTorch]Multi-fidelityベイズ最適化の基礎と実装
tags: ベイズ最適化 ガウス過程 BoTorch GPyTorch
author: narrowlyapplicable
slide: false
---

# Outline
1. Multi-fidelity
2. Multi-fidelity Knowledge Gradient
3. BoTorch実装

# Intro
BoTorch関連記事の3本目です。
- 1本目：[ベイズ最適化ツールBoTorch入門](https://qiita.com/narrowlyapplicable/items/d8e9be53f73d6fa5e4d3)
- 2本目：[[BoTorch]Monte Carlo獲得関数とその最適化](https://qiita.com/narrowlyapplicable/items/3c2c80e05e16fa935cf1)

ベイズ最適化をはじめとしたBlack Box最適化は、評価（＝目的関数値の取得）が容易でない対象での最適化に威力を発揮します。シミュレーションであれば計算コストが高い、実験であれば長時間を要する、といった状況です。代表的な適用先である深層学習のハイパーパラメータ調整も、やはり1回の学習に多大な計算コストを要するexpensiveな例です。  

こうした用途では、最適化にかける計算コストや時間（Budgetと呼びます）を抑えたい場合が多くあります。Budgetを抑えながら最適値を探す際に有効な考え方として、Multi-Fidelity(MF)があります。
本記事では、BoTorchに実装されている`qMultiFidelityKnowledgeGradient`獲得関数を例に、Multi-fidelityの基礎を解説することを目的とします。


# 1. Multi-fidelity

## 1.1. fidelity (忠実度)
多くの実験やシミュレーションにおいて、速度・コストは精度とのトレードオフになります。
-  シミュレーションであれば、精度を犠牲にして高速に計算させる事ができるでしょう。
-  化学実験などにおいても、機材の動作や測定の精度を落として早く結果を得られるものがあります。
-  ベイズ最適化の最もメジャーな適用先である機械学習のハイパーパラメータ調整においても同様です。深層ニューラルネット(DNN)であれば、エポック数や使用するデータ量を落とすことで、その性能を粗く高速に見積もる事ができます。  

このように、評価したい系に **「速度と精度のトレードオフ」を引き起こすパラメータ** があるとき、当該パラメータを ***"fidelity"(忠実度)*** と呼びます。  

一般化のため、表記法を導入しておきます。  
目的関数$f(x), x\in\mathbb{A}$を、fidelityを持つ$g(x, s)$に拡張することを考えます。ここで、fidelityパラメータが1次元であれば、$s\in[0,1]$は1が最高・0が最低（逆なこともあります）すなわち
$$f(x) = g(x, 1), \forall x\in\mathbb{A}$$
となります。  
fidelityパラメータが多次元の場合も、

```math
\begin{align}
&f(x) = g(x, \mathbf{1}_m), \forall x \in \mathbb{A}\\
&\mathbf{s} = (s_1, ..., s_m)\\
&s_i \in [0,1]
\end{align}
```
となります。ただし$\mathbf{1}=(1, 1,..., 1\in\mathbb{R}^m)$です。  

- 例として、DNNのハイパーパラメータ調整を考えてみましょう。
  この場合、精度が目的関数$f$に、ハイパーパラメータが入力$x$に当たります。さらにfidelityパラメータ$s$として、学習データ量を変動させる事にします。  
  本来の学習データ全てを使う場合が$s=1$であり、学習データを半分しか使わない場合を$s=1/2$とします。  
  $f(x)=g(x, 1)$の評価には全データを用いた学習が必要であり、相応の計算時間を要します。これに対し$g(x, 1/2)$の評価では、使うデータ量が半分となるため、より短い計算時間で済む事が期待できます。  
  
  このように、fidelityによって精度と計算コストのトレードオフを調整できる訳です。

$g(x, s)$は$f(x)$のより「安価な」近似、すなわち精度を犠牲に速度・コストを優先した評価といえます。この近似を活用することで、最適値の探索をより少ないコストで実施できる可能性があります。

## 1.2. Multi-fidelity
最適化全体にかかる時間ないしコスト（"Budget"と呼びます）を抑える目的で、fidelityパラメータを変動させる手法をMulti-fidelityと呼びます。  

通常のベイズ最適化では、目的関数$f(x)$の最適値を探すため、
次に評価すべき入力点$x_{new}$を提案 → $f(x_{new})$を評価し結果$y_{new}$を得る → ガウス過程などのモデルを更新  
という作業を繰り返します（outer-loop）。
しかし$f(x_{new})$の評価が重すぎる場合、outer-loop全体に要するBudgetが大きくなりすぎる事があります。そこで評価の一部を低fidelityな$g(x_{new}, s), s<1$で代替することで、outer-loop全体のBudget抑制を狙います。これがMulti-fidelityの基本的考え方です。

[図：outerloop]

具体的には、以下のような変更が加わります。
- ガウス過程回帰モデル（GP）による推定対象を、$f(x)$ではなく$g(x, s)$とする。
  - GPの入力変数は$x$ではなく、fidelityが加わった$(x,s)$となる。
  - GPの目的関数は$g:(x, s)\mapsto\mathbb{R}$となる。
  - 入力$x, x^{'}\in\mathbf{A}$間の内積を定義するカーネルに加えて、fidelityに関するカーネルを定める必要がある。
- 評価コストも、別のGPモデルで推定する。
  - 評価コストは入力とfidelityで定まるものとして推定し、$cost(x, s)$などと表す。

Mulit-fidelityを用いたBOは2017年ごろから登場し、すでに多くの手法が存在します。
代表的なものとしては[FABOLAS](https://proceedings.mlr.press/v54/klein17a.html) (2017)や[Freeze-Thaw](https://arxiv.org/abs/1406.3896)(2014)などがあります。

- FABOLAS論文：[Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets](https://proceedings.mlr.press/v54/klein17a.html)
- Freeze-Thaw論文：[Freeze-Thaw Bayesian Optimization](https://arxiv.org/abs/1406.3896)

しかし本記事では、より後発の手法でBoTorchに採用されているtaKGに対象を絞ります。
taKGは性能面での改善の他に、fidelityをエポック数のような連続的なもの（trace fidelity）と、データ数のような非連続的なものに分けることで、1回の試行から最大限の情報を取得できる利点があります。
また前段階であるcfKGがシンプルで説明しやすいため、Multi-fidelityの入門に向いているという利点もあります。
次章において、まずKnowledge Gradient獲得関数を導入し、そのMulti-fidelity拡張としてcfKG, taKGについて説明します。そして§3において、taKGのBoTorch実装であるqMultiFidelityKnowledgeGradientを用いた実装について説明します。

# 2. Multi-fidelity Knowledge Gradient

## 2.1. Knowledge Gradient (KG) 獲得関数

## 2.2. cfKG

## 2.3. taKG

# 3. BoTorch実装

- 獲得関数：qMFKG
- GPモデルも、Multi-fidelity専用のモデルを用いる必要がある。これは入力と別にfidelityについてもカーネルを作る必要があるため。
  - [SingleTaskMultiFidelityGP](https://botorch.org/api/models.html#botorch.models.gp_regression_fidelity.SingleTaskMultiFidelityGP)：通常の`SingleTaskGP`のMulti-fidelity版
  - `_setup_multifidelity_covar_module`でfidelityのカーネルを追加している。