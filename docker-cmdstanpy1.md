---
title: 【備忘録】Docker+CondaでCmdStanPy 1.0環境を作成する方法
tags: ベイズ統計 Stan CmdStanPy Docker
author: narrowlyapplicable
slide: false
---
# 要約
- cmdstanpy v1.0以降は、conda-forgeからインストールすれば必要な設定が全て完了し、すぐに使うことができる。
- しかしDocker内でcondaを使用すると設定が上手くいかず、cmdstanpy使用時にエラーが生じた。
- 試行錯誤したが、結局以前のように`install_cmdstan`を使うしかない。
  - しかも直接実行する`RUN install_cmdstan`は実行できず、pythonから`RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`とするしかなかった。
  - 併せて`apt install build-essential`も必要。condaとの二重インストールになるが、回避手段は見つからなかった。

# 目的
- condaで作成したcmdstanpy環境をDocker化したい。
  - 既にある（cmdstanpyを含む）環境をyamlファイルとして保存し、それを元に引き継げるDocker環境を作りたい。
  - PythonにおけるStan APIの代表格であったPyStanが、v3.0から機能を大きく削減したため、全ての作業環境をcmdstanpyに置き換えたい。
- 以上の目的で作成したDockerfileがcmdstanpy v1.0以降機能しなくなったため、その更新を図った。途中かなり苦戦したため、解決策を備忘録として残しておく。
  - 解決策は単純なものになったが、途上で得られた知識を含めて記録しておく目的で作成した。
  - 単に使いたいだけであれば、以下は読み飛ばして[GitHubリポジトリ](https://github.com/narrowlyapplicable/stanenv-docker/tree/main/cmdstanpy)ににあるファイル群を使えばOK。

*なお、pipによるcmdstanpy単体のDocker環境としては（v1.0以前の記述ですが）[amber_kshzさんの記事](https://qiita.com/amber_kshz/items/172e88e5feda1e7e3133)が参照できる。  これはcmdstanpyのみをインストールしているが、requirements.txtを使うように改造することは容易であろう。ただしcmdstanpy v1.0以降では未検証であり、やはり変更が必要かもしれない。*

# v1.0以前の設定
昨年、Docker+condaでcmdstanpyを含む環境を作るファイルを公開していた。[→ 当時のGitHubコミット内容](https://github.com/narrowlyapplicable/stanenv-docker/tree/f0af86074037eec31f50615389f40454aaab7e6f/cmdstanpy)  
特徴は下記。
- `docker-compose up`コマンド一発で、cmdstanpyを使える環境を構築し、その環境で動くJupyter Labが立ち上がる。  
- `myenv.yaml`に記載したパッケージ一覧を元に、Dockerfile内でconda環境を再現できる。

## myenv.yaml
- `conda env export`で得られる形式のYamlファイル。ここでは必要最小限のパッケージを指定している。
- 使用している環境を再現したい場合は、このファイルを
  `conda env export -n (仮想環境名) > myenv.yaml`
  として差し替えれば良い。

```yaml :myenv.yaml
dependencies:
- python
- numpy
- scipy
- matplotlib
- seaborn
- cmdstanpy
- arviz
- jupyterlab
```

## Dockerfile
  - `conda env update --prune`を使い、`myenv.yaml`で指定したパッケージ群を**base環境に**インストールする方式。
  - その後`install_cmdstan`コマンドでセットアップ

```Dockerfile :Dockerfile
FROM continuumio/miniconda3
WORKDIR /workdir
COPY ./myenv.yaml ./
RUN apt -y update && \
    apt -y install build-essential && \
    conda config --append channels conda-forge && \
    conda env update --prune --name base -f myenv.yaml && \
    conda clean -afy && \
    install_cmdstan

EXPOSE 8888
ENTRYPOINT ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='hogehoge'"]
CMD ["--notebook-dir=/workdir"]
```
- 3行目：`myenv.yaml`を作業用ディレクトリにCOPY
- 4~5行目：aptでbuild-essentialをインストール
  - `install_cmdstan`の実行に必要。というかStan利用時のモデルコンパイル全般にも必要。
- 7行目：`conda env update`コマンドで、base環境を更新している。
  - base環境を使うことで、Docker内で実行困難な`conda activate (仮想環境名)`を回避できる。
  - `-f`オプションにより、指定したファイルに記載されたパッケージをインストールできる。
  - `--prune`オプションにより、既存の環境を消去し上書き=myenv.yamlに記載した環境を再現できる。実質的にbase環境を`conda env create`したのと同じ。
- 9行目： `install_cmdstan`によりcmdstanをセットアップ。cmdstanpyが使用可能になる。
- 11~最終行：ポートを開けJupyter Labを立ち上げている。

## docker-compose.yaml
- Dockerの一番面倒な点（偏見）であるdockerコマンドを書かなくて済むように、docker-compoer upでコンテナの作成と起動を行う。
- ホスト側のコードを使えるよう、カレントディレクトリをバインドマウントしている。
  - Dockerのマウントの概念については右記参照：<https://amateur-engineer-blog.com/docer-compose-volumes/>

```yaml :docker-compose.yaml
version: '3'
services:
  dev:
    build:
      context: .
      dockerfile: Dockerfile
    image: cmdstanpy
    ports:
    - "8080:8888"
    volumes:
    - .:/workdir
```
- services-volumes：ホスト側のカレントディレクトリを、コンテナ側の作業ディレクトリ`workdir`にマウントしている。

### 使い方
- ファイル群のあるディレクトリに移動して`docker-compose up`
  - 初回はコンテナがビルドされ起動する。2回目以降はすぐに起動する。
- ブラウザから`localhost:8080/`を開くと、Jupyter Labが起動する。

## 背景：cmdstanpy v1.0での変更
- condaでのインストール簡易化
  - [公式のドキュメント](https://cmdstanpy.readthedocs.io/en/v1.0.4/installation.html)を参照。conda（のconda-forgeチャネル）からインストールすれば、特にセットアップせずとも利用できるようになった。
  - 以前は `conda install`後に一手間必要だったが、v1.0以降は普通に使えるパッケージになった。
    (手順例)
    ```
    conda create -n stan -c conda-forge cmdstanpy
    conda activate stan
    ```
- conda install後のセットアップが不要になった一方で、Dockerfile内で実行困難な`conda activate`が避けられなくなった。

# 問題点：詰まったポイント
単純にDockerfile内でconda installするだけでは上手くいかない。  

- 例えば次のようにすると、install_cmdstanをしていないと怒られる

  ```Dockerfile :Dockerfile
  FROM continuumio/miniconda3:latest

  WORKDIR /workdir

  COPY ./myenv.yaml ./
  RUN conda config --append channels conda-forge && \
      conda config --remove channels defaults && \
      conda env update --prune --name base -f myenv.yaml && \
  EXPOSE 8888
  ENTRYPOINT ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='hogehoge'"]
  CMD ["--notebook-dir=/workdir"]
  ```

  - condaによるセットアップでは、cmdstanが正しくインストールされていない。

# 試した内容
原因を調べたかったので、割と遠回りに試行錯誤している。  
試した案をまとめると下記の通りとなる。ただし各案を併用したり外したりしながら試行しており、実際には組み合わせぶんの作業量が発生している。  
（その痕跡が[当初のコミット内容](https://github.com/narrowlyapplicable/stanenv-docker/blob/f1c47ddb2f14142a40dd8e1089728c825d8b6c84/cmdstanpy/Dockerfile)に残っている……）

1. condaのチャンネル設定が勝手に変更されないよう、yamlファイルの側で`nodefaults`を設定した。
   - これは`conda install`などのコマンドにおける`-c conda-forge --override-channels`と同じ効果を持ち、defaults含む他チャンネルを使用しないよう強制する。

     - `conda config --remove channels defaults`では不十分で、`conda install -c conda-forge`のようにチャンネルを指定した場合、defaultsが使用され得るという落とし穴が存在する。（なぜこんな仕様に……?）
     - 詳細は[公式ドキュメントの該当部分](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment)を参照。

    ```yaml :myenv.yaml(1)
    name: base
    channels:
      - conda-forge
      - nodefaults ## defaultチャンネルが使用されないよう強制
    dependencies:
      - python
      - numpy
      - scipy
      - matplotlib
      - seaborn
      - cmdstanpy>=1.0.1 ## cmdstanpyをv1.0以降に指定
      - arviz
      - jupyterlab
      - ipywidgets ## cmdstanpy1.0以降では、これがないと結果の表示が崩れる
    ```

   - インストール時の挙動は変わったが、cmdstanpyの問題は解決せず。

2. パスを追加する
   - パス設定の問題であることを疑い、Linuxと同様のパス設定を追加した
   - `ENV PATH /opt/conda/envs/base/bin:$PATH`を追加したが解決せず。
     - パスの通し方：[公式ドキュメント](https://docs.docker.jp/develop/develop-images/dockerfile_best-practices.html#env)

3. `conda activate`, `conda init`
   - 経験上conda install後のconda activateは不要（しなくてもcmdstanpyは使える）が、念の為condaの再起動を試した。
   - そのままのDockerfileでは、`conda activate`はエラーを生じ実行できない。そのため下記の追加が必要。
     - bashの有効化`SHELL ["/bin/bash", "-c"]`
     - `eval "$(conda shell.bash hook)" && \`
       - この解決策は[GitHubのIssue](<https://github.com/ContinuumIO/docker-images/issues/89#issuecomment-887935997>)から発見した。
     - `conda activate base`を実行。
       - うまく行かなかったため、その前に`conda deactivate`や`conda init bash`を追加したが、結果は変わらず。

   ```Dockerfile :Dockerfile(3)
   FROM continuumio/miniconda3:latest
   SHELL ["/bin/bash", "-c"]
   # bashを使う
   WORKDIR /workdir

   # (中略)

   ENV PATH /opt/conda/envs/base/bin:$PATH
   # この書き方は古く、非推奨。`ENV PATH=`にすべき。
   RUN eval "$(conda shell.bash hook)" && \
       conda init bash && \
       conda deactivate && \
       conda activate base
    ```

4. base環境を諦め、別にconda仮想環境を作る
   - conda activateができないため避けていたが、上記手法で回避できるようになったため試行した。
   - しかし問題は解決せず。

   ```Dockerfile :Dockerfile(4)
   FROM continuumio/miniconda3:latest
   SHELL ["/bin/bash", "-c"]
   WORKDIR /workdir

   # (中略)
   RUN conda config --append channels conda-forge && \
       conda config --remove channels defaults && \
       # conda env update --prune --name base -f myenv.yaml && \
       conda env create -f myenv.yaml && \
       conda clean -afy 
   ENV CONDA_DEFAULT_ENV stan
   ENV PATH /opt/conda/envs/stan/bin:$PATH
   RUN eval "$(conda shell.bash hook)" && \
       conda init bash && \
       conda deactivate && \
       conda activate stan
   ```

   ```yaml :myenv.yaml(4)
   name: stan
   channels:
     # (以下略)
   ```

5. `install_cmdstan`を実行する
   - 最終手段として、condaによるセットアップは無視し、上書きする形で再度セットアップさせ他。
   - Dockerfileにおいて`RUN install_cmdstan`を復活させたが、ImportErrorが発生し実行できなかった。
     - シェルスクリプト内でpythonが起動しているようだが、パス設定等に問題があるのか、cmdstanpyをインポートできていなかった模様。詳細は特定できず。
     - エラーメッセージは保存していなかったため省略。
   - 代わりにpythonから実行する方法を採ったところ、成功した。
    `RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`
 

#　最終的な解決策
エレガントではないが、condaでインストールした後に再度install_cmdstanを（Pythonから）実行する。  

```Dockerfile :Dockerfile
FROM continuumio/miniconda3:latest

WORKDIR /workdir

COPY ./myenv.yaml ./
RUN conda config --append channels conda-forge && \
    conda config --remove channels defaults && \
    conda env update --prune --name base -f myenv.yaml && \
    # 統一性のため、myenv.yamlのchannelsには`nodefaults`を追加
    # 参照<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment>
    conda clean -afy 
# 本来conda installのみでセットアップされるはずだが、Dockerでは上手くいかなかったためinstall_cmdstanを実行
# 通常の`install_cmdstan`ではImportErrorを発するため、Python経由で実行した
RUN apt -y update && \
    apt -y install build-essential && \
    python -c 'import cmdstanpy; cmdstanpy.install_cmdstan(overwrite=True)'

EXPOSE 8888
ENTRYPOINT ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='hogehoge'"]
CMD ["--notebook-dir=/workdir"]
```

- 16行目：前述の通り`RUN install_cmdstan`がImportErrorとなるため、pythonから実行している。
  - 試行錯誤の名残で`install_cmdstanpy()`に引数`overwrite=True`を与えているが、おそらく不要。condaでセットアップされたcmdstanがあっても、それを上書きする意図で設定したもの。

myenv.yamlに変更はなし。  
- [GitHubにおいているファイル](https://github.com/narrowlyapplicable/stanenv-docker/blob/911a54ea09d1f6eaa4a199c543c0c4be3fb8c0cc/cmdstanpy/myenv.yaml)では、condaによるsolve environmentを短縮するため、バージョンをある程度細かく指定している。

以上の設定により、cmdstanpyを動かすことができた。[`demo`フォルダにあるJupyter Notebook](https://github.com/narrowlyapplicable/stanenv-docker/blob/911a54ea09d1f6eaa4a199c543c0c4be3fb8c0cc/cmdstanpy/demo/hello-world.ipynb)から試すことができる。
ただし幾つかの懸念事項が残った。  
- Stanモデルのコンパイル用パッケージがcondaとaptで二重にインストールされるため、コンテナの容量がやや大きくなってしまう難点が残った。
  - conda側でセットアップされたものを使用できるのが最善。しかし原因を突き止められなかった。
- 作成後の環境を`conda env export`すると、チャンネルに除外したはずの`defaults`が現れてしまった。
  - pipが使用されたため（下記）だろうか？
  - exportしたファイルを元に環境を再現したい場合は、channelsから`defaults`を除き、`nodefaults`に代える必要がある。

```yaml :frozen.yaml
name: base
channels:
  - defaults
  - conda-forge
dependencies:
  # (中略)
 - pip:
    - pyqt5-sip==12.9.0
prefix: /opt/conda
```

# まとめ

- cmdstanpy v1.0以降を含む環境をDocker+Condaで作成できた。
- 試行錯誤の副産物として
  - Dockerfile内でconda activateする方法
  - condaのチャンネル設定の不確実性（意図せず`defaults`が使われる点）と、設定を強制する方法（`--override-channels`や`nodefaults`）
