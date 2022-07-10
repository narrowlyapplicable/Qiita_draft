## 要約
- cmdstanpy v1.0以降は、conda-forgeからインストールすれば必要な設定が全て完了し、すぐに使うことができる。
- しかしDocker内でcondaを使用すると設定が上手くいかず、cmdstanpy使用時にエラーが生じた。
- 試行錯誤したが、結局以前のように`install_cmdstan`を使うしかない。
  - しかも直接実行する`RUN install_cmdstan`は実行できず、pythonから`RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`とするしかなかった。
  - 併せて`apt install build-essential`も必要。condaとの二重インストールになるが、回避手段は見つからなかった。

## 目的
- condaで作成したcmdstanpy環境をDocker化したい。
  - 既にある（cmdstanpyを含む）環境をyamlファイルとして保存し、それを元に引き継げるDocker環境を作りたい。
  - PythonにおけるStan APIの代表格であったPyStanが、v3.0から機能を大きく削減したため、全ての作業環境をcmdstanpyに置き換えたい。
- 以上の目的で作成したDockerfileがcmdstanpy v1.0以降機能しなくなったため、その更新を図った。途中かなり苦戦したため、解決策を備忘録として残しておく。

*なお、pipによるcmdstanpy単体のDocker環境としては（v1.0以前の記述ですが）[amber_kshzさんの記事](https://qiita.com/amber_kshz/items/172e88e5feda1e7e3133)が参照できる。  これはcmdstanpyのみをインストールしているが、requirements.txtを使うように改造することは容易であろう。ただしcmdstanpy v1.0以降では未検証であり、やはり変更が必要かもしれない。*

## v1.0以前の設定
昨年、Docker+condaでcmdstanpyを含む環境を作るファイルを公開していた。[→ 当時のGitHubコミット内容](https://github.com/narrowlyapplicable/stanenv-docker/tree/f0af86074037eec31f50615389f40454aaab7e6f/cmdstanpy)  
特徴は下記。
- `docker-compose up`コマンド一発で、cmdstanpyを使える環境を構築し、その環境で動くJupyter Labが立ち上がる。  
- `myenv.yaml`に記載したパッケージ一覧を元に、Dockerfile内でconda環境を再現できる。

### myenv.yaml
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

### Dockerfile
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

### docker-compose.yaml
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

## 背景：cmdstanpy v1.0での変更
- condaでのインストール簡易化
  - 公式のドキュメント参照。conda（のconda-forgeチャネル）からインストールすれば、特にセットアップせずとも利用できるようになった。
  - 以前は `conda install`後に一手間必要だったが、v1.0以降は普通に使えるパッケージになった。
    (手順例)
  - Dockerfileを簡素化できるかも。

## 問題点：詰まったポイント
しかしDockerでは上手くいかない
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


  - condaによるセットアップされるはずが、cmdstanが正しくインストールされていない。

## 試した内容
原因を調べたかったので、割と遠回りに試行錯誤している。  
試した案をまとめると下記の通りとなる。ただし各案を併用したり外したりしながら試行しており、実際には組み合わせぶんの作業量が発生している。  
（その痕跡が[当初のコミット内容](https://github.com/narrowlyapplicable/stanenv-docker/blob/f1c47ddb2f14142a40dd8e1089728c825d8b6c84/cmdstanpy/Dockerfile)に残っている……）

1. condaのチャンネル設定が勝手に変更されないよう、yamlファイルの側で`nodefaults`を設定した。
   - これは`conda install`などのコマンドにおける`-c conda-forge --override-channels`と同じ効果を持ち、defaults含む他チャンネルを使用しないよう強制する。

     - `conda config --remove channels defaults`では不十分で、`conda install -c conda-forge`のようにチャンネルを指定した場合、defaultsが使用され得るという落とし穴が存在する。（なぜこんな仕様に……?）
     - 詳細は[公式ドキュメントの該当部分](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment)を参照。

    ```yaml :myenv.yaml
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
      - ipywidgets
    ```

   - インストール時の挙動は変わったが、cmdstanpyの問題は解決せず。

2. パスを追加する
   - パス設定の問題であることを疑い、Linuxと同様のパス設定を追加した
   - `ENV PATH /opt/conda/envs/base/bin:$PATH`を追加したが解決せず。

3. `conda activate`, `conda init`
   - cmdstanpy実行時のエラーメッセージには、install_cmdstanをやり直すかconda (re)activateするかが必要とあるため、（セットアップをやり直す後者は最終手段として）conda activateを試した。
   - そのままのDockerfileでは、`conda activate`はエラーを生じ実行できない。そのため下記の追加が必要。
     - bashの有効化`SHELL ["/bin/bash", "-c"]`
     - `eval "$(conda shell.bash hook)" && \`
       - この解決策は[GitHubのIssue](<https://github.com/ContinuumIO/docker-images/issues/89#issuecomment-887935997>)から発見した。

4. base環境を諦め、別にconda仮想環境を作る
   - conda activateができないため避けていたが、Dockerfile内でbashを起動すれば回避できた（後述）ため試行した。
   - しかし問題は解決せず。

5. `install_cmdstan`を実行する
   - 最終手段として、condaによるセットアップは無視し、上書きする形で再度セットアップさせ他。
   - Dockerfileにおいて`RUN install_cmdstan`を復活させたが、ImportErrorが発生し実行できなかった。
     - シェルスクリプト内でpythonが起動しているようだが、パス設定等に問題があるのか、cmdstanpyをインポートできていなかった模様。詳細は特定できず。
     - エラーメッセージは保存していなかったため省略。
   - 代わりにpythonから実行する方法を採った。
    `RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`
    - これもconda仮想環境上でやると上手くいかなかったが、base環境に戻り解決した。その名残で`install_cmdstanpy()`に引数`overwrite=True`を与えている。除いた場合は未検証。
 

##　最終的な解決策
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
- Stanモデルのコンパイル用パッケージがcondaとaptで二重にインストールされるため、コンテナの容量がやや大きくなってしまう。
- 前述の通り、`RUN install_cmdstan`がImportErrorとなるため、pythonから実行している。

myenv.yamlに変更はなし。  
- ただし作成後の環境を`conda env export`すると、チャンネルに除外したはずの`defaults`が現れてしまった。
  - pipが使用されたためだろうか？
  - exportしたファイルを元に環境を再現したい場合は、channelsから`defaults`を除き、`nodefaults`に代える必要がある。

## 補足：Dockerfile内でconda activateする方法
- 試行錯誤の副産物として、Dockerfile内で`conda activate`を可能にする方法を得た。結果的に採用はされなかったが、補足としてここに残しておく。