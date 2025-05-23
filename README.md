# GA-Car-Simulation
# AI-GA-Portfolio: Braitenberg車両の制御におけるGA比較実験

このプロジェクトは、シミュレーション環境内において、Braitenberg車両の制御を目的とした２種類の遺伝的アルゴリズム（Spatial GA と Full Microbial GA）の性能比較実験を行ったものです。

## プロジェクト概要
- **背景:** ロボット制御における信号処理とフィードバックを用いたシステムの最適化
- **目的:** GAを用いて最適な運動パラメータ（ニューラルネットワークによる制御パラメータ）の獲得と、シミュレーションおよび実世界での評価
- **手法:** 
  - シミュレーション環境の構築とBraitenbergエージェントの定義
  - 遺伝的アルゴリズム（Spatial GA と Full Microbial GA）の実装と比較
  - 実験結果（軌跡、フィットネス推移）およびレポートでの考察

## リポジトリ構成
- **notebooks/:** 実験の実行および結果可視化用Jupyter Notebook
- **src/:** シミュレーション、GAアルゴリズムの各種クラス実装
- **reports/:** 実験の背景と結果をまとめたレポート（PDF形式）
- **docs/:** GitHub Pages向けのプロジェクト詳細ドキュメント

## 実行方法
1. リポジトリをクローンしてください：git clone https://github.com/your_username/AI-GA-Portfolio.git

2. 必要な依存ライブラリを `requirements.txt` からインストールします。
3. `notebooks/Simulation_Experiments.ipynb` を Jupyter Notebook で開き、セルを実行して実験結果を確認してください。

## GitHub Pages
このプロジェクトの詳細な紹介と実験結果、考察は [GitHub Pages](https://your_username.github.io/AI-GA-Portfolio/) からご覧いただけます。

## レポート
実験の全体像と評価の詳細は、[reports/AIAB_report_subm.pdf](./reports/AIAB_report_subm.pdf) をご参照ください。

