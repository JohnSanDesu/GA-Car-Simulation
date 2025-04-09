# ベースイメージとして、VSCode devcontainer用の Python 3.9 イメージを使用
FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを /workspace に設定
WORKDIR /workspace

# 必要なファイルや依存関係があればここで追加
