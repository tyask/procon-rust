# procon-rust AGENTS

## 目的

このリポジトリは `contest/<contest-id>/` ごとに独立した Rust プロジェクトを持ちます。
このファイルには repo 共通の構造と開発フローだけを書きます。
問題種別ごとの実装方針は、必要に応じて repo-local skill や contest ごとの `AGENTS.md` に分けてください。

## リポジトリ構造

- `contest/<name>/src/bin/a.rs`
  - 各コンテスト問題の主エントリポイント
- `contest/template/src/fumin/`
  - 共通部品の正本。共有 utility はここを基準に扱う
- `compete.toml`
  - contest 雛形と依存関係の基準
- `.agents/skills/`
  - repo-local Codex skill の配置先

## 実装ルール

- Cargo コマンドは、基本的に対象の `contest/<name>/` ディレクトリで実行する前提で書いてください
- 各 `contest/<name>/` は独立した Cargo プロジェクトです。package 名ではなく、今いるプロジェクトを基準に扱ってください
- 新しい utility を書く前に、まず `contest/template/src/fumin/*` を探してください
- コード提出時は単一ファイル提出が前提です。utility を使う場合はライブラリを直接参照せず、必要なソースコードを contest 側の `a.rs` へコピーしてください
- 単一ファイル提出のためにコードをコピーする場合でも、template 側の API から大きく乖離させないでください
- テストはファイル末尾にまとめてください

## 検証フロー

- まず対象の `contest/<name>/` で狭い確認を通してください
- 最低限の確認として `cargo check` を実行してください
- 入力ファイルやローカル検証資産がある場合は、問題に合わせて少なくとも 1 ケース実行してください

## 出力

- 出力は問題文で要求されるフォーマットに厳密に従ってください
- AtCoder では出力形式を 1 文字でも崩さないでください
- 最終提出コードでは debug logs を stdout に出さないでください

## 上書き方針

- このファイルは repo 全体のデフォルトです
- contest ごとに追加ルールが必要なら `contest/<name>/AGENTS.md` を置き、このファイルより具体的にしてください
