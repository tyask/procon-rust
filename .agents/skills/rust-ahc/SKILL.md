---
name: rust-ahc
description: RustでAtCoder Heuristic Contest (AHC) やヒューリスティック系コンテストのsolverを実装・改善・検証するための実戦手順。Use when Codex works on Rust AHC solutions, local tools, scoring, greedy construction, hill climbing, simulated annealing, beam search, time-limited improvement loops, or contest/*/src/bin/a.rs submissions in this procon-rust repo.
---

# Rust AHC

## 基本姿勢

AHC の solver を Rust で進めるときは、まず問題文・入力形式・出力形式・スコア式・制約・時間制限・ローカル `tools/` の有無を確認する。実装前に「状態」「評価関数」「初期解」「近傍」「検証方法」を短く言語化し、狭い確認から始める。

この repo では `contest/<name>/` を独立した Cargo project として扱う。Cargo command は対象 contest directory で実行し、package 名ではなく今いる project を基準にする。

## 実装フロー

- `contest/<name>/src/bin/a.rs` を主エントリポイントとして編集する。
- 新しい utility を書く前に `contest/template/src/fumin/` を探す。特に焼きなましは `sa.rs`、ビームサーチは `tree_beam.rs` を確認する。
- 提出は単一ファイル前提にする。template utility を使う場合も library 参照にせず、必要な source を contest 側の `a.rs` にコピーする。
- コピー後も template 側 API から大きく乖離させない。後で template へ戻しやすい形を保つ。
- テストを書く場合はファイル末尾へまとめる。

## 探索方針

- まず必ず valid な初期解を作る。雑でも出力形式と制約を満たす状態を早く得る。
- greedy で初期解の質を上げ、評価関数と差分更新を整える。
- 局所改善できる構造なら hill climbing または simulated annealing を使う。近傍は「適用」「差分評価」「rollback」を軽くする。
- ターン列や手順列を幅で持つ問題では beam search を検討する。状態の重複排除 hash と比較用 score / 実 score の違いを明確にする。
- 性能が重要な箇所では、allocation と clone を減らし、preallocation、fixed-size arrays、bit operations、単純な loop を優先する。

## 時間制限

改善ループは制限時間まで回す。`Instant::elapsed()` は高頻度で呼ばず、一定回数ごとに確認する。

```rust
let start = std::time::Instant::now();
let mut cnt = 0usize;

loop {
    if (cnt & 0x7F) == 0 {
        if start.elapsed().as_millis() >= 1900 { break; }
    }
    cnt += 1;
    // improve solution
}
```

制限時間は本番の余裕を見て短めに置く。複数ケース実行や tools 経由では、計測 overhead と入出力 overhead も見る。

## デバッグと出力

- debug log は stderr に出す。grep しやすい安定した 1 行 format にする。
- stdout は問題文の出力形式だけに使う。AtCoder では 1 文字でも余分な出力を混ぜない。
- スコア・反復回数・採用回数・温度・改善回数など、探索の調整に効く値を local 実行で見えるようにする。

## 検証

まず対象 contest directory で狭く確認する。

```bash
cargo check
```

ローカル入力がある場合は、少なくとも 1 ケースを release で実行する。

```bash
cargo run --release --bin a < tools/in/0000.txt
```

実行結果を観察しても追加の実装変更をせず、結果をそのまま報告する。
