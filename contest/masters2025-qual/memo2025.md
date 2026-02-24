# 第二回マスターズ選手権
https://atcoder.jp/contests/masters2025-qual

## ヒューリスティックコンテストの概要
* 去年のマスターズ予選を参考に説明
* マスターズ選手権特有の性質 (入力の一部は同じになる)
* 決勝の性質 (入力の傾向が異なるだけの複数の問題が出題される). 今年も多分この方式になりそう?

### ヒューリスティックの手法
* 乱択、山登り、焼きなまし
* 貪欲、ビームサーチ
* 高速化

## 作戦
* 安河内: 開発 (Rustで提出したい(速いので)、ライブラリ揃ってる)
* ~~A: ビジュアライザ作成 (できれば最初の2-3hくらい? Rustで書く必要あるが、ChatGPTを駆使する)~~
* ~~B: アイデアが思い付けば出してもらう、実現可能かどうか実装して試してみる。とはいえ、BFS/DFSとか空で書けないと厳しい。実装は多分ChatGPTに頼るのがいい。私がうまく何か一部の機能の開発をお願いすることができればそれをやってもらうかも。~~
* 齋藤・高木: ペアプロでビジュアライザ作成、アイデア出し

## 決めること
### 役割
* 開発
* ビジュアライザ作成

### 当日の環境
* 予選は在宅
* リモート環境 (会社のTeamsかSkype)
* ファイルの共有方法 (TeamsかGoogle Drive)

### 当日までにやる準備
* 参加登録  
チーム名: N**  
合言葉: NSS197912

* なんでもいいので、コードを提出してみる  
システムに慣れるという意味で。標準入力のやりかたとか覚えておいた方がいい  
アルゴの問題だけど、最初はこれをやってみるといいかも  
https://atcoder.jp/contests/abs

* pahcer(テストツール)の導入  
https://github.com/terry-u16/pahcer  
練習としてAHC043を使うのが良さそう。サンプルコードが提供されてるので。

* visualizer-templateの導入  
https://github.com/yunix-kyopro/visualizer-template-public  
chokudai-contest-005というブランチにサンプルコードがあるので、それを参考にするのが良さそう.

* 適当なAHCの過去問で上のツールを動かしてみる  
https://atcoder.jp/contests/ahc043  
https://atcoder.jp/contests/masters-qual  
https://atcoder.jp/contests/masters2024-final  
 \* 決勝の問題はちょっと特殊なのであまり参考にならないかも
