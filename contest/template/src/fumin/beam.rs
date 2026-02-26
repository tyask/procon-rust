#![allow(dead_code)]

use std::fmt::Debug;
use crate::common::*;

/*
BeamSearch モジュール（汎用）
このクラスは以下を参考に作成しています。
https://github.com/rhoo19937/beam-search-library/blob/main/src/tree_beam.rs
===========================

このモジュールは `doubly_chained_tree::DoublyChainedTree` を使った
再利用可能なビームサーチエンジンです。

BeamSearch を使うために必要な実装
--------------------------------
1) 操作型 `Op` を定義する
   - `Op: NodeValue`（Debug + Clone + Default）
   - 親状態 -> 子状態への1遷移を表す。
2) 文脈型 `Ctx` を定義する
   - `impl BeamContext<Op> for Ctx`
   - `apply(op)`: 子状態へ進める
   - `revert(op)`: 親状態へ巻き戻す
   - `append_cands(parent, out)`: 現在の文脈から次候補を列挙する
3) 生成して実行する
   - `let mut bs = beam::BeamSearch::new(cfg, ctx);`
   - `let best_ops: Vec<Op> = bs.solve();`

最小サンプル（コメント用の疑似コード）
------------------------------------
```ignore
#[derive(Debug, Clone, Default)]
struct Op { action: usize, delta: i64 }
impl beam::NodeValue for Op {}

struct Ctx { score: i64, step: usize }
impl beam::BeamContext<Op> for Ctx {
    fn apply(&mut self, op: &Op) {
        self.step += 1;
        self.score += op.delta;
    }
    fn revert(&mut self, op: &Op) {
        self.step -= 1;
        self.score -= op.delta;
    }
    fn append_cands(&mut self, parent: &beam::Node<Op>, out: &mut Vec<beam::Cand<Op>>) {
        for a in 0..4 {
            let op = Op { action: a, delta: 1 };
            out.push(beam::Cand {
                parent: parent.id,
                score: self.score + op.delta, // ビームの比較スコア
                raw_score: self.score + op.delta, // 最終採用時の実スコア
                hash: ((self.step as u64) << 32) ^ (a as u64), // 状態重複排除キー
                is_end: self.step + 1 == LIMIT,
                op,
            });
        }
    }
}

let cfg = beam::Config { max_width: 1000, tern: LIMIT };
let mut bs = beam::BeamSearch::new(cfg, Ctx { score: 0, step: 0 });
let ops = bs.solve();
```
*/

#[allow(non_camel_case_types)]
type uint = u16;

pub use doubly_chained_tree::Node as Node;
pub use doubly_chained_tree::NodeId as NodeId;
pub use doubly_chained_tree::NodeValue as NodeValue;

// BeamSearch が要求する文脈インターフェース。
// `apply/revert` は共有木を DFS で葉巡回するときに使われる。
// `append_cands` は各葉状態で次遷移候補を生成する。
pub trait BeamContext<Op: NodeValue> {
    fn apply(&mut self, value: &Op);
    fn revert(&mut self, value: &Op);
    fn append_cands(&mut self, parent: &Node<Op>, cands: &mut Vec<Cand<Op>>);
}

// BeamContext を木走査インターフェースへ橋渡しする。
// これにより doubly_chained_tree 側は beam 固有 API を知らずに済む。
impl<Op: NodeValue, C: BeamContext<Op>> doubly_chained_tree::Context<Op> for C {
    fn apply(&mut self, value: &Op) {
        BeamContext::apply(self, value);
    }
    fn revert(&mut self, value: &Op) {
        BeamContext::revert(self, value);
    }
}

pub struct Config {
    // 各深さで最終的に保持する状態数（ソート + 重複排除後）。
    pub max_width: us,
    // 探索する最大深さ（ターン数）。
    pub tern: us,
    // true の場合、is_end=true の候補が出たら探索を早期終了する。
    // ターン最小化問題では true、固定長問題では false が基本。
    pub minimize_turn: bool,
}

#[derive(Debug)]
pub struct Cand<Op> {
    // ビーム木での親ノードID。
    pub parent: NodeId,
    // ビーム選抜に使う比較スコア（ヒューリスティック可）。
    pub score: i64,
    // 実スコア（最終候補選択に使用）。
    pub raw_score: i64,
    // 状態重複排除に使うハッシュ。
    pub hash: u64,
    // 終端状態かどうか。終端候補が出たら早期終了できる。
    pub is_end: bool,
    // 子状態への遷移情報。
    pub op: Op,
}

// 汎用ビームサーチエンジン。
//
// 内部方針:
// - 現在の全葉を展開して候補を作る
// - スコア上位 `2 * width` を残す
// - `hash` で重複排除する
// - 先頭 `width` を採用する
// - 採用候補を次の葉として木に反映する
pub struct BeamSearch<Op: NodeValue, CtxT: BeamContext<Op>> {
    cfg: Config,
    ctx: CtxT,
    tree: doubly_chained_tree::DoublyChainedTree<Op>,
    leaf: Vec<NodeId>,
    next_leaf: Vec<NodeId>,
}

impl<Op, CtxT> BeamSearch<Op, CtxT>
where
    Op: NodeValue,
    CtxT: BeamContext<Op>,
{
    pub fn new(cfg: Config, ctx: CtxT) -> Self {
        // 一時ノードぶんの余裕を持たせる。
        let max_nodes = cfg.max_width * 5;
        assert!(max_nodes < uint::MAX as usize, "uintのサイズが足りないよ");
        let mut leaf = Vec::with_capacity(cfg.max_width);
        let next_leaf = Vec::with_capacity(cfg.max_width);
        leaf.push(0);

        Self {
            cfg,
            ctx,
            tree: doubly_chained_tree::DoublyChainedTree::new(max_nodes, Op::default()),
            leaf,
            next_leaf,
        }
    }

    pub fn solve(&mut self) -> Vec<Op> {
        use std::cmp::Reverse;

        let mut cands: Vec<Cand<Op>> = vec![];
        let mut dup = rustc_hash::FxHashSet::default();
        for _t in 0..self.cfg.tern {
            if _t != 0 {
                let m0 = self.cfg.max_width * 2;

                if cands.len() > m0 {
                    cands.select_nth_unstable_by_key(m0, |a| Reverse(a.score));
                    cands.truncate(m0);
                }

                cands.sort_unstable_by_key(|a| Reverse(a.score));

                // ターン最小化問題向けの早期終了判定
                if self.cfg.minimize_turn && cands.iter().any(|c| c.is_end) {
                    cands = cands.into_iter().filter(|c| c.is_end).cv();
                    break;
                }

                dup.clear();
                let cands = cands
                    .drain(..)
                    .filter(|cand| dup.insert(cand.hash))
                    .take(self.cfg.max_width);
                self.update(cands);
            }

            cands.clear();
            self.enum_cands(&mut cands);
            assert!(!cands.is_empty());
        }

        let best = cands.into_iter().max_by_key(|a| a.raw_score).unwrap();
        let mut ret = self.restore(best.parent);
        ret.push(best.op.clone());
        ret
    }

    fn enum_cands(&mut self, cands: &mut Vec<Cand<Op>>) {
        // 全葉を巡回し、それぞれの葉状態から候補を追加する。
        self.tree.walk_leaf(&mut self.ctx, |ctx, parent| {
            ctx.append_cands(parent, cands);
        });
    }

    fn update(&mut self, cands: impl Iterator<Item = Cand<Op>>) {
        // 次フロンティアを構築する。
        self.next_leaf.clear();
        for cand in cands {
            let child = self.tree.add_node(cand.parent, cand.op.clone());
            self.next_leaf.push(child);
        }

        // 子を持たなくなった旧葉を削除する。
        for &id in &self.leaf {
            if !self.tree.nodes[id as usize].has_child() {
                self.tree.remove_node(id);
            }
        }

        std::mem::swap(&mut self.leaf, &mut self.next_leaf);
    }

    fn restore(&self, mut idx: NodeId) -> Vec<Op> {
        // root -> 最良葉 の経路を復元する。
        let mut ret = vec![];
        loop {
            let node = &self.tree.nodes[idx as usize];
            if node.is_root() {
                break;
            }
            ret.push(node.value.clone());
            idx = node.parent;
        }

        ret.reverse();
        ret
    }
}

pub mod doubly_chained_tree {

use itertools::Itertools;

pub type NodeId = u16;
const INF: NodeId = !0;

// 各ノードが保持する値（このプロジェクトでは beam の操作 Op）。
pub trait NodeValue: std::fmt::Debug + Clone + Default {}

#[derive(Debug, Clone, Default)]
pub struct Node<T: NodeValue> {
    pub id: NodeId,
    // 親ノードID。root の親は INF。
    pub parent: NodeId,
    // 先頭の子ノードID。
    child: NodeId,
    // 親の子リスト内での兄弟リンク。
    prev: NodeId,
    next: NodeId,
    // ノードが持つ値。
    pub value: T,
}

impl<T: NodeValue> Node<T> {
    pub fn is_root(&self) -> bool {
        self.parent == INF
    }
    pub fn has_child(&self) -> bool {
        self.child != INF
    }
}

pub trait Context<T: NodeValue> {
    // 木を下るときに値を適用する。
    fn apply(&mut self, value: &T);
    // 木を上るときに値を巻き戻す。
    fn revert(&mut self, value: &T);
}

// ノード再利用を行う双方向連結木。
//
// - `nodes`: ノード本体の格納領域
// - `free`: 再利用可能なノードID
// - 親子・兄弟関係は parent/child/prev/next で管理
pub struct DoublyChainedTree<T: NodeValue> {
    pub nodes: Vec<Node<T>>,
    pub free: Vec<NodeId>,
}

impl<T: NodeValue> DoublyChainedTree<T> {
    pub fn new(max_nodes: usize, root: T) -> Self {
        let mut nodes = vec![Node::default(); max_nodes];
        nodes[0] = Node {
            id: 0,
            parent: INF,
            child: INF,
            prev: INF,
            next: INF,
            value: root,
        };
        let free = (1..nodes.len() as NodeId).rev().collect_vec();
        Self { nodes, free }
    }

    pub fn reset(&mut self, root: Node<T>) {
        self.nodes[0] = root;
        self.free.clear();
        self.free.extend((1..self.nodes.len() as NodeId).rev());
    }

    pub fn add_node(&mut self, parent: NodeId, value: T) -> NodeId {
        // Nodeのイメージ
        // (追加前)
        // 1
        // v
        // 2 > 3
        // (4を追加)
        // 1
        // v
        // 4 > 2 > 3

        // 新しいNodeを親の子の兄弟として追加する
        let next = self.nodes[parent as usize].child;
        let new = if let Some(n) = self.free.pop() {
            self.nodes[n as usize] = Node {
                id: n,
                parent,
                next,
                child: INF,
                prev: INF,
                value,
            };
            n
        } else {
            let n = self.nodes.len() as NodeId;
            assert!(n != 0, "Not enough size for NodeId");
            self.nodes.push(Node {
                id: n,
                parent,
                next,
                child: INF,
                prev: INF,
                value,
            });
            n
        };

        // 兄弟が既にいる場合、その兄弟のprevに新しいNodeを追加
        if next != INF {
            self.nodes[next as usize].prev = new;
        }

        // 親の子として新しいNodeを追加
        self.nodes[parent as usize].child = new;

        new
    }

    pub fn remove_node(&mut self, mut idx: NodeId) {
        // 葉側から上方向に不要ノードを連鎖削除する。
        // 親が一人っ子連鎖になる場合は再帰的に親も消す。
        loop {
            self.free.push(idx);
            let Node {
                prev, next, parent, ..
            } = self.nodes[idx as usize];
            assert_ne!(parent, INF, "全てのノードを消そうとしています");

            // 削除対象Nodeが一人っ子の場合、親Nodeを残す意味がないため削除する
            if prev & next == INF {
                idx = parent;
                continue;
            }

            // 削除対象Nodeのnextを付け替え
            if prev != INF {
                self.nodes[prev as usize].next = next;
            } else {
                self.nodes[parent as usize].child = next;
            }

            // 削除対象Nodeのprevを付け替え
            if next != INF {
                self.nodes[next as usize].prev = prev;
            }

            break;
        }
    }

    pub fn walk_leaf<C: Context<T>>(
        &self,
        ctx: &mut C,
        mut walker: impl FnMut(&mut C, &Node<T>),
    ) {
        // 葉ノードを DFS で巡回しつつ、外部文脈を apply/revert で同期する。
        //
        // 保証:
        // - 葉で `walker` を呼ぶ時点で `ctx` はその葉までの経路状態を反映している
        // - 葉間移動時は差分巻き戻しで文脈を更新し、毎回の全再構築はしない
        let mut cur_node = 0;
        loop {
            let Node { next, child, .. } = self.nodes[cur_node];
            if next == INF || child == INF {
                break;
            }
            cur_node = child as usize;
            ctx.apply(&self.nodes[cur_node].value);
        }

        let root = cur_node;
        loop {
            let child = self.nodes[cur_node].child;
            if child == INF {
                walker(ctx, &self.nodes[cur_node]);

                loop {
                    if cur_node == root {
                        return;
                    }
                    let node = &self.nodes[cur_node];
                    ctx.revert(&node.value);
                    // 兄弟に移動
                    if node.next != INF {
                        cur_node = node.next as usize;
                        ctx.apply(&self.nodes[cur_node].value);
                        break;
                    }
                    // 親に移動
                    cur_node = node.parent as usize;
                }
            } else {
                // 子に移動
                cur_node = child as usize;
                ctx.apply(&self.nodes[cur_node].value);
            }
        }
    }
}

}
