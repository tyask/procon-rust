#![allow(dead_code)]

use std::{cmp::Reverse, collections::BinaryHeap, fmt::Debug};
use rustc_hash::FxHashMap;
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
2) 文脈型 `State` を定義する
   - `impl BeamState<Op> for State`
   - `apply(op)`: 子状態へ進める
   - `revert(op)`: 親状態へ巻き戻す
   - `append_cands(parent, out)`: 現在の文脈から次候補を列挙する
3) 生成して実行する
   - `let mut bs = beam::BeamSearch::<Op, State, beam::HashCandSelector<Op>>::new(cfg, state);`
   - `let best_ops: Vec<Op> = bs.solve();`

最小サンプル（コメント用の疑似コード）
------------------------------------
```ignore
#[derive(Debug, Clone, Default)]
struct Op { action: usize, delta: i64 }
impl beam::NodeValue for Op {}

struct State { score: i64, step: usize }
impl beam::BeamState<Op> for State {
    fn apply(&mut self, op: &Op) {
        self.step += 1;
        self.score += op.delta;
    }
    fn revert(&mut self, op: &Op) {
        self.step -= 1;
        self.score -= op.delta;
    }
    fn append_cands(&mut self, parent: &beam::Node<Op>, out: &mut impl beam::CandSelector<Op>) {
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
let mut bs = beam::BeamSearch::<Op, State, beam::HashCandSelector<Op>>::new(
    cfg,
    State { score: 0, step: 0 },
);
let ops = bs.solve();
```
*/

#[allow(non_camel_case_types)]
pub use doubly_chained_tree::Node as Node;
pub use doubly_chained_tree::NodeId as NodeId;
pub use doubly_chained_tree::NodeValue as NodeValue;

// BeamSearch が要求する文脈インターフェース。
// `apply/revert` は共有木を DFS で葉巡回するときに使われる。
// `append_cands` は各葉状態で次遷移候補を生成する。
pub trait BeamState<Op: NodeValue> {
    fn apply(&mut self, value: &Op);
    fn revert(&mut self, value: &Op);
    fn append_cands(&mut self, parent: &Node<Op>, cands: &mut impl CandSelector<Op>);
}

// BeamState を木走査インターフェースへ橋渡しする。
// これにより doubly_chained_tree 側は beam 固有 API を知らずに済む。
impl<Op: NodeValue, St: BeamState<Op>> doubly_chained_tree::Context<Op> for St {
    fn apply(&mut self, value: &Op) {
        BeamState::apply(self, value);
    }
    fn revert(&mut self, value: &Op) {
        BeamState::revert(self, value);
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

#[derive(Debug, Clone)]
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

pub trait CandSelector<Op: NodeValue> {
    fn new(size: usize) -> Self;
    fn clear(&mut self);
    fn is_empty(&self) -> bool;
    fn capacity(&self) -> usize;
    fn push(&mut self, cand: Cand<Op>) -> bool;
    fn drain(&mut self) -> Vec<Cand<Op>>;
}

struct NoHashCandSelectorEntry<Op> {
    key: Reverse<i64>,
    cand: Cand<Op>,
}

impl<Op> PartialEq for NoHashCandSelectorEntry<Op>  { fn eq(&self, other: &Self) -> bool { self.key == other.key } }
impl<Op> Eq for NoHashCandSelectorEntry<Op>         {}
impl<Op> PartialOrd for NoHashCandSelectorEntry<Op> { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }
impl<Op> Ord for NoHashCandSelectorEntry<Op>        { fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.key.cmp(&other.key) } }

pub struct NoHashCandSelector<Op: NodeValue> {
    size: usize,
    heap: BinaryHeap<NoHashCandSelectorEntry<Op>>,
}

impl<Op: NodeValue> CandSelector<Op> for NoHashCandSelector<Op> {
    fn new(size: usize) -> Self {
        Self {
            size,
            heap: BinaryHeap::with_capacity(size),
        }
    }

    fn clear(&mut self) {
        self.heap.clear();
    }

    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    fn capacity(&self) -> usize {
        self.size
    }

    fn push(&mut self, cand: Cand<Op>) -> bool {
        if self.size == 0 {
            return false;
        }

        let score = cand.score;
        let entry = NoHashCandSelectorEntry {
            key: Reverse(score),
            cand,
        };

        if self.heap.len() < self.size {
            self.heap.push(entry);
            return true;
        }

        let worst_score = self.heap.peek().unwrap().key.0;
        if score > worst_score {
            self.heap.pop();
            self.heap.push(entry);
            return true;
        }
        false
    }

    fn drain(&mut self) -> Vec<Cand<Op>> {
        std::mem::take(&mut self.heap)
            .into_sorted_vec()
            .into_iter()
            .map(|entry| entry.cand)
            .collect()
    }
}

pub struct HashCandSelector<Op: NodeValue> {
    size: usize,
    best: FxHashMap<u64, Cand<Op>>,
    heap: BinaryHeap<HashCandSelectorEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HashCandSelectorEntry {
    key: Reverse<(i64, u64)>,
}

impl<Op: NodeValue> HashCandSelector<Op> {
    // `best` が正本で、heap は stale entry を含みうる。
    // worst 候補を読む前に、先頭に溜まった stale entry を落とす。
    fn drop_stale_top(&mut self) {
        while let Some(top) = self.heap.peek() {
            let (score, hash) = top.key.0;
            match self.best.get(&hash) {
                Some(cand) if cand.score == score => break,
                _ => {
                    self.heap.pop();
                }
            }
        }
    }
}

impl<Op: NodeValue> CandSelector<Op> for HashCandSelector<Op> {
    fn new(size: usize) -> Self {
        Self {
            size,
            best: FxHashMap::default(),
            heap: BinaryHeap::with_capacity(size),
        }
    }

    fn clear(&mut self) {
        self.best.clear();
        self.heap.clear();
    }

    fn is_empty(&self) -> bool {
        self.best.is_empty()
    }

    fn capacity(&self) -> usize {
        self.size
    }

    fn push(&mut self, cand: Cand<Op>) -> bool {
        if self.size == 0 {
            return false;
        }

        if let Some(current_score) = self.best.get(&cand.hash).map(|current| current.score) {
            if cand.score > current_score {
                self.heap.push(HashCandSelectorEntry {
                    key: Reverse((cand.score, cand.hash)),
                });
                self.best.insert(cand.hash, cand);
                return true;
            }
            return false;
        }

        if self.best.len() < self.size {
            self.heap.push(HashCandSelectorEntry {
                key: Reverse((cand.score, cand.hash)),
            });
            self.best.insert(cand.hash, cand);
            return true;
        }

        self.drop_stale_top();
        debug_assert_eq!(self.best.len(), self.size);
        let top = self.heap.peek().expect("heap must contain a live entry when best is full");
        let (worst_score, worst_hash) = top.key.0;
        if cand.score > worst_score {
            self.best.remove(&worst_hash);
            self.heap.push(HashCandSelectorEntry {
                key: Reverse((cand.score, cand.hash)),
            });
            self.best.insert(cand.hash, cand);
            return true;
        }
        false
    }

    fn drain(&mut self) -> Vec<Cand<Op>> {
        let mut ret: Vec<_> = std::mem::take(&mut self.best).into_values().collect();
        ret.sort_unstable_by(|a, b| b.score.cmp(&a.score).then_with(|| a.hash.cmp(&b.hash)));
        self.heap.clear();
        ret
    }
}

// 汎用ビームサーチエンジン。
//
// 内部方針:
// - 現在の全葉を展開して候補を作る
// - selector がスコア上位候補を保持する
// - `hash` 重複排除は selector 側の責務にできる
// - 先頭 `width` を採用する
// - 採用候補を次の葉として木に反映する
pub struct BeamSearch<Op: NodeValue, State: BeamState<Op>, Selector: CandSelector<Op> = HashCandSelector<Op>> {
    cfg: Config,
    state: State,
    tree: doubly_chained_tree::DoublyChainedTree<Op>,
    leaf: Vec<NodeId>,
    next_leaf: Vec<NodeId>,
    _selector: std::marker::PhantomData<fn()->Selector>,
}

impl<Op, State, Selector> BeamSearch<Op, State, Selector>
where
    Op: NodeValue,
    State: BeamState<Op>,
    Selector: CandSelector<Op>,
{
    pub fn new(cfg: Config, state: State) -> Self {
        let max_nodes = cfg.max_width * 5;
        assert!(max_nodes < NodeId::MAX as usize, "NodeIdのサイズが足りないよ");
        let mut leaf = Vec::with_capacity(cfg.max_width);
        let next_leaf = Vec::with_capacity(cfg.max_width);
        leaf.push(0);

        Self {
            cfg,
            state,
            tree: doubly_chained_tree::DoublyChainedTree::new(max_nodes, Op::default()),
            leaf,
            next_leaf,
            _selector: std::marker::PhantomData,
        }
    }

    pub fn solve(&mut self) -> Vec<Op> {
        let mut selector = Selector::new(self.cfg.max_width * 2);
        for _t in 0..self.cfg.tern {
            if _t != 0 {
                let selected = selector.drain();
                if self.cfg.minimize_turn && selected.iter().any(|c| c.is_end) {
                    let best = selected
                        .into_iter()
                        .filter(|c| c.is_end)
                        .max_by_key(|a| a.raw_score)
                        .unwrap();
                    let mut ret = self.restore(best.parent);
                    ret.push(best.op.clone());
                    return ret;
                }
                if selected.is_empty() {
                    break;
                }
                self.update(selected.into_iter().take(self.cfg.max_width));
            }

            selector.clear();
            self.enum_cands(&mut selector);
            assert!(!selector.is_empty());
        }

        let best = selector.drain().into_iter().max_by_key(|a| a.raw_score).unwrap();
        let mut ret = self.restore(best.parent);
        ret.push(best.op.clone());
        ret
    }

    fn enum_cands(&mut self, cands: &mut Selector) {
        self.tree.walk_leaf(&mut self.state, |st, parent| {
            st.append_cands(parent, cands);
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
        let free = (1..nodes.len() as NodeId).rev().collect::<Vec<_>>();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct TestOp(u8);
    impl NodeValue for TestOp {}

    fn cand(hash: u64, score: i64, op: u8) -> Cand<TestOp> {
        Cand {
            parent: 0,
            score,
            raw_score: score,
            hash,
            is_end: false,
            op: TestOp(op),
        }
    }

    #[test]
    fn hash_cand_selector_replaces_same_hash_only_when_score_is_higher() {
        let mut cands = HashCandSelector::new(4);
        cands.push(cand(10, 5, 1));
        cands.push(cand(10, 4, 2));
        cands.push(cand(10, 7, 3));

        let got = cands.drain();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].hash, 10);
        assert_eq!(got[0].score, 7);
        assert_eq!(got[0].op, TestOp(3));
    }

    #[test]
    fn hash_cand_selector_evicts_only_worst_when_full() {
        let mut cands = HashCandSelector::new(2);
        cands.push(cand(1, 5, 1));
        cands.push(cand(2, 7, 2));
        cands.push(cand(3, 6, 3));

        let got = cands.drain();
        let summary: Vec<_> = got.into_iter().map(|cand| (cand.hash, cand.score)).collect();
        assert_eq!(summary, vec![(2, 7), (3, 6)]);
    }

    #[test]
    fn hash_cand_selector_returns_score_descending_order() {
        let mut cands = HashCandSelector::new(4);
        cands.push(cand(1, 3, 1));
        cands.push(cand(2, 8, 2));
        cands.push(cand(3, 5, 3));

        let scores: Vec<_> = cands.drain().into_iter().map(|cand| cand.score).collect();
        assert_eq!(scores, vec![8, 5, 3]);
    }

    #[test]
    fn hash_cand_selector_evicts_using_live_worst_even_with_stale_heap_entries() {
        let mut cands = HashCandSelector::new(2);
        cands.push(cand(1, 5, 1));
        cands.push(cand(2, 7, 2));
        cands.push(cand(1, 9, 3));
        cands.push(cand(3, 6, 4));

        let got = cands.drain();
        let summary: Vec<_> = got.into_iter().map(|cand| (cand.hash, cand.score)).collect();
        assert_eq!(summary, vec![(1, 9), (2, 7)]);
    }

    #[test]
    fn no_hash_cand_selector_caps_size() {
        let mut cands = NoHashCandSelector::new(2);
        cands.push(cand(1, 5, 1));
        cands.push(cand(2, 7, 2));
        cands.push(cand(3, 6, 3));

        let summary: Vec<_> = cands.drain().into_iter().map(|cand| (cand.hash, cand.score)).collect();
        assert_eq!(summary, vec![(2, 7), (3, 6)]);
    }

    #[test]
    fn no_hash_cand_selector_keeps_same_score_candidates_without_hash_dedup() {
        let mut cands = NoHashCandSelector::new(3);
        cands.push(cand(10, 5, 1));
        cands.push(cand(10, 5, 2));
        cands.push(cand(20, 8, 3));

        let got = cands.drain();
        assert_eq!(got.len(), 3);
        let scores: Vec<_> = got.iter().map(|cand| cand.score).collect();
        assert_eq!(scores, vec![8, 5, 5]);
    }
}
