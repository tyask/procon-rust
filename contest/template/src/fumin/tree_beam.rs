#![allow(dead_code)]

use std::{cmp::Reverse, collections::BinaryHeap, fmt::Debug};
use rustc_hash::FxHashMap;
use super::doubly_chained_tree;
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

let cfg = beam::Config { max_width: 1000, turn: LIMIT };
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

// CAP(fumin::doubly_chained_tree)

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
struct BeamContext<'a, Op: NodeValue, State: BeamState<Op>> {
    state: &'a mut State,
    _marker: std::marker::PhantomData<fn(Op)>,
}

impl<Op: NodeValue, State: BeamState<Op>> doubly_chained_tree::Context<Op>
    for BeamContext<'_, Op, State>
{
    fn apply(&mut self, value: &Op) {
        self.state.apply(value);
    }

    fn revert(&mut self, value: &Op) {
        self.state.revert(value);
    }
}

pub struct Config {
    // 各深さで最終的に保持する状態数（ソート + 重複排除後）。
    pub max_width: us,
    // 探索する最大深さ（ターン数）。
    pub turn: us,
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
        for _t in 0..self.cfg.turn {
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
        let tree = &self.tree;
        let mut ctx = BeamContext {
            state: &mut self.state,
            _marker: std::marker::PhantomData,
        };
        tree.walk_leaf(&mut ctx, |ctx, parent| {
            ctx.state.append_cands(parent, cands);
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
