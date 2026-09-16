#![allow(dead_code)]

use rustc_hash::FxHashMap;
use std::marker::PhantomData;

use super::doubly_chained_tree;
use crate::common::*;

/*
MultiTurn BeamSearch モジュール（汎用）
====================================

通常の `tree_beam.rs` は深さ i の状態から深さ i+1 の状態だけを作る。
このモジュールは、深さ i の状態から深さ i+n の候補を直接作れる版です。

使い方の要点
------------
1) 操作型 `Op` を定義する
   - `Op: NodeValue`（Debug + Clone + Default）
2) 文脈型 `State` を定義する
   - `impl BeamStateMulti<Op> for State`
   - `append_cands(turn, parent, out)` で `next_turn > turn` の候補を push する
3) `BeamSearchMulti::new(cfg, state).solve()` で最良操作列を得る

`Cand::score` はビーム内の比較スコア、`raw_score` は最終候補や終端候補の採用に使う実スコアです。
*/

#[allow(non_camel_case_types)]
pub use doubly_chained_tree::Node as Node;
pub use doubly_chained_tree::NodeId as NodeId;
pub use doubly_chained_tree::NodeValue as NodeValue;

// CAP(fumin::doubly_chained_tree)

pub trait BeamStateMulti<Op: NodeValue> {
    fn apply(&mut self, value: &Op);
    fn revert(&mut self, value: &Op);
    fn append_cands(
        &mut self,
        turn: us,
        parent: &Node<Op>,
        cands: &mut impl CandCollector<Op>,
    );
}

struct MultiContext<'a, Op: NodeValue, State: BeamStateMulti<Op>> {
    state: &'a mut State,
    _marker: PhantomData<fn(Op)>,
}

impl<Op: NodeValue, State: BeamStateMulti<Op>> doubly_chained_tree::Context<Op>
    for MultiContext<'_, Op, State>
{
    fn apply(&mut self, value: &Op) {
        self.state.apply(value);
    }

    fn revert(&mut self, value: &Op) {
        self.state.revert(value);
    }
}

pub struct Config {
    // 各ターンで最終的に保持する状態数（score 上位 + hash 重複排除後）。
    pub max_width: us,
    // 探索する最大ターン数。候補の next_turn は 1..=turn に入れる。
    pub turn: us,
    // true の場合、is_end=true の候補が採用されたターンで早期終了する。
    pub minimize_turn: bool,
}

#[derive(Debug, Clone)]
pub struct Cand<Op> {
    // ビーム木での親ノードID。
    pub parent: NodeId,
    // この候補が着地するターン。現在ターンより大きく、cfg.turn 以下であること。
    pub next_turn: us,
    // ビーム選抜に使う比較スコア。
    pub score: i64,
    // 実スコア（最終候補や終端候補の選択に使用）。
    pub raw_score: i64,
    // 同じ next_turn 内での状態重複排除キー。
    pub hash: u64,
    // 終端状態かどうか。minimize_turn=true のとき早期終了判定に使う。
    pub is_end: bool,
    // 子状態への遷移情報。
    pub op: Op,
}

pub trait CandCollector<Op: NodeValue> {
    fn push(&mut self, cand: Cand<Op>);
}

struct BucketCandCollector<'a, Op: NodeValue> {
    turn: us,
    max_turn: us,
    parent: NodeId,
    pushed: us,
    buckets: &'a mut [Vec<Cand<Op>>],
}

impl<'a, Op: NodeValue> BucketCandCollector<'a, Op> {
    fn new(turn: us, max_turn: us, parent: NodeId, buckets: &'a mut [Vec<Cand<Op>>]) -> Self {
        Self {
            turn,
            max_turn,
            parent,
            pushed: 0,
            buckets,
        }
    }

    fn pushed(&self) -> us {
        self.pushed
    }
}

impl<Op: NodeValue> CandCollector<Op> for BucketCandCollector<'_, Op> {
    fn push(&mut self, cand: Cand<Op>) {
        assert_eq!(cand.parent, self.parent, "Cand::parent が append_cands の parent と一致していません");
        assert!(
            self.turn < cand.next_turn && cand.next_turn <= self.max_turn,
            "Cand::next_turn は現在ターンより後、かつ cfg.turn 以下にしてください"
        );

        let next_turn = cand.next_turn;
        self.buckets[next_turn].push(cand);
        self.pushed += 1;
    }
}

// dp[i] -> dp[i+n] の候補を管理するビームサーチエンジン。
//
// refs[node] は「そのノードを親にする未処理候補」または「採用済み子ノード」からの参照数。
// refs が 0 になった葉は、共有木から安全に削除できる。
pub struct BeamSearchMulti<Op: NodeValue, State: BeamStateMulti<Op>> {
    cfg: Config,
    state: State,
    tree: doubly_chained_tree::DoublyChainedTree<Op>,
    buckets: Vec<Vec<Cand<Op>>>,
    refs: Vec<us>,
    active: Vec<bool>,
    frontier: Vec<NodeId>,
    dead_leaf: Vec<NodeId>,
}

impl<Op, State> BeamSearchMulti<Op, State>
where
    Op: NodeValue,
    State: BeamStateMulti<Op>,
{
    pub fn new(cfg: Config, state: State) -> Self {
        let max_nodes = cfg.max_width * 10 + 1;
        assert!(max_nodes < NodeId::MAX as usize, "NodeIdのサイズが足りないよ");
        let buckets = (0..=cfg.turn)
            .map(|_| Vec::with_capacity(cfg.max_width * 4))
            .collect::<Vec<_>>();

        let mut active = vec![false; max_nodes];
        active[0] = true;

        Self {
            cfg,
            state,
            tree: doubly_chained_tree::DoublyChainedTree::new(max_nodes, Op::default()),
            buckets,
            refs: vec![0; max_nodes],
            active,
            frontier: vec![0],
            dead_leaf: vec![],
        }
    }

    pub fn solve(&mut self) -> Vec<Op> {
        if self.cfg.turn == 0 {
            return vec![];
        }

        for turn in 0..=self.cfg.turn {
            if turn != 0 {
                let selected = self.select_turn_candidates(turn);
                if selected.is_empty() {
                    self.frontier.clear();
                    continue;
                }

                if self.cfg.minimize_turn && selected.iter().any(|cand| cand.is_end) {
                    let best = selected
                        .into_iter()
                        .filter(|cand| cand.is_end)
                        .max_by_key(|cand| cand.raw_score)
                        .unwrap();
                    let mut ret = self.restore(best.parent);
                    ret.push(best.op);
                    return ret;
                }

                if turn == self.cfg.turn {
                    let best = selected.into_iter().max_by_key(|cand| cand.raw_score).unwrap();
                    let mut ret = self.restore(best.parent);
                    ret.push(best.op);
                    return ret;
                }

                self.update_frontier(selected);
            }

            if self.frontier.is_empty() || turn == self.cfg.turn {
                continue;
            }

            self.mark_active_frontier();
            self.enum_cands(turn);
            self.remove_dead_leaves();
        }

        panic!("BeamSearchMulti に最終ターンへ到達する候補がありません");
    }

    fn select_turn_candidates(&mut self, turn: us) -> Vec<Cand<Op>> {
        let mut best_by_hash = FxHashMap::<u64, Cand<Op>>::default();
        let mut rejected_parent = vec![];

        for cand in self.buckets[turn].drain(..) {
            match best_by_hash.get_mut(&cand.hash) {
                Some(current) if cand.score > current.score => {
                    rejected_parent.push(current.parent);
                    *current = cand;
                }
                Some(_) => {
                    rejected_parent.push(cand.parent);
                }
                None => {
                    best_by_hash.insert(cand.hash, cand);
                }
            }
        }

        let mut selected = best_by_hash.into_values().collect::<Vec<_>>();
        selected.sort_unstable_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then_with(|| a.hash.cmp(&b.hash))
                .then_with(|| a.raw_score.cmp(&b.raw_score))
        });

        if selected.len() > self.cfg.max_width {
            for cand in selected.drain(self.cfg.max_width..) {
                rejected_parent.push(cand.parent);
            }
        }

        for parent in rejected_parent {
            self.release_ref(parent);
        }

        selected
    }

    fn update_frontier(&mut self, selected: Vec<Cand<Op>>) {
        self.frontier.clear();
        for cand in selected {
            let child = self.tree.add_node(cand.parent, cand.op);
            self.ensure_aux_len();
            self.refs[child as usize] = 0;
            self.active[child as usize] = false;
            self.frontier.push(child);
        }
    }

    fn enum_cands(&mut self, turn: us) {
        self.dead_leaf.clear();

        let tree = &self.tree;
        let active = &self.active;
        let refs = &mut self.refs;
        let dead_leaf = &mut self.dead_leaf;
        let buckets = &mut self.buckets;
        let max_turn = self.cfg.turn;
        let mut ctx = MultiContext {
            state: &mut self.state,
            _marker: PhantomData,
        };

        tree.walk_leaf_filtered(
            &mut ctx,
            |id| active.get(id as usize).copied().unwrap_or(false),
            |ctx, parent| {
                let mut collector = BucketCandCollector::new(turn, max_turn, parent.id, buckets);
                ctx.state.append_cands(turn, parent, &mut collector);
                let pushed = collector.pushed();
                refs[parent.id as usize] += pushed;
                if pushed == 0 && !parent.is_root() {
                    dead_leaf.push(parent.id);
                }
            },
        );
    }

    fn remove_dead_leaves(&mut self) {
        let dead_leaf = std::mem::take(&mut self.dead_leaf);
        for id in dead_leaf {
            if !self.tree.nodes[id as usize].is_root()
                && self.refs[id as usize] == 0
                && !self.tree.nodes[id as usize].has_child()
            {
                self.remove_node(id);
            }
        }
    }

    fn release_ref(&mut self, id: NodeId) {
        let refs = &mut self.refs[id as usize];
        assert!(*refs > 0, "参照されていないノードの refs を減らそうとしています");
        *refs -= 1;

        if !self.tree.nodes[id as usize].is_root()
            && self.refs[id as usize] == 0
            && !self.tree.nodes[id as usize].has_child()
        {
            self.remove_node(id);
        }
    }

    fn remove_node(&mut self, id: NodeId) {
        let refs = &mut self.refs;
        self.tree.remove_node_guarded(id, |parent| {
            let parent_ref = &mut refs[parent as usize];
            assert!(*parent_ref > 0, "親ノードの refs が 0 のまま子を削除しようとしています");
            *parent_ref -= 1;
            parent != 0 && *parent_ref == 0
        });
    }

    fn mark_active_frontier(&mut self) {
        self.active.fill(false);
        for &id in &self.frontier {
            let mut cur = id;
            loop {
                self.active[cur as usize] = true;
                let node = &self.tree.nodes[cur as usize];
                if node.is_root() {
                    break;
                }
                cur = node.parent;
            }
        }
    }

    fn ensure_aux_len(&mut self) {
        let len = self.tree.nodes.len();
        if self.refs.len() < len {
            self.refs.resize(len, 0);
        }
        if self.active.len() < len {
            self.active.resize(len, false);
        }
    }

    fn restore(&self, mut idx: NodeId) -> Vec<Op> {
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
    struct TestOp {
        label: u8,
        delta: i64,
    }
    impl NodeValue for TestOp {}

    #[derive(Default)]
    struct TestState {
        score: i64,
        path: Vec<u8>,
    }

    impl TestState {
        fn push(
            out: &mut impl CandCollector<TestOp>,
            parent: NodeId,
            next_turn: us,
            label: u8,
            score: i64,
            hash: u64,
            is_end: bool,
        ) {
            out.push(Cand {
                parent,
                next_turn,
                score,
                raw_score: score,
                hash,
                is_end,
                op: TestOp {
                    label,
                    delta: score,
                },
            });
        }
    }

    impl BeamStateMulti<TestOp> for TestState {
        fn apply(&mut self, value: &TestOp) {
            self.score += value.delta;
            self.path.push(value.label);
        }

        fn revert(&mut self, value: &TestOp) {
            self.score -= value.delta;
            assert_eq!(self.path.pop(), Some(value.label));
        }

        fn append_cands(
            &mut self,
            turn: us,
            parent: &Node<TestOp>,
            out: &mut impl CandCollector<TestOp>,
        ) {
            match (turn, self.path.as_slice()) {
                (0, []) => {
                    Self::push(out, parent.id, 1, b'A', 10, 1, false);
                    Self::push(out, parent.id, 2, b'B', 20, 2, false);
                }
                (1, [b'A']) => {
                    Self::push(out, parent.id, 3, b'C', 15, 3, false);
                }
                (2, [b'B']) => {
                    Self::push(out, parent.id, 3, b'D', 30, 4, false);
                }
                _ => {}
            }
        }
    }

    fn labels(ops: Vec<TestOp>) -> Vec<u8> {
        ops.into_iter().map(|op| op.label).collect()
    }

    #[test]
    fn restores_best_generic_operation_sequence_with_multi_turn_jump() {
        let cfg = Config {
            max_width: 2,
            turn: 3,
            minimize_turn: false,
        };
        let mut beam = BeamSearchMulti::new(cfg, TestState::default());

        assert_eq!(labels(beam.solve()), vec![b'B', b'D']);
    }

    #[derive(Default)]
    struct SkipState {
        path: Vec<u8>,
    }

    impl BeamStateMulti<TestOp> for SkipState {
        fn apply(&mut self, value: &TestOp) {
            self.path.push(value.label);
        }

        fn revert(&mut self, value: &TestOp) {
            assert_eq!(self.path.pop(), Some(value.label));
        }

        fn append_cands(
            &mut self,
            turn: us,
            parent: &Node<TestOp>,
            out: &mut impl CandCollector<TestOp>,
        ) {
            match (turn, self.path.as_slice()) {
                (0, []) => TestState::push(out, parent.id, 2, b'B', 10, 1, false),
                (2, [b'B']) => TestState::push(out, parent.id, 3, b'D', 20, 2, false),
                _ => {}
            }
        }
    }

    #[test]
    fn skips_empty_intermediate_turns() {
        let cfg = Config {
            max_width: 1,
            turn: 3,
            minimize_turn: false,
        };
        let mut beam = BeamSearchMulti::new(cfg, SkipState::default());

        assert_eq!(labels(beam.solve()), vec![b'B', b'D']);
    }

    #[derive(Default)]
    struct DedupState;

    impl BeamStateMulti<TestOp> for DedupState {
        fn apply(&mut self, _value: &TestOp) {}
        fn revert(&mut self, _value: &TestOp) {}

        fn append_cands(
            &mut self,
            turn: us,
            parent: &Node<TestOp>,
            out: &mut impl CandCollector<TestOp>,
        ) {
            if turn == 0 {
                TestState::push(out, parent.id, 1, b'L', 5, 10, false);
                TestState::push(out, parent.id, 1, b'H', 7, 10, false);
            }
        }
    }

    #[test]
    fn keeps_best_score_for_same_hash_at_same_target_turn() {
        let cfg = Config {
            max_width: 2,
            turn: 1,
            minimize_turn: false,
        };
        let mut beam = BeamSearchMulti::new(cfg, DedupState);

        assert_eq!(labels(beam.solve()), vec![b'H']);
    }

    #[derive(Default)]
    struct PruneState {
        path: Vec<u8>,
    }

    impl BeamStateMulti<TestOp> for PruneState {
        fn apply(&mut self, value: &TestOp) {
            self.path.push(value.label);
        }

        fn revert(&mut self, value: &TestOp) {
            assert_eq!(self.path.pop(), Some(value.label));
        }

        fn append_cands(
            &mut self,
            turn: us,
            parent: &Node<TestOp>,
            out: &mut impl CandCollector<TestOp>,
        ) {
            match (turn, self.path.as_slice()) {
                (0, []) => {
                    TestState::push(out, parent.id, 1, b'A', 20, 1, false);
                    TestState::push(out, parent.id, 1, b'B', 10, 2, false);
                }
                (1, [b'A']) => TestState::push(out, parent.id, 2, b'C', 30, 3, false),
                (1, [b'B']) => panic!("pruned branch must not be traversed"),
                _ => {}
            }
        }
    }

    #[test]
    fn rejected_candidates_are_not_traversed_later() {
        let cfg = Config {
            max_width: 1,
            turn: 2,
            minimize_turn: false,
        };
        let mut beam = BeamSearchMulti::new(cfg, PruneState::default());

        assert_eq!(labels(beam.solve()), vec![b'A', b'C']);
    }

    #[derive(Default)]
    struct MinimizeTurnState;

    impl BeamStateMulti<TestOp> for MinimizeTurnState {
        fn apply(&mut self, _value: &TestOp) {}
        fn revert(&mut self, _value: &TestOp) {}

        fn append_cands(
            &mut self,
            turn: us,
            parent: &Node<TestOp>,
            out: &mut impl CandCollector<TestOp>,
        ) {
            if turn == 0 {
                TestState::push(out, parent.id, 1, b'E', 10, 1, true);
                TestState::push(out, parent.id, 3, b'L', 100, 2, false);
            }
        }
    }

    #[test]
    fn minimize_turn_returns_terminal_candidate_early() {
        let cfg = Config {
            max_width: 2,
            turn: 3,
            minimize_turn: true,
        };
        let mut beam = BeamSearchMulti::new(cfg, MinimizeTurnState);

        assert_eq!(labels(beam.solve()), vec![b'E']);
    }
}
