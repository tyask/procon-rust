#![allow(dead_code)]

pub type NodeId = u16;
const INF: NodeId = !0;
pub const NIL: NodeId = INF;

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
#[derive(Clone)]
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

    pub fn first_child(&self, parent: NodeId) -> NodeId {
        self.nodes[parent as usize].child
    }

    pub fn next_sibling(&self, idx: NodeId) -> NodeId {
        self.nodes[idx as usize].next
    }

    pub fn prev_sibling(&self, idx: NodeId) -> NodeId {
        self.nodes[idx as usize].prev
    }

    pub fn number_of_children(&self, parent: NodeId) -> usize {
        let mut count = 0;
        let mut child = self.nodes[parent as usize].child;
        while child != INF {
            count += 1;
            child = self.nodes[child as usize].next;
        }
        count
    }

    pub fn detach_node(&mut self, idx: NodeId) {
        // `idx` 単体だけを木から抜き取る。
        //
        // 子を持つ場合は子リストを `idx` のいた兄弟位置へ昇格させる。
        // つまり、木全体としては `idx` だけが消え、`idx` の子孫は
        // `idx` の親へ一段持ち上がる。
        //
        // `idx` 自体は free に戻さない。value を保持した detached node として、
        // あとで `insert_node` で別の場所へ挿し直せるようにする。
        let parent = self.nodes[idx as usize].parent;
        assert_ne!(parent, INF, "root は detach_node できません");

        let prev = self.nodes[idx as usize].prev;
        let next = self.nodes[idx as usize].next;
        let child = self.nodes[idx as usize].child;

        if child == INF {
            // 子がいない場合は、通常の双方向連結リスト削除と同じ。
            // prev と next を直接つなぎ、親の先頭子ポインタも必要なら更新する。
            if prev != INF {
                self.nodes[prev as usize].next = next;
            } else {
                self.nodes[parent as usize].child = next;
            }

            if next != INF {
                self.nodes[next as usize].prev = prev;
            }
        } else {
            // 子がいる場合は、子リスト全体を `idx` の兄弟列へ splice する。
            // idx: prev <-> idx <-> next
            // child list: child -> ... -> tail
            // after: prev <-> child -> ... -> tail <-> next
            if prev != INF {
                self.nodes[prev as usize].next = child;
            } else {
                self.nodes[parent as usize].child = child;
            }
            self.nodes[child as usize].prev = prev;

            // 昇格する子は全て新しい親を指す必要がある。
            // 同時に tail を探して、元の next とつなげる。
            let mut tail = child;
            loop {
                self.nodes[tail as usize].parent = parent;
                let tail_next = self.nodes[tail as usize].next;
                if tail_next == INF {
                    break;
                }
                tail = tail_next;
            }

            self.nodes[tail as usize].next = next;
            if next != INF {
                self.nodes[next as usize].prev = tail;
            }
        }

        // 抜き取った `idx` は「子なし・兄弟なし・親なし」の detached node にする。
        // child を INF にするので、元の子孫は idx には残らない。
        self.nodes[idx as usize].parent = INF;
        self.nodes[idx as usize].child = INF;
        self.nodes[idx as usize].prev = INF;
        self.nodes[idx as usize].next = INF;
    }

    pub fn insert_node(&mut self, idx: NodeId, parent: NodeId, pos: usize) {
        // detached node/subtree root を `parent` の子リストへ挿入する。
        //
        // pos == 0: parent の先頭の子として挿入
        // pos == number_of_children(parent): parent の末尾の子として挿入
        //
        // `idx.child` は変更しないので、`detach_sub_tree` した subtree も
        // そのまま別の場所へ移植できる。
        assert_ne!(idx, 0, "root は insert_node できません");
        assert_ne!(parent, INF, "parent が INF です");
        assert_eq!(
            self.nodes[idx as usize].parent, INF,
            "idx は detached node/subtree root である必要があります"
        );
        assert_eq!(
            self.nodes[idx as usize].prev, INF,
            "idx は兄弟リストから外れている必要があります"
        );
        assert_eq!(
            self.nodes[idx as usize].next, INF,
            "idx は兄弟リストから外れている必要があります"
        );
        let child_count = self.number_of_children(parent);
        assert!(
            pos <= child_count,
            "pos が子ノード数を超えています: pos={}, children={}",
            pos,
            child_count
        );

        let prev = if pos == 0 {
            INF
        } else {
            let mut prev = self.nodes[parent as usize].child;
            for _ in 1..pos {
                prev = self.nodes[prev as usize].next;
            }
            prev
        };

        // 挿入後に `idx` の次になるノードを先に確定しておく。
        // 先頭挿入なら現在の first child、それ以外なら現在の prev.next。
        let next = if prev == INF {
            self.nodes[parent as usize].child
        } else {
            assert_ne!(idx, prev, "idx と prev が同じです");
            self.nodes[prev as usize].next
        };

        // idx 側の三本の外向きリンクを張る。subtree 内部の child は維持する。
        self.nodes[idx as usize].parent = parent;
        self.nodes[idx as usize].prev = prev;
        self.nodes[idx as usize].next = next;

        // 親または直前兄弟から idx へ入るリンクを張る。
        if prev == INF {
            self.nodes[parent as usize].child = idx;
        } else {
            self.nodes[prev as usize].next = idx;
        }

        // 元々 idx の次になるノードがあれば、その prev も idx に向け直す。
        if next != INF {
            self.nodes[next as usize].prev = idx;
        }
    }

    pub fn detach_sub_tree(&mut self, idx: NodeId) {
        // `idx` を根とする subtree 全体を、現在の親の子リストから外す。
        //
        // `detach_node` と違い、idx.child は保持する。
        // そのため `idx` 以下の構造を丸ごと `insert_node` で移動できる。
        // こちらも free には戻さない。
        let parent = self.nodes[idx as usize].parent;
        assert_ne!(parent, INF, "root は detach_sub_tree できません");

        let prev = self.nodes[idx as usize].prev;
        let next = self.nodes[idx as usize].next;

        // 親側・兄弟側から idx だけを抜き、idx 以下の親子リンクは触らない。
        if prev != INF {
            self.nodes[prev as usize].next = next;
        } else {
            self.nodes[parent as usize].child = next;
        }

        if next != INF {
            self.nodes[next as usize].prev = prev;
        }

        // subtree root としては生きているが、外側の木からは切り離された状態にする。
        self.nodes[idx as usize].parent = INF;
        self.nodes[idx as usize].prev = INF;
        self.nodes[idx as usize].next = INF;
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

    pub fn remove_node_guarded(
        &mut self,
        mut idx: NodeId,
        mut parent_after_child_removed: impl FnMut(NodeId) -> bool,
    ) {
        // `parent_after_child_removed(parent)` は、idx が親から外れた直後の
        // 外部参照カウント更新と「その親も消してよいか」の判定を担う。
        loop {
            self.free.push(idx);
            let Node {
                prev, next, parent, ..
            } = self.nodes[idx as usize];
            assert_ne!(parent, INF, "全てのノードを消そうとしています");

            let can_remove_parent = parent_after_child_removed(parent);
            if prev == INF && next == INF && can_remove_parent {
                idx = parent;
                continue;
            }

            if prev != INF {
                self.nodes[prev as usize].next = next;
            } else {
                self.nodes[parent as usize].child = next;
            }

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

    pub fn walk_leaf_filtered<C: Context<T>>(
        &self,
        ctx: &mut C,
        mut is_active: impl FnMut(NodeId) -> bool,
        mut walker: impl FnMut(&mut C, &Node<T>),
    ) {
        if !is_active(0) {
            return;
        }
        self.walk_leaf_filtered_inner(0, ctx, &mut is_active, &mut walker);
    }

    fn walk_leaf_filtered_inner<C, F, W>(
        &self,
        idx: NodeId,
        ctx: &mut C,
        is_active: &mut F,
        walker: &mut W,
    )
    where
        C: Context<T>,
        F: FnMut(NodeId) -> bool,
        W: FnMut(&mut C, &Node<T>),
    {
        let mut child = self.nodes[idx as usize].child;
        let mut has_active_child = false;
        while child != INF {
            let child_id = child;
            child = self.nodes[child_id as usize].next;
            if is_active(child_id) {
                has_active_child = true;
                ctx.apply(&self.nodes[child_id as usize].value);
                self.walk_leaf_filtered_inner(child_id, ctx, is_active, walker);
                ctx.revert(&self.nodes[child_id as usize].value);
            }
        }

        if !has_active_child {
            walker(ctx, &self.nodes[idx as usize]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct TestOp(u8);
    impl NodeValue for TestOp {}

    fn child_list(tree: &DoublyChainedTree<TestOp>, parent: NodeId) -> Vec<NodeId> {
        let mut ret = vec![];
        let mut cur = tree.nodes[parent as usize].child;
        while cur != INF {
            ret.push(cur);
            cur = tree.nodes[cur as usize].next;
        }
        ret
    }

    fn assert_sibling_links(tree: &DoublyChainedTree<TestOp>, parent: NodeId, expected: &[NodeId]) {
        assert_eq!(child_list(tree, parent), expected);
        for (i, &id) in expected.iter().enumerate() {
            let prev = if i == 0 { INF } else { expected[i - 1] };
            let next = expected.get(i + 1).copied().unwrap_or(INF);
            assert_eq!(tree.nodes[id as usize].parent, parent);
            assert_eq!(tree.nodes[id as usize].prev, prev);
            assert_eq!(tree.nodes[id as usize].next, next);
        }
    }

    #[test]
    fn detach_node_promotes_children_and_can_insert_it_again() {
        let mut tree = DoublyChainedTree::new(16, TestOp(0));
        let tail = tree.add_node(0, TestOp(4));
        let idx = tree.add_node(0, TestOp(2));
        let prev = tree.add_node(0, TestOp(1));
        let child_tail = tree.add_node(idx, TestOp(22));
        let child_head = tree.add_node(idx, TestOp(21));
        let free_len = tree.free.len();

        assert_sibling_links(&tree, 0, &[prev, idx, tail]);
        assert_sibling_links(&tree, idx, &[child_head, child_tail]);

        tree.detach_node(idx);

        assert_eq!(tree.free.len(), free_len);
        assert_sibling_links(&tree, 0, &[prev, child_head, child_tail, tail]);
        assert_eq!(tree.nodes[idx as usize].parent, INF);
        assert_eq!(tree.nodes[idx as usize].child, INF);
        assert_eq!(tree.nodes[idx as usize].prev, INF);
        assert_eq!(tree.nodes[idx as usize].next, INF);
        assert_eq!(tree.number_of_children(0), 4);

        tree.insert_node(idx, 0, 2);

        assert_sibling_links(&tree, 0, &[prev, child_head, idx, child_tail, tail]);
        assert_eq!(tree.nodes[idx as usize].child, INF);
    }

    #[test]
    fn detach_sub_tree_keeps_descendants_and_can_insert_it_again() {
        let mut tree = DoublyChainedTree::new(16, TestOp(0));
        let tail = tree.add_node(0, TestOp(4));
        let idx = tree.add_node(0, TestOp(2));
        let prev = tree.add_node(0, TestOp(1));
        let child_tail = tree.add_node(idx, TestOp(22));
        let child_head = tree.add_node(idx, TestOp(21));
        let free_len = tree.free.len();

        tree.detach_sub_tree(idx);

        assert_eq!(tree.free.len(), free_len);
        assert_sibling_links(&tree, 0, &[prev, tail]);
        assert_eq!(tree.nodes[idx as usize].parent, INF);
        assert_eq!(tree.nodes[idx as usize].prev, INF);
        assert_eq!(tree.nodes[idx as usize].next, INF);
        assert_sibling_links(&tree, idx, &[child_head, child_tail]);
        assert_eq!(tree.number_of_children(0), 2);
        assert_eq!(tree.number_of_children(idx), 2);

        tree.insert_node(idx, 0, 0);

        assert_sibling_links(&tree, 0, &[idx, prev, tail]);
        assert_sibling_links(&tree, idx, &[child_head, child_tail]);
    }
}
