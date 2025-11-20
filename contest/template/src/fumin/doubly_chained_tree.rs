use itertools::Itertools;

pub type NodeId = u16;
const INF: NodeId = !0;

pub trait NodeValue : std::fmt::Debug + Clone + Default {}

#[derive(Debug, Clone, Default)]
pub struct Node<T: NodeValue> {
    pub parent: NodeId,
    child: NodeId,
    prev: NodeId,
    next: NodeId,
    pub value: T,
}

impl<T: NodeValue> Node<T> {
    pub fn is_root(&self) -> bool { self.parent == INF }
    pub fn has_child(&self) -> bool { self.child != INF }
}

pub trait Context<T: NodeValue> {
    fn apply(&mut self, value: &T);
    fn revert(&mut self, value: &T);
}


pub struct DoublyChainedTree<T: NodeValue> {
    pub nodes:Vec<Node<T>>,
    free:Vec<NodeId>,
}

impl<T: NodeValue> DoublyChainedTree<T> {
    pub fn new(max_nodes: usize, root: T) -> Self {
        let mut nodes = vec![Node::default(); max_nodes];
        nodes[0] = Node {parent: INF, child: INF, prev: INF, next: INF, value: root};
        let free=(1..nodes.len() as NodeId).rev().collect_vec();
        Self {
            nodes,
            free,
        }
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
            self.nodes[n as usize] = Node{parent, next, child: INF, prev: INF, value};
            n
        } else {
            let n = self.nodes.len() as NodeId;
            assert!(n!=0,"uintのサイズが足りないよ");
            self.nodes.push(Node{parent, next, child: INF, prev: INF, value});
            n
        };

        // 兄弟が既にいる場合、その兄弟のprevに新しいNodeを追加
        if next != INF { self.nodes[next as usize].prev = new; }

        // 親の子として新しいNodeを追加
        self.nodes[parent as usize].child = new;

        new
    }

    pub fn remove_node(&mut self, mut idx: NodeId) {
        loop{
            self.free.push(idx);
            let Node{prev,next,parent,..} = self.nodes[idx as usize];
            assert_ne!(parent, INF,"全てのノードを消そうとしています");

            // 削除対象Nodeが一人っ子の場合、親Nodeを残す意味がないため削除する
            if prev & next == INF {
                idx = parent;
                continue;
            }

            // 削除対象Nodeのnextを付け替え
            if prev != INF {
                self.nodes[prev as usize].next = next;
            } else{
                self.nodes[parent as usize].child = next;
            }

            // 削除対象Nodeのprevを付け替え
            if next != INF {
                self.nodes[next as usize].prev = prev;
            }
            
            break;
        }
    }

    pub fn walk_leaf<C: Context<T>>(&self, ctx: &mut C, mut walker: impl FnMut(&C, &Node<T>)) {
        let mut cur_node = 0;
        loop {
            let Node{next,child,..} = self.nodes[cur_node];
            if next==INF || child==INF { break; }
            cur_node = child as usize;
            ctx.apply(&self.nodes[cur_node].value);
        }

        let root = cur_node;
        loop {
            let child = self.nodes[cur_node].child;
            if child == INF {
                walker(ctx, &self.nodes[cur_node]);

                loop {
                    if cur_node == root { return; }
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

// CAP(IGNORE_BELOW)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        impl NodeValue for usize {}
        let mut t = DoublyChainedTree::<usize>::new(10, 0);
        // 0 + 1 + 3
        //   |   + 4
        //   |
        //   + 2 + 5

        t.add_node(0, 1);
        t.add_node(0, 2);
        t.add_node(1, 3);
        t.add_node(1, 4);
        t.add_node(2, 5);

        assert_eq!(INF, t.nodes[0].parent);
        assert_eq!(0,   t.nodes[1].parent);
        assert_eq!(0,   t.nodes[2].parent);
        assert_eq!(1,   t.nodes[3].parent);
        assert_eq!(1,   t.nodes[4].parent);
        assert_eq!(2,   t.nodes[5].parent);

        struct Ctx { x: usize, }
        impl Context<usize> for Ctx {
            fn apply(&mut self, value: &usize) {
                self.x += value;
            }
            fn revert(&mut self, value: &usize) {
                self.x -= value;
            }
        }
        let mut ctx = Ctx {x:0};
        let mut v = vec![];
        t.walk_leaf(&mut ctx, |ctx,node|{
            v.push((ctx.x, node.value));
        });
        assert_eq!(vec![(7,5),(5,4),(4,3)], v);
    }
}