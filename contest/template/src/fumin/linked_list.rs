#![allow(dead_code)]
use crate::common::*;

type Id = us;

#[derive(Debug, Clone, Copy)]
struct Link {
    prev: Id,
    next: Id,
}

#[derive(Debug, Clone)]
struct Node<T> {
    id: Id,
    link: Link,
    value: T,
}

#[derive(Debug, Clone)]
pub struct LinkedList<T> {
    m: Vec<Node<T>>,
}

impl<T: Default> LinkedList<T> {
    pub fn new() -> Self {
        Self {
            m: vec![
                Node { id: Self::HEAD, link: Link { prev: Self::NIL, next: Self::TAIL }, value: T::default(), },
                Node { id: Self::TAIL, link: Link { prev: Self::HEAD, next: Self::NIL }, value: T::default(), },
            ],
        }
    }
}

impl<T> LinkedList<T> {
    const NIL: Id = us::INF;
    const HEAD: Id = 0;
    const TAIL: Id = 1;

    // 外向けのid (0-index)を、HEAD/TAILを考慮したidに変換
    fn resolve_id(&self, id: us) -> us { id + 2 }

    pub fn len(&self) -> us { self.m.len() - 2 }

    pub fn push(&mut self, value: T) -> us {
        self.insert_next(self.m[Self::TAIL].link.prev, value)
    }

    pub fn remove(&mut self, id: Id) -> T {
        assert!(id < self.m.len());
        // idの位置にある要素をリンクから外す
        let link = self.m[id].link;
        self.m[link.prev].link.next = link.next;
        self.m[link.next].link.prev = link.prev;

        // idの位置にある要素を一番後ろとswap
        let last = self.m.len() - 1;
        self.m.swap(id, last);

        // idの位置に来た要素の前後のリンクを更新
        let link = self.m[id].link;
        if link.prev != Self::NIL { self.m[link.prev].link.next = id; }
        if link.next != Self::NIL { self.m[link.next].link.prev = id; }

        // 削除
        let t = self.m.pop().unwrap();
        t.value
    }

    pub fn insert_prev(&mut self, id: Id, value: T) -> Id {
        unimplemented!()
    }

    pub fn insert_next(&mut self, id: Id, value: T) -> Id {
        assert!(id != Self::TAIL && id < self.m.len());
        let aid = id;
        let cid = self.m[aid].link.next;
        let bid = self.m.len();
        let b = Node { id: bid, link: Link { prev: aid, next: cid }, value, };

        self.m[aid].link.next = bid;
        self.m[cid].link.prev = bid;
        self.m.push(b);
        bid
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item=&T> + '_ {
        struct Iter<'a, T> {
            id: Id,
            list: &'a LinkedList<T>,
        }

        impl<'a, T> Iterator for Iter<'a, T> {
            type Item = &'a T;
            fn next(&mut self) -> Option<Self::Item> {
                self.id = self.list.m[self.id].link.next;
                if self.id == LinkedList::<T>::TAIL || self.id == LinkedList::<T>::NIL { None }
                else { Some(&self.list.m[self.id].value) }
            }
        }

        impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.id = self.list.m[self.id].link.prev;
                if self.id == LinkedList::<T>::HEAD || self.id == LinkedList::<T>::NIL { None }
                else { Some(&self.list.m[self.id].value) }
            }
        }

        Iter { id: Self::HEAD, list: self }
    }
}
