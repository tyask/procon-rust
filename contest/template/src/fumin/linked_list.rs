#![allow(dead_code)]
use crate::common::*;

type Id = us;

#[derive(Debug, Clone, Copy)]
struct Link {
    prev: Id,
    next: Id,
}

#[derive(Debug)]
struct Node<T> {
    id: Id,
    prev: Id,
    next: Id,
    value: T,
}

impl<T> Node<T> {
    pub fn link(&self) -> Link { Link { prev: self.prev, next: self.next } }
}

#[derive(Debug)]
pub struct LinkedList<T> {
    m: Vec<Node<T>>,
}

impl<T: Default> LinkedList<T> {
    pub fn new() -> Self {
        Self {
            m: vec![
                Node { id: Self::HEAD, prev: Self::NIL, next: Self::TAIL, value: T::default(), },
                Node { id: Self::TAIL, prev: Self::HEAD, next: Self::NIL, value: T::default(), },
            ],
        }
    }
}

impl<T> LinkedList<T> {
    const NIL: Id = us::INF;
    const HEAD: Id = 0;
    const TAIL: Id = 1;

    pub fn push(&mut self, value: T) -> us {
        self.insert_next(self.m[Self::TAIL].prev, value)
    }

    pub fn remove(&mut self, id: Id) {
        assert!(2 <= id && id < self.m.len());
        let link = self.m[id].link();
        self.m[link.prev].next = link.next;
        self.m[link.next].prev = link.prev;
    }

    pub fn insert_next(&mut self, id: Id, value: T) -> Id {
        assert!(id != Self::TAIL && id < self.m.len());
        let aid = id;
        let cid = self.m[aid].next;
        let bid = self.m.len();
        let b = Node { id: bid, prev: aid, next: cid, value, };

        self.m[aid].next = bid;
        self.m[cid].prev = bid;
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
                self.id = self.list.m[self.id].next;
                if self.id == LinkedList::<T>::TAIL || self.id == LinkedList::<T>::NIL { None }
                else { Some(&self.list.m[self.id].value) }
            }
        }

        impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.id = self.list.m[self.id].prev;
                if self.id == LinkedList::<T>::HEAD || self.id == LinkedList::<T>::NIL { None }
                else { Some(&self.list.m[self.id].value) }
            }
        }

        Iter { id: Self::HEAD, list: self }
    }
}