#![allow(dead_code)]
use std::rc::Rc;

type Link<T> = Option<Rc<Node<T>>>;

#[derive(Debug, Clone)]
struct Node<T> {
    next: Link<T>,
    value: T,
}

#[derive(Debug, Clone)]
struct PersistentStack<T> {
    head: Link<T>,
}

impl<T> PersistentStack<T> {
    pub fn new() -> Self { Self { head: None } }

    pub fn push(&self, value: T) -> Self {
        Self { head: Some(Rc::new(Node {
            next: self.head.clone(),
            value,
        }))}
    }

    pub fn pop(&self) -> Self {
        Self { head: self.head.as_ref().and_then(|n|n.next.clone()) }
    }

    pub fn empty(&self) -> bool { self.head.is_none() }
    pub fn top(&self) -> Option<&T> { self.head.as_ref().map(|n|&n.value) }
}

pub struct Iter<'a, T> {
    next: Option<&'a Node<T>>,
}

impl<T> PersistentStack<T> {
    pub fn iter(&self) -> Iter<'_, T> { Iter { next: self.head.as_deref() } }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.next.map(|n| {
            self.next = n.next.as_deref();
            &n.value
        })
    }
}
