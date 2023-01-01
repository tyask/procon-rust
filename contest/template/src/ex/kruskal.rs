
// pub mod kruskal {
// use super::super::unionfind::unionfind::Unionfind;
use super::unionfind::unionfind::Unionfind;

#[derive(Clone,Copy)]
pub struct Edge { u: usize, v: usize, cost: isize }
impl Edge { pub fn new(u: usize, v: usize, cost: isize) -> Edge { Edge{u:u, v:v, cost:cost} } }

pub struct Kraskal { n: usize, edges: Vec<Edge> }
impl Kraskal {
    pub fn new(n: usize) -> Kraskal { Kraskal { n: n, edges: vec![] }}
    pub fn add(&mut self, e: Edge) { self.edges.push(e); }
    pub fn adds(&mut self, u: usize, v: usize, cost: isize) { self.edges.push(Edge::new(u, v, cost)); }

    pub fn run(&mut self) -> Vec<Edge> {
        self.edges.sort_by_key(|e|e.cost);
        let mut uf = Unionfind::new(self.n);
        let mut ret = vec![];
        for &e in &self.edges { if uf.unite(e.u, e.v) { ret.push(e); }}
        return ret;
    }
}

// }