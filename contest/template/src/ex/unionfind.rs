
#[allow(dead_code)]
pub mod unionfind {

pub struct Unionfind { d: Vec<isize> }

impl Unionfind {
    pub fn new(n:usize) -> Unionfind { Unionfind{d: vec![-1; n]} }
    pub fn leader(&mut self, x: usize) -> usize {
        if self.d[x] < 0 { x }
        else {
            self.d[x] = self.leader(self.d[x] as usize) as isize;
            self.d[x] as usize
        }
    }
    pub fn unite(&mut self, x: usize, y: usize) -> bool {
        let (mut x, mut y) = (self.leader(x), self.leader(y));
        if x == y { return false }
        if self.d[x] > self.d[y] { std::mem::swap(&mut x, &mut y) }
        self.d[x] += self.d[y];
        self.d[y] = x as isize;
        true
    }
    pub fn same(&mut self, x: usize, y: usize) -> bool { self.leader(x) == self.leader(y) }
    pub fn size(&mut self, x: usize) -> usize { let p = self.leader(x); -self.d[p] as usize }
}

impl Unionfind {
    pub fn groups(&mut self) -> Vec<Vec<usize>> {
        let n = self.d.len();
        let (mut leader_buf, mut group_size, mut res) = (vec![0; n], vec![0; n], vec![vec![]; n]);
        for i in 0..n { leader_buf[i] = self.leader(i); group_size[leader_buf[i]]+=1; }
        for i in 0..n { res[i].reserve(group_size[i]); }
        for i in 0..n { res[leader_buf[i]].push(i); }
        res.iter().filter(|v|v.len()>0).cloned().collect()
    }
}

}