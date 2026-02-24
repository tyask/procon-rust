#![allow(unused_imports)]
use std::{*, collections::*, ops::*, cmp::*, iter::*};
use bitset::BitSetTrait;
use cell::RefCell;
use grid_v::{GridV, ShortestPath};
use itertools::{iproduct, Itertools};
use proconio::{input, fastout};
use common::*;
use fumin::*;
use pt::Dir;
use rand::{seq::{IteratorRandom, SliceRandom}, SeedableRng};
use rc::Rc;
use time::Instant;

fn main() {
    solve();
}

type P = pt::Pt<u16>;
const N: us = 50;
const T: us = 800;
const LIMIT: u128 = 2950*1;

fn dist2(v: P) -> impl Iterator<Item=P> {
    const DS: [P; 13] = [
        P{x:0,y:0},P{x:0,y:1},P{x:0,y:2},P{x:0,y:!0},P{x:0,y:!0-1},
        P{x:1,y:0},P{x:1,y:1},P{x:1,y:!0},
        P{x:2,y:0},
        P{x:!0,y:0},P{x:!0,y:1},P{x:!0,y:!0},
        P{x:!0-1,y:0},
        ];
    DS.iter().map(move|&d|v.wrapping_add(d)).filter(|v|v.x<N.u16()&&v.y<N.u16())
}

struct Io {
    start: Instant,
    _n:us,
    m:us,
    k:us,
    _t:us,
    st: Vec<(P, P)>,
    dist: Vec<u16>,
}

impl Io {
    fn new() -> Self {
        let start = Instant::now();
        input! {n:us,m:us,k:us,t:us,st:[(P,P);m]}
        let dist = st.iter().map(|&(s,t)|s.manhattan_distance(t)).cv();
        Self {
            start,
            _n: n,
            m,
            k,
            _t: t,
            st,
            dist,
        }
    }

    fn is_over(&self, limit: u128) -> bool { self.start.elapsed().as_millis() > limit }
    fn dist(&self, i:us) -> u16 { self.dist[i] }
    fn dist_sum(&self, b: &impl BitSetTrait) -> us {
        b.ones().into_iter().map(|i|self.dist(i).us()).sum::<us>()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionType {
    RailOrStation(SectionType, SectionType, P),
    Wait(us),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Action {
    t: ActionType,
    memo: Str,
}

impl Action {
    fn new(t: ActionType, memo: Str) -> Self {
        Self { t, memo, }
    }
}

impl From<ActionType> for Action {
    fn from(value: ActionType) -> Self { Action::new(value, "".to_owned()) }
}

#[derive(Debug, Clone)]
struct Out {
    actions: Vec<Action>,
    money: us,
    hist_incomes: [us; 8],
    end_tern: us,
}

impl Out {
    fn new() -> Self {
        Self { actions: vec![], money: 0, hist_incomes: [0; 8], end_tern: 0, }
    }

    fn push(&mut self, a: Action) {
        self.actions.push(a);
    }

    fn push_memo(&mut self, memo: Str) {
        if let Some(a) = self.actions.last_mut() {
            a.memo += &memo;
            a.memo.push('\n');
        }
    }

    fn extend(&mut self, acts: &[ActionType]) {
        self.actions.extend(acts.map(|&a|Action::from(a)))
    }

    fn truncate(&mut self, n: us) {
        let mut ret = vec![];
        let mut tern = 0;
        for a in &self.actions {
            match a.t {
                ActionType::RailOrStation(_, _, _) => {
                    if tern + 1 <= n { ret.push(a.clone()); } else { break; }
                    tern += 1;
                },
                ActionType::Wait(x) => {
                    if tern + x <= n { ret.push(a.clone()); }
                    else {
                        ret.push(Action::new(ActionType::Wait(n-tern), "".to_owned()));
                        break;
                    }
                },
            }
        }
    }

    #[fastout]
    fn out(&self) {
        for a in &self.actions {
            if !a.memo.is_empty() { println!("{}", a.memo); }
            match a.t {
                ActionType::RailOrStation(_, r, p) => println!("{} {} {}", r as us, p.x, p.y),
                ActionType::Wait(n) => for _ in 0..n { println!("-1"); },
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, hash::Hash)]
enum SectionType {
    Station,
    LR,
    UD,
    LD,
    LU,
    RU,
    RD,
    Empty,
}

impl SectionType {
    fn is_station(self) -> bool { self == SectionType::Station }
    fn is_empty(self) -> bool { self == SectionType::Empty }
    fn is_rail(self) -> bool { !self.is_station() && !self.is_empty() }

    // v0 -> v1 -> v2 と進むときのv1の線路の形
    fn gen_rail(v0:P, v1:P, v2:P) -> SectionType {
        let d0 = Dir::from(v0 - v1);
        let d1 = Dir::from(v2 - v1);
        let (d0, d1) = if d0 < d1 { (d0, d1) } else { (d1, d0) };
        // R,L,D,U
        match (d0, d1) {
            (Dir::R, Dir::R) => SectionType::LR,
            (Dir::R, Dir::L) => SectionType::LR,
            (Dir::R, Dir::D) => SectionType::RD,
            (Dir::R, Dir::U) => SectionType::RU,
            (Dir::L, Dir::L) => SectionType::LR,
            (Dir::L, Dir::D) => SectionType::LD,
            (Dir::L, Dir::U) => SectionType::LU,
            (Dir::D, Dir::D) => SectionType::UD,
            (Dir::D, Dir::U) => SectionType::UD,
            (Dir::U, Dir::U) => SectionType::UD,
            _ => unreachable!("unreachable: ({:?},{:?},{:?}) ({:?},{:?})", v0, v1, v2, d0, d1),
        }
    }

    fn can_move_to(self, d: Dir) -> bool {
        if self.is_empty() { return false; }
        if self.is_station() { return true; }
        match self {
            SectionType::LR => d.is_or(Dir::L|Dir::R),
            SectionType::UD => d.is_or(Dir::U|Dir::D),
            SectionType::LD => d.is_or(Dir::L|Dir::D),
            SectionType::LU => d.is_or(Dir::L|Dir::U),
            SectionType::RU => d.is_or(Dir::R|Dir::U),
            SectionType::RD => d.is_or(Dir::R|Dir::D),
            _ => unreachable!(),
        }
    }

    fn can_move_d(self, d: Dir) -> bool {
        match self {
            SectionType::LR => d.is_or(Dir::L|Dir::R),
            SectionType::UD => d.is_or(Dir::U|Dir::D),
            SectionType::LD => d.is_or(Dir::L|Dir::D),
            SectionType::LU => d.is_or(Dir::L|Dir::U),
            SectionType::RU => d.is_or(Dir::R|Dir::U),
            SectionType::RD => d.is_or(Dir::R|Dir::D),
            SectionType::Empty => false,
            SectionType::Station => d.is_or((Dir::L|Dir::R) | (Dir::U|Dir::D)),
        }
    }

    fn can_move(t0: SectionType, v0: P, t1: SectionType, v1: P) -> bool {
        let d = Dir::from(v1-v0);
        match d {
            Dir::L => t0.can_move_d(Dir::L) && t1.can_move_d(Dir::R),
            Dir::R => t0.can_move_d(Dir::R) && t1.can_move_d(Dir::L),
            Dir::U => t0.can_move_d(Dir::U) && t1.can_move_d(Dir::D),
            Dir::D => t0.can_move_d(Dir::D) && t1.can_move_d(Dir::U),
            _ => false,
        }
    }

}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Section<B: BitSetTrait> {
    t: SectionType,
    homes: B,
    works: B,
}

impl<B: BitSetTrait> Section<B> {
    fn new(io:&Io) -> Self {
        Self {
            t: SectionType::Empty,
            homes: B::with_capacity(io.m),
            works: B::with_capacity(io.m),
        }
    }
}

#[derive(Debug, Clone)]
struct NextAction<B: BitSetTrait> {
    score: i64,
    actions: Vec<ActionType>,
    tern: us,
    hx: B, // この駅を繋ぐことで駅近になる家の人
    wx: B, // この駅を繋ぐことで駅近になる職場の人
    cost :us,
    income: us,
    curr_income: us,
    income_on_execution: us,
}

impl<B: BitSetTrait> NextAction<B> {
    fn new(score: i64, actions: &Vec<ActionType>, hx: B, wx: B, cost: us, income: us, curr_income:us) -> Self {
        let mut t = Self {
            score,
            actions: vec![],
            tern: 0,
            hx,
            wx,
            cost,
            income,
            curr_income: curr_income,
            income_on_execution: 0,
        };
        t.extend(actions);
        t
    }

    fn merge(&self, a: &Self) -> Self {
        let mut t = Self {
            score: self.score + a.score,
            actions: vec![],
            tern: 0,
            hx: self.hx.bitor(&a.hx),
            wx: self.wx.bitor(&a.wx),
            cost: self.cost + a.cost,
            income: self.income + a.income,
            curr_income: 0, // dummy
            income_on_execution: 0,
        };
        t.extend(&self.actions.iter().chain(&a.actions).cloned().cv());
        t.income_on_execution = self.income_on_execution + a.income_on_execution;
        t
    }


    fn wait(io: &Io, n: us, curr_income:us) -> Self {
        Self {
            score: 0,
            actions: vec![ActionType::Wait(n)],
            tern: n,
            hx: B::with_capacity(io.m),
            wx: B::with_capacity(io.m),
            cost: 0,
            income: 0,
            curr_income: curr_income,
            income_on_execution: curr_income * n,
        }
    }

    fn tern(&self) -> us { self.tern }

    fn income_on_execution(&self) -> us {
        self.curr_income * self.tern() + self.income
    }

    fn push(&mut self, a: ActionType) {
        self.actions.push(a);
        self.tern += match a {
            ActionType::RailOrStation(_, _, _) => 1,
            ActionType::Wait(n) => n,
        };
        self.income_on_execution = self.income_on_execution();
    }

    fn extend(&mut self, a: &[ActionType]) { a.iter().for_each(|a|self.push(*a)); }

    fn station(&self) -> Option<P> {
        self.actions.iter()
            .filter_map(|&a| {
                match a {
                    ActionType::RailOrStation(_, t, v) => {
                        if t.is_station() { return Some(v); }
                    },
                    _ => {},
                }
                None
            })
            .next()
    }
}

fn new_bounded_list<K: PartialOrd+Copy, V:Clone>() -> bounded_sorted_list::BoundedSortedList<K, V> {
    new_bounded_list_n(30)
}

fn new_bounded_list_n<K: PartialOrd+Copy, V:Clone>(n:us) -> bounded_sorted_list::BoundedSortedList<K, V> {
    bounded_sorted_list::BoundedSortedList::new(n)
}

struct Initializer<B: BitSetTrait> {
    hw: GridV<Section<B>>,
    homes_around: GridV<B>,
    works_around: GridV<B>,
    any_around: GridV<B>,
    initial_station_positions: Vec<(P,P,us,us)>,
    initial_action_candidates: Vec<NextAction<B>>,
    beam_depth: us,
    beam_width: us,
}

impl<B: BitSetTrait> Initializer<B> {
    fn new(io: &Io) -> Self {
        let mut hw = GridV::<Section<B>>::with_default(N, N, Section::<B>::new(io));
        for (i,&(s, t)) in io.st.iter().enumerate() {
            hw[s].homes.set(i, true);
            hw[t].works.set(i, true);
        }
        let mut homes_around = GridV::with_default(N, N, B::with_capacity(io.m));
        let mut works_around = GridV::with_default(N, N, B::with_capacity(io.m));
        let mut any_around = GridV::with_default(N, N, B::with_capacity(io.m));
        for v in P::iter(0..N.u16(), 0..N.u16()) {
            homes_around[v] = dist2(v).map(|v|&hw[v].homes).fold(B::with_capacity(io.m), |a,x|a.bitor(x));
            works_around[v] = dist2(v).map(|v|&hw[v].works).fold(B::with_capacity(io.m), |a,x|a.bitor(x));
            any_around[v] = homes_around[v].bitor(&works_around[v]);
        }

        let initial_station_positions = Self::initial_station_positions(io, &homes_around, &works_around);
        let initial_action_candidates = Self::initial_action_candidates(
            &initial_station_positions, &homes_around, &works_around);

        let (mut beam_depth, mut beam_width) = 
            if io.m <= 128 {
                (20, 10)
            } else if io.m <= 128*3 {
                (3, 10)
            } else {
                (2, 10)
            };

        // if let Ok(val) = env::var("BEAM_DEPTH") { beam_depth = val.parse().unwrap(); }
        // if let Ok(val) = env::var("BEAM_WIDTH") { beam_width = val.parse().unwrap(); }

        Self {
            hw,
            homes_around,
            works_around,
            any_around,
            initial_station_positions,
            initial_action_candidates,
            beam_depth,
            beam_width,
        }
    }

    fn initial_station_positions(io: &Io, homes_around: &GridV<B>, works_around: &GridV<B>) -> Vec<(P,P,us,us)> {
        (0..io.m)
            .flat_map(|i| {
                let (s, t) = io.st[i];
                dist2(s).cartesian_product(dist2(t).cv())
            })
            .map(|(s,t)|min_max(s, t))
            .sorted().dedup()
            .filter_map(|(v0,v1)|{
                let (hx0, wx0) = (homes_around[v0], works_around[v0]);
                let (hx1, wx1) = (homes_around[v1], works_around[v1]);
                let cost = 10000 + (v0.manhattan_distance(v1).us() - 1) * 100;
                if cost > io.k { return None; }
                let income = io.dist_sum(&hx0.bitand(&wx1).bitor(&hx1.bitand(&wx0)));
                Some((v0,v1,cost,income))
            })
            .sorted_by_key(|t|Reverse(t.3))
            .take(200)
            .cv()
    }

    fn initial_action_candidates(
        cand: &Vec<(P,P,us,us)>,
        homes_around: &GridV<B>, 
        works_around: &GridV<B>, 
    ) -> Vec<NextAction<B>> {
        let mut m = set::default();
        let mut ret = vec![];
        for v in cand.iter().take(20).flat_map(|t|[t.0,t.1]) {
            if !m.insert(v) { continue; }
            ret.push(NextAction::<B>::new(
                    0,
                    &vec![ActionType::RailOrStation(SectionType::Empty, SectionType::Station, v)],
                    homes_around[v],
                    works_around[v],
                    5000,
                    0,
                    0,
                ));
        }
        ret
    }
}

#[derive(Debug, Clone)]
struct ExploreGrid {
    stations: set<P>,
    stations_or_rails: map<P,SectionType>,
}

impl ExploreGrid {
    fn new() -> Self {
        Self {
            stations: set::default(),
            stations_or_rails: map::default(),
        }
    }

    fn cost2(&self, v0: P, v1: P) -> us {
        let t0 = self[v0];
        let t1 = self[v1];
        if t0.is_empty() {
            if t1.is_empty() { 100 } // 線路を引くコスト
            else if t1.is_station() { 0 } // 駅はそのまま通過できる
            else { 5000 } // 線路を駅に変えるコスト
        } else if t0.is_rail() {
            if t1.is_empty() { us::INF } // 線路から更地へは行けない
            else if SectionType::can_move(t0, v0, t1, v1) { 0 } // 繋がっている線路
            else { 5000 } // 線路を駅に変えるコスト
        } else { // station
            if t1.is_empty() { 100 } // 線路を引くコスト
            else if SectionType::can_move(t0, v0, t1, v1) { 0 } // 繋がっている線路
            else { 5000 } // 線路を駅に変えるコスト
        }
    }

    fn set_t(&mut self, v: P, t: SectionType) {
        if t.is_station() { self.stations.insert(v); } else { self.stations.remove(&v); }
        if t.is_station() || t.is_rail() { self.stations_or_rails.insert(v,t); } else { self.stations_or_rails.remove(&v); }
    }

}

impl Index<P> for ExploreGrid {
    type Output = SectionType;
    fn index(&self, index: P) -> &Self::Output {
        if let Some(t) = self.stations_or_rails.get(&index) { t }
        else { &SectionType::Empty }
    }
}
struct Cache {
    paths: map<Vec<(P,SectionType)>, ExplorePath>,
}

impl Cache {
    fn new() -> Self {
        Self {
            paths: map::default(),
        }
    }

    fn expore_path(&mut self, g: &ExploreGrid) -> &ExplorePath {
        let k = self.key(g);
        self.paths.entry(k)
            .or_insert_with(||Self::explore_path_impl(g))
    }

    fn key(&self, g: &ExploreGrid) -> Vec<(P,SectionType)> {
        g.stations_or_rails.clone().into_iter().sorted_by_key(|t|t.0).cv()
    }

    fn explore_path_impl(g: &ExploreGrid) -> ExplorePath {
        let mut ret = ExplorePath::new();

        let mut q = bheap::new();

        for &v in &g.stations {
            q.push(Reverse((0, v)));
            ret[v].cost = 0;
            ret[v].dist = 0;
        }

        while let Some(Reverse((cost, v))) = q.pop() {
            if ret[v].cost < cost { continue; }
            for d in Dir::VAL4 {
                let nv = v.next(d);
                if !ret.g.is_in_p(nv) { continue; }
                let nc = cost + g.cost2(v, nv);
                if chmin!(ret[nv].cost, nc) {
                    q.push(Reverse((nc, nv)));
                    ret[nv].prev = v;
                    ret[nv].dist = ret[v].dist + if g[nv].is_empty() { 1 } else { 0 };
                    ret[nv].cost2 = cost;
                }
            }
        }

        ret
    }

    fn explore_path_impl_bfs(g: &ExploreGrid) -> ExplorePath {
        let mut ret = ExplorePath::new();

        let mut q = deque::new();

        for &v in &g.stations {
            q.push(v);
            ret[v].cost = 0;
            ret[v].cost2 = 0;
            ret[v].dist = 0;
        }

        while let Some(v) = q.pop_front() {
            for d in Dir::VAL4 {
                let nv = v.next(d);
                if !ret.g.is_in_p(nv) { continue; }
                let c = g.cost2(v, nv);
                if c > 100 { continue; }
                assert!(c == 0 || c == 100);
                if chmin!(ret[nv].cost, ret[v].cost+c) {
                    if c == 0 { q.push_front(nv); } else { q.push_back(nv); }
                    ret[nv].prev = v;
                    ret[nv].dist = ret[v].dist + if g[nv].is_empty() { 1 } else { 0 };
                    ret[nv].cost2 = ret[v].cost;
                }
            }
        }

        ret
    }

}

struct Solver<'a, B: BitSetTrait> {
    io: &'a Io,
    ini: &'a Initializer<B>,
    cache: Rc<RefCell<Cache>>,
    g: ExploreGrid,
    money: us,
    income: us,
    tern :us,
    homes: B, // 家の近くに駅がある人
    works: B, // 職場の近くに駅がある人
    try_count: us,
    out: Out,
    limit: u128,
}

impl<'a, B: BitSetTrait> Solver<'a, B> {
    fn new(io: &'a Io, ini: &'a Initializer<B>, cache: Rc<RefCell<Cache>>) -> Self {
        Self {
            io,
            ini,
            cache,
            g: ExploreGrid::new(), 
            money: io.k,
            income: 0,
            tern: 0,
            homes: B::with_capacity(io.m),
            works: B::with_capacity(io.m),
            try_count: 0,
            out: Out::new(),
            limit: LIMIT,
        }
    }

    fn left_tern(&self) -> us { T - self.tern }
    fn is_limit(&self) -> bool { self.io.is_over(self.limit) }

    fn init(&mut self, rng: &mut impl rand_core::RngCore) {
        // self.apply_next_action(&self.initial_stations(rng));
        let x = self.initial_stations2(rng);
        // let x = self.initial_stations3(rng);
        self.apply_next_action(&x);
    }

    fn solve_until_tern(&mut self, rng: &mut impl rand_core::RngCore, end_tern: us) {

        // 初期位置
        while self.tern < end_tern {
            if self.is_limit() { break; }

            let cand = self.next_candicate_actions();
            if cand.is_empty() { break; }

            // let x = cand.first().unwrap();
            // let x = cand.into_iter().take(3).choose(rng).unwrap();
            let x = self.choose_next_action_beam(&cand);

            let cur_tern = self.tern;
            self.apply_next_action(&x);

            for i in (cur_tern+1)/100..=self.tern/100 { self.out.hist_incomes[i.min(7)] = self.income; }
            self.out.push_memo(format!("#tern={}, money={}, income={}", self.tern, self.money, self.income));
        }
    }


    fn solve(&mut self, rng: &mut impl rand_core::RngCore) {
        self.solve_until_tern(rng, T);

        self.apply_next_action(&NextAction::wait(self.io, T-self.tern, self.income));
        if self.tern > T { self.out.truncate(T); }
        self.out.money = self.money;
        self.out.end_tern = self.tern;
    }

    fn choose_next_action_beam(&mut self, cand: &Vec<NextAction<B>>) -> NextAction<B> {
        let io = self.io;
        if cand.len() == 1 { return cand.first().unwrap().clone(); }
        if io.is_over(LIMIT*9/10) { return cand.first().unwrap().clone(); }

        let (d, w) = (self.ini.beam_depth, self.ini.beam_width);
        if d == 0 { return cand.first().unwrap().clone(); }

        let eval = |s: &NextAction<B>| -> i64 {
            // let t = tern;
            // ((T-t)*(income+s.income) + t * (money+income*s.tern+s.income)) * 100
            //     + s.hx.count_ones() + s.wx.count_ones()
            s.score*100000/s.tern().i64()
            // if tern < 400 {
            //     s.income*100000/(tern+s.tern())
            // } else {
            //     s.score*100000/(tern+s.tern())
            // }
        };

        let mut list = new_bounded_list_n(w);
        for (i,s) in cand.iter().enumerate() {
            list.insert(Reverse(eval(s)), (i, s.clone()));
        }

        for _ in 0..d {
            if list.is_empty() { break; }
            let mut dup = map::<B,(us,NextAction<B>)>::default();
            for (si, ss) in list.values() {
                self.apply_next_action(&ss);
                for s in self.next_candicate_actions() {
                    let ns = ss.merge(&s);
                    self.apply_next_action(&s);
                    dup.entry(ns.hx.bitor(&ns.wx))
                        .and_modify(|e|if eval(&s) < eval(&ns) { *e = (si,ns.clone()); })
                        .or_insert((si,ns));
                    self.revert_next_action(&s);
                }
                self.revert_next_action(&ss);
            }
            let mut nlist = new_bounded_list_n(w);
            for (si,s) in dup.into_values() {
                nlist.insert(Reverse(eval(&s)), (si,s));
            }
            list = nlist;
        }

        if list.is_empty() { return cand.first().unwrap().clone(); }

        let si = list.values().first().unwrap().0;
        cand[si].clone()

    }

    fn apply_next_action(&mut self, s: &NextAction<B>) {
        self.out.extend(&s.actions);
        self.homes.bitor_assign(&s.hx);
        self.works.bitor_assign(&s.wx);
        self.tern += s.tern();
        self.money += s.income_on_execution;// self.income * s.tern() + s.income;
        self.money -= s.cost;
        self.income += s.income;
        self.apply_actions(&s.actions);
    }

    fn revert_next_action(&mut self, s: &NextAction<B>) {
        self.revert_actions(&s.actions);
        self.income -= s.income;
        self.money += s.cost;
        self.money -= s.income_on_execution; // self.income * s.tern() + s.income; // TOTO マージすると壊れる
        self.tern -= s.tern();
        self.works.bitand_assign(&s.wx.rev());
        self.homes.bitand_assign(&s.hx.rev());
        for _ in 0..s.actions.len() { self.out.actions.pop(); }
    }

    fn apply_actions(&mut self, acts: &Vec<ActionType>) {
        for &a in acts {
            match a {
                ActionType::RailOrStation(_, t, v) => self.g.set_t(v, t),
                _ => {},
            }
        }
    }

    fn revert_actions(&mut self, acts: &Vec<ActionType>) {
        for &a in acts.iter().rev() {
            match a {
                ActionType::RailOrStation(pre, _, v) => self.g.set_t(v, pre),
                _ => {},
            }
        }
    }

    fn next_candicate_actions(&self) -> Vec<NextAction<B>> {
        let io = self.io;
        let mut cand = vec![];
        let income = self.income;
        let money = self.money;

        if self.left_tern() == 0 { return cand; }

        // 駅+それなりの距離の線路を作成できるくらいにmoneryが溜まるまでは待機しておく. (もう少しいい方法あるかも)
        if 0 < income && income < 100 && money < 7000 {
            let wait_tern = ceil(7000 - money, income).min(self.left_tern());
            cand.push(NextAction::wait(&io, wait_tern, income));
        } else if 0 < income && money < 5000 {
            let wait_tern = ceil(5000 - money, income).min(self.left_tern());
            cand.push(NextAction::wait(&io, wait_tern, income));
        } else {
            let mut cache = self.cache.borrow_mut();
            let path = cache.expore_path(&self.g);
            // let path = self.explore_path();
            cand.extend(self.next_station1(&path));
            if cand.is_empty() {
                cand.extend(self.next_station2(&path));
                // cand.sort_by_key(|s|Reverse(s.score));
            }

            if cand.is_empty() {
                cand.push(NextAction::wait(&io, 1, income));
            }
        }
        cand
    }

    fn next_station1(&self, path: &ExplorePath) -> Vec<NextAction<B>> {
        let io = self.io;
        let inv_homes = self.homes.rev(); // 家の近くに駅がない人
        let inv_works = self.works.rev(); // 職場の近くに駅がない人

        let cand = inv_homes.ones().into_iter().flat_map(|i|dist2(io.st[i].0))
            .chain(inv_works.ones().into_iter().flat_map(|i|dist2(io.st[i].1)))
            .filter(|&v|!self.g[v].is_station())
            .cset()
            ;

        let mut dup = map::<us,(i64,P,us,us,us)>::default();
        for v in cand {
            let hx = self.homes_around(v).bitand(&inv_homes);
            let wx = self.works_around(v).bitand(&inv_works);

            let d1 = io.dist_sum(&hx.bitand(&self.works));
            let d2 = io.dist_sum(&wx.bitand(&self.homes));
            let income = d1 + d2;
            let cost = path[v].cost2 + 5000;
            // if self.income < 100 && cost > self.money || self.money < 5000 { continue; }

            let construction_tern = path[v].dist;
            let wait_tern = if self.money >= cost || self.money >= 5000 && self.income >= 100 { 0 }
                else { if self.income == 0 { us::INF } else { ceil(self.money.abs_diff(cost), self.income) }};

            // if wait_tern > 20 { continue; }
            let left_tern = T.saturating_sub(self.tern + wait_tern + construction_tern);
            if left_tern == 0 { continue; }

            let x = hx.bitor(&wx).count_ones().i64();
            let profit = (income * left_tern).i64() - cost.i64();
            // if profit < 0 && left_tern <= 100 { continue; }
            // if profit < -1000 && left_tern <= 100 { continue; }
            // if profit < -1000 { continue; }
            if profit < 0 { continue; }
            // if profit < -300 { continue; }

            let score = profit + x * 800;
            let t = (score, v, cost, income, wait_tern);
            dup.entry(income)
                .and_modify(|e|if e.0 < score { *e = t; })
                .or_insert(t);
        }

        let mut ret = new_bounded_list::<Reverse<i64>, (P, us, us, us)>();
        dup.values()
            .for_each(|&(score,v,cost,income,wait_tern)|ret.insert(Reverse(score), (v,cost,income,wait_tern)));

        ret.list().map(|&(Reverse(score), (v, cost, income, wt))|{
            NextAction::<B>::new(
                score,
                &[ActionType::Wait(wt)].iter().cloned().chain(self.to_actions(&path.restore_shortest_path(v))).cv(),
                self.homes_around(v).bitand(&inv_homes),
                self.works_around(v).bitand(&inv_works),
                cost,
                income,
                self.income,
            )
        })
    }

    fn next_station2(&self, path: &ExplorePath) -> Vec<NextAction<B>> {
        let io = self.io;
        let inv_homes = self.homes.rev();
        let inv_works = self.works.rev();
        let inv_any = inv_homes.bitor(&inv_works);

        let cand = inv_homes.bitand(&inv_works).ones().into_iter()
            .flat_map(|i| {
                let (s, t) = io.st[i];
                let hs = dist2(s).filter(|&v|!self.g[v].is_station())
                    .map(|v|(v,self.any_around(v).bitand(&inv_any).count_ones()))
                    .sorted_by_key(|t|Reverse(t.1))
                    .take(3);
                let ws = dist2(t).filter(|&v|!self.g[v].is_station())
                    .map(|v|(v,self.any_around(v).bitand(&inv_any).count_ones()))
                    .sorted_by_key(|t|Reverse(t.1))
                    .take(3)
                    .cv();
                hs.cartesian_product(ws)
                    .map(|((s,sc),(t,tc))|(s,t,sc+tc))
            })
            .map(|(s,t,x)|(min_max(s, t),x))
            .sorted_by_key(|&(vu,x)|Reverse((x,vu)))
            .dedup()
            .take(1000)
            .map(|t|t.0)
            .cset();

        let mut m = map::default();
        for &v in cand.iter().flat_map(|(v,u)|[v,u]) {
            if !m.contains_key(&v) {
                let (hx0, wx0) = (self.homes_around(v).bitand(&inv_homes), self.works_around(v).bitand(&inv_works));
                m.insert(v, (hx0, wx0));
            }
        }

        let mut mx = map::<us,(i64,P,P,us,us)>::default();
        for (v, u) in cand {
            let (hx0, wx0) = &m[&v];
            let (hx1, wx1) = &m[&u];
            let income = io.dist_sum(&hx0.bitand(&wx1)
                    .bitor(&hx1.bitand(&wx0))
                    .bitor(&hx0.bitand(&self.works))
                    .bitor(&wx0.bitand(&self.homes))
                    .bitor(&hx1.bitand(&self.works))
                    .bitor(&wx1.bitand(&self.homes))
                );
            let cost = path[v].cost2 + path[u].cost2 + 10000;
            let construction_tern = path[v].dist + path[u].dist;
            let wait_tern = if self.money >= cost || self.money >= 10000 && self.income >= 100 { 0 }
                else { if self.income == 0 { us::INF } else { ceil(self.money.abs_diff(cost), self.income) } };
            let left_tern = T.saturating_sub(self.tern + construction_tern + wait_tern);
            if left_tern == 0 { continue; }
            if left_tern * income < cost { continue; }
            let x0 = hx0.bitor(&wx0).count_ones().i64();
            let x1 = hx1.bitor(&wx1).count_ones().i64();
            let (v, u, x) = if x0 >= x1 { (v, u, x0) } else { (u, v, x1) };

            let profit = (income * left_tern).i64() - cost.i64();
            // if profit < 0 && left_tern <= 100 { continue; }
            if profit < -1000 { continue; }
            let score = profit + x * 800;

            let t = (score, v, u, income, wait_tern);
            mx.entry(income)
                .and_modify(|e|if e.0 < score { *e = t; })
                .or_insert(t);
        }

        let mut ret = new_bounded_list::<Reverse<i64>, (P, P, us)>();
        mx.values()
            .for_each(|&(score,v,u,_,wt)|ret.insert(Reverse(score), (v,u,wt)));

        ret.list().map(|&(Reverse(score), (v, _, wt))| {
            let ps0 = path.restore_shortest_path(v);
            let (hx0, wx0) = &m[&v];
            NextAction::<B>::new(
                score,
                &[ActionType::Wait(wt)].iter().cloned().chain(self.to_actions(&ps0)).cv(),
                hx0.clone(),
                wx0.clone(),
                path[v].cost2 + 5000,
                io.dist_sum(
                    &hx0.bitand(&self.works)
                        .bitor(&wx0.bitand(&self.homes))
                    ),
                self.income
            )
        })
    }

    fn initial_stations(&self, rng: &mut impl rand_core::RngCore) -> NextAction<B> {
        let cand = &self.ini.initial_station_positions;

        let (v0, v1, cost, income) =
            if self.try_count < 10 { cand[self.try_count.min(cand.len()-1)] }
            else { cand.choose(rng).cloned().unwrap() };

        let (hx0, wx0) = (self.homes_around(v0), self.works_around(v0));
        let (hx1, wx1) = (self.homes_around(v1), self.works_around(v1));
        NextAction::<B>::new(
            0, // dummy
            &self.to_actions(&self.root(v0, v1)),
            hx0.bitor(hx1),
            wx0.bitor(wx1),
            cost,
            income,
            self.income
        )
    }

    fn initial_stations2(&mut self, rng: &mut impl rand_core::RngCore) -> NextAction<B> {
        let cand = self.ini.initial_action_candidates.iter()
            .shuffled(rng)
            .cloned()
            .cv();

        self.choose_next_action_beam(&cand)
    }

    fn initial_stations3(&mut self, rng: &mut impl rand_core::RngCore) -> NextAction<B> {
        let cand = self.ini.initial_station_positions.iter().take(20)
            .map(|&(v0,v1,cost,income)|{
                let (hx0, wx0) = (self.homes_around(v0), self.works_around(v0));
                let (hx1, wx1) = (self.homes_around(v1), self.works_around(v1));
                NextAction::<B>::new(
                    0, // dummy
                    &self.to_actions(&self.root(v0, v1)),
                    hx0.bitor(hx1),
                    wx0.bitor(wx1),
                    cost,
                    income,
                    self.income
                )
            })
            .shuffled(rng)
            .cv();

        self.choose_next_action_beam(&cand)
    }


    fn root(&self, v0: P, v1: P) -> Vec<P> {
        let mut ps = vec![];
        if v0.x <= v1.x {
            for x in v0.x..=v1.x { ps.push(P::new(x, v0.y)); }
        } else {
            for x in (v1.x..=v0.x).rev() { ps.push(P::new(x, v0.y)); }
        }
        if v0.y <= v1.y {
            for y in v0.y+1..=v1.y { ps.push(P::new(v1.x, y)); }
        } else {
            for y in (v1.y..v0.y).rev() { ps.push(P::new(v1.x, y)); }
        }
        ps
    }

    fn to_actions(&self, ps: &Vec<P>) -> Vec<ActionType> {
        if ps.is_empty() { return vec![] }

        let mut acts = vec![];
        for i in [0, ps.len()-1] {
            let v = ps[i];
            if !self.g[v].is_station() {
                acts.push(ActionType::RailOrStation(self.g[v], SectionType::Station, v));
            }
        }

        for i in 0..ps.len()-2 {
            let v0 = ps[i];
            let v1 = ps[i+1];
            let v2 = ps[i+2];
            let t0 = self.g[v0];
            let t1 = self.g[v1];

            if t0.is_empty() {
                if t1.is_empty() {
                    let r = SectionType::gen_rail(v0, v1, v2);
                    acts.push(ActionType::RailOrStation(t1, r, v1));
                }
            } else if t0.is_station() {
                if t1.is_empty() {
                    let r = SectionType::gen_rail(v0, v1, v2);
                    acts.push(ActionType::RailOrStation(t1, r, v1));
                }
            }
        }

        acts
    }

    fn homes_around(&self, v: P) -> &B { &self.ini.homes_around[v] }
    fn works_around(&self, v: P) -> &B { &self.ini.works_around[v] }
    fn any_around(&self, v: P) -> &B { &self.ini.any_around[v] }

}

#[derive(Debug, Clone)]
pub struct ExploreNode {
    cost: us,
    cost2: us, // 最後の1マスのコストを除いたコスト
    prev: P,
    dist: us,
}

pub struct ExplorePath {
    g: GridV<ExploreNode>,
}

impl ExplorePath {
    fn new() -> Self {
        Self {
            g: GridV::with_default(N, N, ExploreNode{cost:us::INF, cost2:us::INF, prev: P::INF, dist: us::INF})
        }
    }

    pub fn restore_shortest_path(&self, mut t: P) -> Vec<P> {
        let mut ps = vec![];
        while t != P::INF { ps.push(t); t = self.g[t].prev; }
        ps.reverse();
        ps
    }
}

impl Index<P> for ExplorePath {
    type Output = ExploreNode;
    fn index(&self, index: P) -> &Self::Output { &self.g[index] }
}
impl IndexMut<P> for ExplorePath {
    fn index_mut(&mut self, index: P) -> &mut Self::Output { &mut self.g[index] }
}

fn solve() {
    let io = Io::new();
    if io.m <= 64 { solve_impl::<bitset::BitSet64>(&io); }
    else if io.m <= 128 { solve_impl::<bitset::BitSet128>(&io); }
    else if io.m <= 128*2 { solve_impl::<bitset::BitSetVec<2>>(&io); }
    else if io.m <= 128*3 { solve_impl::<bitset::BitSetVec<3>>(&io); }
    else if io.m <= 128*4 { solve_impl::<bitset::BitSetVec<4>>(&io); }
    else if io.m <= 128*5 { solve_impl::<bitset::BitSetVec<5>>(&io); }
    else if io.m <= 128*6 { solve_impl::<bitset::BitSetVec<6>>(&io); }
    else if io.m <= 128*7 { solve_impl::<bitset::BitSetVec<7>>(&io); }
    else if io.m <= 128*8 { solve_impl::<bitset::BitSetVec<8>>(&io); }
    else if io.m <= 128*9 { solve_impl::<bitset::BitSetVec<9>>(&io); }
    else if io.m <= 128*10 { solve_impl::<bitset::BitSetVec<10>>(&io); }
    else if io.m <= 128*11 { solve_impl::<bitset::BitSetVec<11>>(&io); }
    else if io.m <= 128*12 { solve_impl::<bitset::BitSetVec<12>>(&io); }
    else { solve_impl::<bitset::BitSetVec<13>>(&io); }
}

fn solve_impl<B: BitSetTrait>(io: &Io)  {
    // solve_impl0::<B>(io);
    solve_impl1::<B>(io);
}

fn solve_impl0<B: BitSetTrait>(io: &Io)  {
    let mut rng = rand_pcg::Pcg64Mcg::from_entropy();
    let ini = Initializer::new(io);
    let mut ret = Out::new();
    let mut cnt = 0;
    let cache = Rc::new(RefCell::new(Cache::new()));
    while !io.is_over(LIMIT) {
        cnt += 1;
        let mut solver = Solver::<B>::new(&io, &ini, Rc::clone(&cache));
        solver.try_count = cnt;
        solver.init(&mut rng);
        solver.solve(&mut rng);
        if ret.money < solver.out.money { ret = solver.out.clone(); }
        // break;
    }

    ret.out();
    let mx = io.st.map(|&(s,t)|s.manhattan_distance(t)).sum();
    // eprintln!("# M={},MAX={},MONEY={},CNT={},END={},INCOMES={:?}",
    //     io.m, mx, ret.money, cnt, ret.end_tern, ret.hist_incomes);
    eprintln!("# M={},MAX={},MONEY={},CNT={}", io.m, mx, ret.money, cnt);

}

fn solve_impl1<B: BitSetTrait>(io: &Io) {
    let mut rng = rand_pcg::Pcg64Mcg::from_entropy();
    let ini = Initializer::new(io);

    let mut solvers = Vec::<Solver<B>>::new();
    let mut cnt0 = 0;
    let cache = Rc::new(RefCell::new(Cache::new()));
    const LIMIT0: u128 = LIMIT/2;
    while !io.is_over(LIMIT0) && cnt0 < 50 {
        let mut solver = Solver::<B>::new(&io, &ini, Rc::clone(&cache));
        solver.try_count = cnt0;
        solver.limit = LIMIT0;

        solver.init(&mut rng);
        solver.solve_until_tern(&mut rng, 300);
        solvers.push(solver);
        cnt0 += 1;
    }
    solvers.sort_by_key(|s|(s.income, s.money));

    let mut ret = Out::new();
    let mut cnt = 0;
    loop {
        cnt += 1;
        let mut solver = if let Some(solver) = solvers.pop() {
            solver
        } else {
            let mut solver = Solver::<B>::new(&io, &ini, Rc::clone(&cache));
            solver.init(&mut rng);
            solver
        };

        solver.limit = LIMIT;
        solver.solve(&mut rng);
        if ret.money < solver.out.money { ret = solver.out.clone(); }
        // break;
        if io.is_over(LIMIT) { break; }
    }

    ret.out();
    let mx = io.dist_sum(&B::with_capacity(io.m).rev());
    // eprintln!("# M={},MAX={},MONEY={},CNT0={},CNT={},END={},INCOMES={:?}",
    //     io.m, mx, ret.money, cnt0, cnt, ret.end_tern, ret.hist_incomes);
    eprintln!("# M={},MAX={},MONEY={},CNT0={},CNT={}", io.m, mx, ret.money, cnt0, cnt);
}

// #CAP(fumin::modint)
pub mod fumin {
pub mod bitset {
use std::convert::TryInto;
use std::hash;
use std::ops::BitAnd;
use std::ops::BitOr;
use itertools::Itertools;

use crate::common::*;

pub trait BitSetTrait: Clone + Copy + std::fmt::Debug + PartialEq + Eq + PartialOrd + Ord + hash::Hash {
    fn with_capacity(n:usize) -> Self;
    fn bitand(&self, other: &Self) -> Self;
    fn bitor(&self, other: &Self) -> Self;
    fn rev(&self) -> Self;
    fn count_ones(&self) -> usize;
    fn ones(&self) -> Vec<usize>;
    fn bitand_assign(&mut self, other: &Self);
    fn bitor_assign(&mut self, other: &Self);
    fn set(&mut self, i: usize, f: bool);
}

#[macro_export] macro_rules! impl_bitset {
    ($t:ty, $name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, hash::Hash)]
        pub struct $name { b: $t, mask: $t }
        impl BitSetTrait for $name {
            fn with_capacity(n:usize) -> Self {
                const SZ: us = std::mem::size_of::<$t>()*8;
                assert!(n <= SZ);
                let mask = if n == SZ { !0 } else { ((1 as $t)<<n)-1 };
                // let mask = if n == 0 { 0 } else { (1 as $t).wrapping_shl((n-1) as u32).wrapping_sub(1) };
                Self { b: 0, mask, }
            }
            fn bitand(&self, other: &Self) -> Self { Self { b: self.b & other.b, mask: self.mask } }
            fn bitor(&self, other: &Self) -> Self { Self { b: self.b | other.b, mask: self.mask } }
            fn rev(&self) -> Self { Self { b: (!self.b) & self.mask, mask: self.mask } }
            fn count_ones(&self) -> usize { self.b.count_ones() as usize }
            fn ones(&self) -> Vec<usize> {
                let mut b = self.b;
                let mut ret = vec![];
                while b != 0 {
                    ret.push(b.trailing_zeros().us());
                    b &= b - 1;
                }
                ret
            }
            fn bitand_assign(&mut self, other: &Self) { self.b &= other.b }
            fn bitor_assign(&mut self, other: &Self) { self.b |= other.b; }
            fn set(&mut self, i: usize, f: bool) {
                if f { self.b |= 1<<i; } else { self.b &= !(1<<i) }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:b}", self.b)
            }
        }
    }
}
impl_bitset!{u64,  BitSet64}
impl_bitset!{u128, BitSet128}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, hash::Hash)]
pub struct BitSetVec<const BLOCK: usize> {
    b: [BitSet128; BLOCK],
    block: us,
}
impl<const BLOCK: usize> BitSetVec<BLOCK> {
    const M: usize = std::mem::size_of::<u128>() * 8;
}
impl<const BLOCK: usize> BitSetTrait for BitSetVec<BLOCK> {
    fn with_capacity(n:usize) -> Self {
        let mut b = [BitSet128::with_capacity(Self::M); BLOCK];
        let block = (n.saturating_sub(1)>>7)+1;
        b[block-1] = BitSet128::with_capacity(n&(Self::M-1));
        b[block..].fill(BitSet128::with_capacity(0));
        Self {
            b, 
            block,
        }
    }
    fn bitand(&self, other: &Self) -> Self {
        let mut t = self.clone();
        t.bitand_assign(other);
        t
    }
    fn bitor(&self, other: &Self) -> Self {
        let mut t = self.clone();
        t.bitor_assign(other);
        t
    }
    fn rev(&self) -> Self {
        let mut t = self.clone();
        for i in 0..self.block { t.b[i] = self.b[i].rev(); }
        t
    }
    fn count_ones(&self) -> usize { self.b.iter().map(|b|b.count_ones()).sum::<us>() }
    fn ones(&self) -> Vec<usize> {
        (0..self.block).flat_map(|i|self.b[i].ones().into_iter().map(move|bi|i*Self::M+bi)).cv()
    }
    fn bitand_assign(&mut self, other: &Self) {
        for i in 0..self.block { self.b[i].bitand_assign(&other.b[i]); }
    }
    fn bitor_assign(&mut self, other: &Self) {
        for i in 0..self.block { self.b[i].bitor_assign(&other.b[i]); }
    }
    fn set(&mut self, i: usize, f: bool) {
        self.b[i>>7].set(i&(Self::M-1), f);
    }
}

}
pub mod grid_v {
#![allow(dead_code)]
use std::{ops::{Index, IndexMut}, cmp::Reverse};

use crate::{common::*, chmin};
use super::pt::{Pt, Dir};


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridV<T> {
    pub g: Vec<T>,
    h: us,
    w: us,
}

impl <T> GridV<T> {
    pub fn is_in_p<N: IntoT<us>>(&self, p: Pt<N>) -> bool { self.is_in_t(p.tuple()) }
    pub fn is_in_t<N: IntoT<us>>(&self, t: (N, N)) -> bool { t.0.into_t() < self.h && t.1.into_t() < self.w }
}

impl <T: Clone + Default> GridV<T> {
    pub fn new(h: us, w: us) -> Self {
        Self { g: vec![T::default(); h * w], h, w, }
    }
}

impl <T: Clone> GridV<T> {
    pub fn with_default(h: us, w: us, v: T) -> Self {
        Self { g: vec![v; h * w], h, w, }
    }
}

impl<T, N: IntoT<us>> Index<N> for GridV<T> {
    type Output = [T];
    fn index(&self, i: N) -> &Self::Output {
        let idx = i.into_t() * self.h;
        &self.g[idx..idx+self.w]
    }
}
impl<T, N: IntoT<us>> IndexMut<N> for GridV<T> {
    fn index_mut(&mut self, i: N) -> &mut Self::Output {
        let idx = i.into_t() * self.h;
        &mut self.g[idx..idx+self.w]
    }
}

impl<T, N: IntoT<us>> Index<(N,N)> for GridV<T> {
    type Output = T;
    fn index(&self, index: (N,N)) -> &Self::Output { &self[index.0.into_t()][index.1.into_t()] }
}
impl<T, N: IntoT<us>> IndexMut<(N,N)> for GridV<T> {
    fn index_mut(&mut self, index: (N,N)) -> &mut Self::Output { &mut self[index.0.into_t()][index.1.into_t()] }
}
impl<T, N: IntoT<us>> Index<Pt<N>> for GridV<T> {
    type Output = T;
    fn index(&self, p: Pt<N>) -> &Self::Output { &self[p.tuple()] }
}
impl<T, N: IntoT<us>> IndexMut<Pt<N>> for GridV<T> {
    fn index_mut(&mut self, p: Pt<N>) -> &mut Self::Output { &mut self[p.tuple()] }
}
impl<T: Clone> From<&Vec<Vec<T>>> for GridV<T> {
    fn from(value: &Vec<Vec<T>>) -> Self {
        let (h, w) = (value.len(), value[0].len());
        GridV{ g: value.iter().cloned().flatten().cv(), h, w }
    }
}

pub struct ShortestPath {
    pub start: Pt<us>,
    pub dist: GridV<i64>,
    pub prev: GridV<Pt<us>>,
}

impl ShortestPath {
    pub fn restore_shortest_path(&self, mut t: Pt<us>) -> Vec<Pt<us>> {
        let mut ps = vec![];
        while t != Pt::<us>::INF { ps.push(t); t = self.prev[t]; }
        ps.reverse();
        assert!(ps[0] == self.start);
        ps
    }
}

}
pub mod pt {
#![allow(dead_code)]
use std::{*, ops::*, iter::Sum};
use itertools::iproduct;
use num_traits::Signed;
use rand::Rng;

use crate::{common::*, enrich_enum, count};


// Pt
#[derive(Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash,Default)]
pub struct Pt<N> { pub x: N, pub y: N }

impl<N:fmt::Display> fmt::Debug for Pt<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.x, self.y)
    }
}

impl<N> Pt<N> {
    pub fn new(x: impl IntoT<N>, y: impl IntoT<N>) -> Pt<N> { Pt{x:x.into_t(), y:y.into_t()} }
    pub fn of(x: N, y: N) -> Pt<N> { Pt{x:x, y:y} }
    pub fn tuple(self) -> (N, N) { (self.x, self.y) }
}
impl Pt<us> {
    pub fn iter(rx: Range<us>, ry: Range<us>) -> impl Iterator<Item=Self> { iproduct!(rx, ry).map(|t|Self::from(t)) }
    pub fn u16(self) -> Pt<u16> { Pt::new(self.x as u16, self.y as u16) }
}
impl Pt<u16> {
    pub fn iter(rx: Range<u16>, ry: Range<u16>) -> impl Iterator<Item=Self> { iproduct!(rx, ry).map(|t|Self::from(t)) }
    pub fn us(self) -> Pt<us> { Pt::new(self.x as us, self.y as us) }
    pub fn next(self, d: Dir) -> Self { self.wrapping_add(d.p()) }
}
impl<N: SimplePrimInt> Pt<N> {
    pub fn norm2(self) -> N   { self.x * self.x + self.y * self.y }
    pub fn on(self, h: Range<N>, w: Range<N>) -> bool { h.contains(&self.x) && w.contains(&self.y) }
    pub fn manhattan_distance(self, p: Pt<N>) -> N { abs_diff(self.x, p.x) + abs_diff(self.y, p.y) }
}
impl<N: SimplePrimInt+FromT<i64>+ToF64> Pt<N> {
    pub fn norm(self) -> f64 { self.norm2().f64().sqrt() }
}
impl<N: Wrapping> Wrapping for Pt<N> {
    fn wrapping_add(self, a: Self) -> Self { Self::of(self.x.wrapping_add(a.x), self.y.wrapping_add(a.y)) }
}
impl Pt<us> {
    pub fn wrapping_sub(self, a: Self) -> Self { Self::of(self.x.wrapping_sub(a.x), self.y.wrapping_sub(a.y)) }
    pub fn wrapping_mul(self, a: Self) -> Self { Self::of(self.x.wrapping_mul(a.x), self.y.wrapping_mul(a.y)) }
    // pub fn next(self, d: Dir) -> Self { self.wrapping_add(d.p()) }
    // pub fn iter_next_4d(self) -> impl Iterator<Item=Self> { Dir::VAL4.iter().map(move|&d|self.next(d)) }
    // pub fn iter_next_8d(self) -> impl Iterator<Item=Self> { Dir::VALS.iter().map(move|&d|self.next(d)) }
    // pub fn prev(self, d: Dir) -> Self { self.wrapping_sub(d.p()) }
}

impl<T: Inf> Inf for Pt<T> {
    const INF: Self  = Pt::<T>{x: T::INF,  y: T::INF};
    const MINF: Self = Pt::<T>{x: T::MINF, y: T::MINF};
}

impl<N: Copy + Signed> Pt<N> {
    // 外積 (ベクトルself, vからなる平行四辺形の符号付面積)
    pub fn cross(&self, v: Pt<N>) -> N {
        self.x * v.y - self.y * v.x
    }

    // couter cross wise (ベクトルself, vが反時計回りかどうか)
    pub fn ccw(&self, v: Pt<N>) -> i32 {
        let a = self.cross(v);
        if a.is_positive() { 1 } // ccw
        else if a.is_negative() { -1 } // cw
        else { 0 } // colinear
    }
}

pub type Radian = f64;
impl Pt<f64> {
    pub fn rot(self, r: Radian) -> Pt<f64> {
        let (x, y) = (self.x, self.y);
        Self::new(r.cos()*x-r.sin()*y, r.sin()*x+r.cos()*y) // 反時計回りにr度回転(rはradian)
    }
}
impl<N: SimplePrimInt+fmt::Display> fmt::Display  for Pt<N> { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{} {}", self.x, self.y) } }
impl<N: SimplePrimInt+fmt::Display> Fmt           for Pt<N> { fn fmt(&self) -> String { format!("{} {}", self.x, self.y) } }
impl<N: AddAssign<N>+Copy> AddAssign<Pt<N>> for Pt<N> { fn add_assign(&mut self, rhs: Pt<N>) { self.x += rhs.x; self.y += rhs.y; } }
impl<N: SubAssign<N>+Copy> SubAssign<Pt<N>> for Pt<N> { fn sub_assign(&mut self, rhs: Pt<N>) { self.x -= rhs.x; self.y -= rhs.y; } }
impl<N: AddAssign<N>+Copy> AddAssign<N>     for Pt<N> { fn add_assign(&mut self, rhs: N) { self.x += rhs; self.y += rhs; } }
impl<N: SubAssign<N>+Copy> SubAssign<N>     for Pt<N> { fn sub_assign(&mut self, rhs: N) { self.x -= rhs; self.y -= rhs; } }
impl<N: MulAssign<N>+Copy> MulAssign<N>     for Pt<N> { fn mul_assign(&mut self, rhs: N) { self.x *= rhs; self.y *= rhs; } }
impl<N: DivAssign<N>+Copy> DivAssign<N>     for Pt<N> { fn div_assign(&mut self, rhs: N) { self.x /= rhs; self.y /= rhs; } }
impl<N: RemAssign<N>+Copy> RemAssign<N>     for Pt<N> { fn rem_assign(&mut self, rhs: N) { self.x %= rhs; self.y %= rhs; } }
impl<N: AddAssign<N>+Copy> Add<Pt<N>>       for Pt<N> { type Output = Pt<N>; fn add(mut self, rhs: Pt<N>) -> Self::Output { self += rhs; self } }
impl<N: SubAssign<N>+Copy> Sub<Pt<N>>       for Pt<N> { type Output = Pt<N>; fn sub(mut self, rhs: Pt<N>) -> Self::Output { self -= rhs; self } }
impl<N: AddAssign<N>+Copy> Add<N>           for Pt<N> { type Output = Pt<N>; fn add(mut self, rhs: N) -> Self::Output { self += rhs; self } }
impl<N: SubAssign<N>+Copy> Sub<N>           for Pt<N> { type Output = Pt<N>; fn sub(mut self, rhs: N) -> Self::Output { self -= rhs; self } }
impl<N: MulAssign<N>+Copy> Mul<N>           for Pt<N> { type Output = Pt<N>; fn mul(mut self, rhs: N) -> Self::Output { self *= rhs; self } }
impl<N: DivAssign<N>+Copy> Div<N>           for Pt<N> { type Output = Pt<N>; fn div(mut self, rhs: N) -> Self::Output { self /= rhs; self } }
impl<N: RemAssign<N>+Copy> Rem<N>           for Pt<N> { type Output = Pt<N>; fn rem(mut self, rhs: N) -> Self::Output { self %= rhs; self } }
impl<N: SimplePrimInt+FromT<is>> Neg        for Pt<N> { type Output = Pt<N>; fn neg(mut self) -> Self::Output { self *= N::from_t(-1); self } }
impl<N: SimplePrimInt+Default> Sum          for Pt<N> { fn sum<I: Iterator<Item=Self>>(iter: I) -> Self { iter.fold(Self::default(), |a, b| a + b) } }

impl<N: SimplePrimInt+FromT<is>+proconio::source::Readable<Output=N>+IntoT<N>> proconio::source::Readable for Pt<N> {
    type Output = Pt<N>;
    fn read<R: io::BufRead, S: proconio::source::Source<R>>(source: &mut S) -> Self::Output {
        Pt::new(N::read(source), N::read(source))
    }
}
impl<T> From<(T, T)> for Pt<T> {
    fn from(t: (T, T)) -> Self { Self::of(t.0, t.1) }
}

enrich_enum! {
    pub enum Dir {
        R, L, D, U, RU, RD, LD, LU,
    }
}

impl Dir {
    pub const C4: [char; 4] = ['R', 'L', 'D', 'U'];
    pub const VAL4: [Self; 4] = [Self::R,Self::L,Self::D,Self::U];
    pub const P8: [Pt<u16>; 8] = [
        Pt::<u16>{x:0,y:1},Pt::<u16>{x:0,y:!0},Pt::<u16>{x:1,y:0},Pt::<u16>{x:!0,y:0},
        Pt::<u16>{x:1,y:1},Pt::<u16>{x:1,y:!0},Pt::<u16>{x:!0,y:1},Pt::<u16>{x:!0,y:!0},
        ];
    pub const REV8: [Self; 8] = [
        Self::L,Self::R,Self::U,Self::D,
        Self::LD,Self::LU,Self::RU,Self::RD,
        ];
    pub const RROT90: [Self; 4] = [Self::D,Self::U,Self::L,Self::R];
    pub const LROT90: [Self; 4] = [Self::U,Self::D,Self::R,Self::L];
    pub const RROT45: [Self; 8] = [
        Self::RD,Self::LU,Self::LD,Self::RU,
        Self::R,Self::D,Self::L,Self::U,
        ];
    pub const LROT45: [Self; 8] = [
        Self::RU,Self::LD,Self::RD,Self::LU,
        Self::U,Self::R,Self::D,Self::L,
        ];

    #[inline] pub const fn c(self) -> char   { Self::C4[self.id()] }
    #[inline] pub const fn p(self) -> Pt<u16> { Self::P8[self.id()] }
    #[inline] pub const fn rev(self) -> Self { Self::REV8[self.id()] }
    #[inline] pub const fn rrot90(self) -> Self { Self::RROT90[self.id()] }
    #[inline] pub const fn lrot90(self) -> Self { Self::LROT90[self.id()] }
    #[inline] pub const fn rrot45(self) -> Self { Self::RROT45[self.id()] }
    #[inline] pub const fn lrot45(self) -> Self { Self::LROT45[self.id()] }

    // #[inline] pub fn dir(a: Pt<us>, b: Pt<us>) -> Self { (b.wrapping_sub(a)).into() } // a -> b
    #[inline] pub fn rng4(rng: &mut impl rand_core::RngCore) -> Dir { Self::VALS[rng.gen_range(0..4)] }
    #[inline] pub fn rng8(rng: &mut impl rand_core::RngCore) -> Dir { Self::VALS[rng.gen_range(0..8)] }

}
// impl From<Pt<us>> for Dir { fn from(value: Pt<us>) -> Self { Self::P8.pos(&value).unwrap().into() } }
impl From<Pt<u16>> for Dir { fn from(value: Pt<u16>) -> Self { Self::P8.pos(&value).unwrap().into() } }
impl From<char>   for Dir { fn from(value: char) -> Self { Self::C4.pos(&value).unwrap().into() } }


}
pub mod enrich_enum {
#![allow(dead_code)]

#[macro_export]
macro_rules! count {
     () => (0usize);
     ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
}

#[macro_export]
macro_rules! enrich_enum {
    ($(#[$meta:meta])* $vis:vis enum $name:ident { $($e:ident,)* }) => {

        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, hash::Hash)]
        $(#[$meta])*
        $vis enum $name {
            $($e,)*
        }
        impl $name {
            pub const VALS: [Self; count!($($e)*)] = [$(Self::$e,)*];
            #[inline] pub const fn id(self) -> us { self as us }
            #[inline] pub const fn b(self) -> u32 { 1<<self.id() }
            #[inline] pub const fn is_or(self, b: u32) -> bool { self.b() & b > 0 }
            #[inline] pub const fn from_id(id: us) -> Self { Self::VALS[id] }
            #[inline] pub const fn from_bit(b: u32) -> Self {
                assert!(b.count_ones() <= 1);
                Self::from_id(b.trailing_zeros() as usize + 1)
            }
        }
        impl BitOr for $name {
            type Output = u32;
            fn bitor(self, rhs: Self) -> Self::Output { self.b() | rhs.b() }
        }

        impl From<us>  for $name { fn from(value: us) -> Self { Self::from_id(value) } }
        impl From<u32> for $name { fn from(value: u32) -> Self { Self::from_bit(value) } }
    };
}
}
pub mod dijkstra {
#![allow(dead_code)]
use std::{*, cmp::Reverse};
use crate::{common::*, chmin};

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub from: us,
    pub to:   us,
    pub cost: i64
}
pub struct Dijkstra {
    pub g: Vec<Vec<Edge>>,
    pub dist: Vec<i64>,
    pub prev: Vec<us>,
}

impl Edge {
    pub fn new(from: us, to: us, cost: i64) -> Self { Self { from, to, cost } }
    pub fn rev(self) -> Self { let mut r = self.clone(); mem::swap(&mut r.from, &mut r.to); r }
}

impl Dijkstra {
    pub fn new(n: us) -> Self { Self { g: vec![vec![]; n], dist: vec![0;n], prev: vec![0;n] }}
    pub fn add(&mut self, e: Edge) -> &mut Self { self.g[e.from].push(e); self }
    pub fn add2(&mut self, e: Edge) -> &mut Self { self.add(e).add(e.rev()) }

    pub fn run(&mut self, s: us) -> &Vec<i64> {
        type P = (i64, us); // cost, node

        self.dist.fill(i64::INF);
        self.prev.fill(us::INF);

        let g = &self.g;
        let dist = &mut self.dist;
        let prev = &mut self.prev;
        let mut que = bheap::new();

        dist[s] = 0;
        que.push(Reverse((0, s)));
        while let Some(Reverse((cost, v))) = que.pop() {
            if dist[v] < cost { continue }
            for e in &g[v] {
                let nc = cost + e.cost;
                if chmin!(dist[e.to], nc) {
                    que.push(Reverse((dist[e.to], e.to)));
                    prev[e.to] = v;
                }
            }
        }

        dist
    }

    pub fn restore_shortest_path(&self, mut t: us) -> Vec<us> {
        let mut p = vec![];
        while t != us::INF { p.push(t); t = self.prev[t]; }
        p.reverse();
        p
    }

}
}
pub mod bounded_sorted_list {
#![allow(dead_code)]

use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
struct Entry<K, V> {
    k: K,
    v: V,
}

impl<K: PartialOrd, V> Ord for Entry<K, V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<K: PartialOrd, V> PartialOrd for Entry<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.k.partial_cmp(&other.k)
    }
}

impl<K: PartialEq, V> PartialEq for Entry<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.k.eq(&other.k)
    }
}

impl<K: PartialEq, V> Eq for Entry<K, V> {}

/// K が小さいトップn個を保持
#[derive(Clone, Debug)]
pub struct BoundedSortedList<K: PartialOrd + Copy, V: Clone> {
    que: BinaryHeap<Entry<K, V>>,
    size: usize,
}

impl<K: PartialOrd + Copy, V: Clone> BoundedSortedList<K, V> {
    pub fn new(size: usize) -> Self {
        Self {
            que: BinaryHeap::with_capacity(size),
            size,
        }
    }
    pub fn can_insert(&self, k: K) -> bool {
        self.que.len() < self.size || self.que.peek().unwrap().k > k
    }
    pub fn insert(&mut self, k: K, v: V) {
        if self.que.len() < self.size {
            self.que.push(Entry { k, v });
        } else if let Some(mut top) = self.que.peek_mut() {
            if top.k > k {
                top.k = k;
                top.v = v;
            }
        }
    }
    pub fn list(self) -> Vec<(K, V)> {
        let v = self.que.into_sorted_vec();
        v.into_iter().map(|e| (e.k, e.v)).collect()
    }
    pub fn values(self) -> Vec<V> {
        self.que.into_sorted_vec().into_iter().map(|e|e.v).collect()
    }
    pub fn len(&self) -> usize { self.que.len() }
    pub fn is_empty(&self) -> bool { self.que.is_empty() }

}
}
}

pub mod common {
#![allow(dead_code, unused_imports, unused_macros, non_snake_case, non_camel_case_types)]
use std::{*, ops::*, collections::*, iter::{Sum, FromIterator}};
use hash::BuildHasherDefault;
use itertools::Itertools;
use ::num::{One, Zero};
use rustc_hash::FxHasher;

pub type us        = usize;
pub type is        = isize;
pub type us1       = proconio::marker::Usize1;
pub type is1       = proconio::marker::Isize1;
pub type chars     = proconio::marker::Chars;
pub type bytes     = proconio::marker::Bytes;
pub type Str       = String;
pub type map<K,V>  = HashMap<K,V,BuildHasherDefault<FxHasher>>;
pub type bmap<K,V> = BTreeMap<K,V>;
pub type set<V>    = HashSet<V, BuildHasherDefault<FxHasher>>;
pub type bset<V>   = BTreeSet<V>;
pub type bheap<V>  = BinaryHeap<V>;
pub type deque<V>  = VecDeque<V>;

pub trait FromT<T> { fn from_t(t: T) -> Self; }
pub trait IntoT<T> { fn into_t(self) -> T; }

// PrimNum
pub trait SimplePrimInt:
        Copy
        + PartialOrd<Self>
        + Add<Output=Self>
        + Sub<Output=Self>
        + Mul<Output=Self>
        + Div<Output=Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Default
        + PartialEq
        + Zero
        + One
{
    fn two() -> Self { Self::one() + Self::one() }
}
 
pub trait ExPrimInt: SimplePrimInt
        + Rem<Output=Self>
        + RemAssign
        + FromT<us>
{}


#[macro_export] macro_rules! impl_prim_num {
    ($($t:ty),*) => {$(
        impl SimplePrimInt for $t { }
        impl ExPrimInt     for $t { }
        impl FromT<us>   for $t { fn from_t(n: us) -> Self { n as $t } }
        impl FromT<is>   for $t { fn from_t(n: is) -> Self { n as $t } }
        impl IntoT<us>   for $t { fn into_t(self)  -> us   { self as us  } }
        impl IntoT<is>   for $t { fn into_t(self)  -> is   { self as is  } }
        impl IntoT<f64>  for $t { fn into_t(self)  -> f64  { self as f64 } }
        impl IntoT<u8>   for $t { fn into_t(self)  -> u8   { self as u8  } }
        impl IntoT<u16>  for $t { fn into_t(self)  -> u16  { self as u16  } }
        impl IntoT<u32>  for $t { fn into_t(self)  -> u32  { self as u32 } }
        impl IntoT<u64>  for $t { fn into_t(self)  -> u64  { self as u64 } }
        impl IntoT<i32>  for $t { fn into_t(self)  -> i32  { self as i32 } }
        impl IntoT<i64>  for $t { fn into_t(self)  -> i64  { self as i64 } }
        impl IntoT<char> for $t { fn into_t(self)  -> char { (self as u8) as char } }
    )*}
}
impl_prim_num! {isize, i8, i32, i64, usize, u8, u16, u32, u64, f32, f64}

pub trait ToUs   { fn us(self) -> us; }
pub trait ToIs   { fn is(self) -> is; }
pub trait ToI64  { fn i64(self) -> i64; }
pub trait ToF64  { fn f64(self) -> f64; }
pub trait ToU8   { fn u8(self) -> u8; }
pub trait ToU16   { fn u16(self) -> u16; }
pub trait ToU32  { fn u32(self) -> u32; }
pub trait ToI32  { fn i32(self) -> i32; }
pub trait ToChar { fn char(self) -> char; }

impl<T: IntoT<us>>   ToUs   for T { fn us(self)   -> us   { self.into_t() } }
impl<T: IntoT<is>>   ToIs   for T { fn is(self)   -> is   { self.into_t() } }
impl<T: IntoT<i64>>  ToI64  for T { fn i64(self)  -> i64  { self.into_t() } }
impl<T: IntoT<f64>>  ToF64  for T { fn f64(self)  -> f64  { self.into_t() } }
impl<T: IntoT<u8>>   ToU8   for T { fn u8(self)   -> u8   { self.into_t() } }
impl<T: IntoT<u16>>  ToU16  for T { fn u16(self)  -> u16  { self.into_t() } }
impl<T: IntoT<u32>>  ToU32  for T { fn u32(self)  -> u32  { self.into_t() } }
impl<T: IntoT<i32>>  ToI32  for T { fn i32(self)  -> i32  { self.into_t() } }
impl<T: IntoT<char>> ToChar for T { fn char(self) -> char { self.into_t() } }
impl IntoT<us>   for char  { fn into_t(self) -> us   { self as us } }
impl IntoT<is>   for char  { fn into_t(self) -> is   { self as is } }
impl IntoT<u8>   for char  { fn into_t(self) -> u8   { self as u8 } }
impl IntoT<u32>  for char  { fn into_t(self) -> u32  { self as u32 } }
impl IntoT<i32>  for char  { fn into_t(self) -> i32  { self as i32 } }
impl IntoT<u64>  for char  { fn into_t(self) -> u64  { self as u64 } }
impl IntoT<i64>  for char  { fn into_t(self) -> i64  { self as i64 } }
impl IntoT<char> for char  { fn into_t(self) -> char { self } }
impl IntoT<char> for &char { fn into_t(self) -> char { *self } }

pub trait Inf {
    const INF: Self;
    const MINF: Self;
}
impl Inf for us  {
    const INF: Self = std::usize::MAX / 4;
    const MINF: Self = 0;
}
impl Inf for u8  {
    const INF: Self = std::u8::MAX;
    const MINF: Self = 0;
}
impl Inf for u16  {
    const INF: Self = std::u16::MAX;
    const MINF: Self = 0;
}
impl Inf for is {
    const INF: Self = std::isize::MAX / 4;
    const MINF: Self = -Self::INF;
}
impl Inf for i64 {
    const INF: Self = std::i64::MAX / 4;
    const MINF: Self = -Self::INF;
}

pub trait Wrapping {
    fn wrapping_add(self, a: Self) -> Self;
}
impl Wrapping for u8  { fn wrapping_add(self, a: Self) -> Self { self.wrapping_add(a) } }
impl Wrapping for u16 { fn wrapping_add(self, a: Self) -> Self { self.wrapping_add(a) } }
impl Wrapping for us  { fn wrapping_add(self, a: Self) -> Self { self.wrapping_add(a) } }
impl Wrapping for is  { fn wrapping_add(self, a: Self) -> Self { self.wrapping_add(a) } }
impl Wrapping for i64 { fn wrapping_add(self, a: Self) -> Self { self.wrapping_add(a) } }

// Utilities
#[macro_export] macro_rules! or    { ($cond:expr;$a:expr,$b:expr) => { if $cond { $a } else { $b } }; }
#[macro_export] macro_rules! chmax { ($a:expr,$b:expr) => { { let v = $b; if $a < v { $a = v; true } else { false } } } }
#[macro_export] macro_rules! chmin { ($a:expr,$b:expr) => { { let v = $b; if $a > v { $a = v; true } else { false } } } }
#[macro_export] macro_rules! add   { ($a:expr,$b:expr) => { { let v = $b; $a += v; } } }
#[macro_export] macro_rules! sub   { ($a:expr,$b:expr) => { { let v = $b; $a -= v; } } }
#[macro_export] macro_rules! mul   { ($a:expr,$b:expr) => { { let v = $b; $a *= v; } } }
#[macro_export] macro_rules! div   { ($a:expr,$b:expr) => { { let v = $b; $a /= v; } } }
#[macro_export] macro_rules! rem   { ($a:expr,$b:expr) => { { let v = $b; $a %= v; } } }

pub fn abs_diff<T:PartialOrd+Sub<Output=T>>(n1: T, n2: T) -> T { if n1 >= n2 { n1 - n2 } else { n2 - n1 } }
pub fn floor<N: SimplePrimInt>(a: N, b: N) -> N { a / b }
pub fn ceil<N: SimplePrimInt>(a: N, b: N) -> N { (a + b - N::one()) / b }
pub fn asc <T:Ord>(a: &T, b: &T) -> cmp::Ordering { a.cmp(b) }
pub fn desc<T:Ord>(a: &T, b: &T) -> cmp::Ordering { b.cmp(a) }
pub fn to_int<T:Zero+One>(a: bool) -> T { if a { T::one() } else { T::zero() } }
pub fn min_max<T: Ord+Copy>(a: T, b: T) -> (T, T) { (cmp::min(a,b), cmp::max(a,b)) }
pub fn bin_search<T: ExPrimInt+Shr<Output=T>>(mut ok: T, mut ng: T, f: impl Fn(T)->bool) -> T {
    while abs_diff(ok, ng) > T::one() {
        let m = (ok + ng) >> T::one();
        if f(m) { ok = m; } else { ng = m; }
    }
    ok
}

pub trait IterTrait : Iterator {
    fn counts<C>(&mut self) -> CountIter<Self, C>
        where
            Self: Sized,
            Self::Item: hash::Hash + Eq + Clone,
            C: Eq + AddAssign<C> + One + Default + Copy,
    {
        CountIter::new(self)
    }
    fn grouping_to_bmap<'a, K:Ord+Clone, V>(&'a mut self, get_key: impl Fn(&Self::Item)->K, get_val: impl Fn(&Self::Item)->V) -> bmap<K, Vec<V>> {
        self.fold(bmap::<_,_>::new(), |mut m, x| { m.entry(get_key(&x)).or_default().push(get_val(&x)); m })
    }
    fn grouping_to_map<K:Eq+hash::Hash+Clone, V>(&mut self, get_key: impl Fn(&Self::Item)->K, get_val: impl Fn(&Self::Item)->V) -> map<K, Vec<V>> {
        self.fold(map::<_,_>::default(), |mut m, x| { m.entry(get_key(&x)).or_default().push(get_val(&x)); m })
    }
    fn cv(&mut self) -> Vec<Self::Item> { self.collect_vec() }

    fn shuffled(self, rng: &mut impl rand_core::RngCore) -> vec::IntoIter<Self::Item> where Self: Sized {
        use rand::seq::SliceRandom;
        let mut v = Vec::from_iter(self);
        v.shuffle(rng);
        v.into_iter()
    }

}

pub struct CountIter<I: Iterator, C> {
    nexts: deque<(I::Item, C)>,
}
impl<I: Iterator, C> CountIter<I, C>
    where
        I::Item: hash::Hash+Eq+Clone,
        C: Eq + AddAssign<C> + One + Default + Copy,
        {
    pub fn new(iter: &mut I) -> Self {
        let mut cnt = map::default();
        let mut keys = Vec::new();
        while let Some(e) = iter.next() {
            *cnt.entry(e.clone()).or_default() += C::one();
            if cnt[&e] == C::one() { keys.push(e); }
        }
        let nexts = deque::from_iter(
            keys.into_iter().map(|k| { let c = cnt[&k]; (k,c) } ));
        Self { nexts }
    }
}
impl<I: Iterator, C> Iterator for CountIter<I, C> where I::Item: hash::Hash + Eq + Clone {
    type Item = (I::Item, C);
    fn next(&mut self) -> Option<Self::Item> {
        self.nexts.pop_front()
    }
}

pub trait CharIterTrait<T: IntoT<char>> : Iterator<Item=T> {
    fn cstr(&mut self) -> String { self.map(|c|c.into_t()).collect::<Str>() }
}
pub trait HashIterTrait : Iterator where Self::Item: Eq+hash::Hash {
    fn cset(&mut self) -> set<Self::Item> { self.collect::<set<_>>() }
}
pub trait OrdIterTrait : Iterator where Self::Item: Ord {
    fn cbset(&mut self) -> bset<Self::Item> { self.collect::<bset<_>>() }
}
pub trait PairHashIterTrait<T, U> : Iterator<Item=(T,U)> where T: Eq+hash::Hash {
    fn cmap(&mut self) -> map<T, U> { self.collect::<map<_,_>>() }
}
pub trait PairOrdIterTrait<T, U> : Iterator<Item=(T,U)> where T: Ord {
    fn cbmap(&mut self) -> bmap<T, U> { self.collect::<bmap<_,_>>() }
}

impl<I> IterTrait     for I where I: Iterator { }
impl<I, T: IntoT<char>> CharIterTrait<T> for I where I: Iterator<Item=T> { }
impl<I> HashIterTrait for I where I: Iterator, Self::Item: Eq+hash::Hash { }
impl<I> OrdIterTrait  for I where I: Iterator, Self::Item: Ord { }
impl<I, T, U> PairHashIterTrait<T, U> for I where I: Iterator<Item=(T,U)>, T: Eq+hash::Hash { }
impl<I, T, U> PairOrdIterTrait<T, U>  for I where I: Iterator<Item=(T,U)>, T: Ord { }


// Vec
pub trait VecCount<T> { fn count(&self, f: impl FnMut(&T)->bool) -> us; }
impl<T> VecCount<T> for [T] { fn count(&self, mut f: impl FnMut(&T)->bool) -> us { self.iter().filter(|&x|f(x)).count() } }

pub trait VecMax<T> { fn vmax(&self) -> T; }
impl<T:Clone+Ord> VecMax<T> for [T] { fn vmax(&self) -> T  { self.iter().cloned().max().unwrap() } }

pub trait VecMin<T> { fn vmin(&self) -> T; }
impl<T:Clone+Ord> VecMin<T> for [T] { fn vmin(&self) -> T  { self.iter().cloned().min().unwrap() } }

pub trait VecSum<T> { fn sum(&self) -> T; }
impl<T:Clone+Sum<T>> VecSum<T> for [T] { fn sum(&self)  -> T  { self.iter().cloned().sum::<T>() } }

pub trait VecStr<T> { fn str(&self) -> Str; }
impl<T:ToString> VecStr<T> for [T] { fn str(&self)  -> Str { self.iter().map(|x|x.to_string()).collect::<Str>() } }

pub trait VecMap<T> { fn map<U>(&self, f: impl FnMut(&T)->U) -> Vec<U>; }
impl<T> VecMap<T> for [T] { fn map<U>(&self, mut f: impl FnMut(&T)->U) -> Vec<U> { self.iter().map(|x|f(x)).cv() } }

pub trait VecPos<T> { fn pos(&self, t: &T) -> Option<us>; }
impl<T:Eq> VecPos<T> for [T] { fn pos(&self, t: &T) -> Option<us> { self.iter().position(|x|x==t) } }

pub trait VecRpos<T> { fn rpos(&self, t: &T) -> Option<us>; }
impl<T:Eq> VecRpos<T> for [T] { fn rpos(&self, t: &T) -> Option<us> { self.iter().rposition(|x|x==t) } }

pub trait VecSet<T> { fn set(&mut self, i: us, t: T) -> T; }
impl<T> VecSet<T> for [T] { fn set(&mut self, i: us, mut t: T) -> T { std::mem::swap(&mut self[i], &mut t); t } }

// Deque
pub trait DequePush<T> { fn push(&mut self, t: T); }
impl<T> DequePush<T> for VecDeque<T> { fn push(&mut self, t: T) { self.push_back(t); } }

pub trait DequePop<T> { fn pop(&mut self) -> Option<T>; }
impl<T> DequePop<T> for VecDeque<T> { fn pop(&mut self) -> Option<T> { self.pop_back() } }

pub trait Identify {
    type Ident;
    fn ident(&self) -> Self::Ident;
    fn ident_by(&self, s: &str) -> Self::Ident;
}

impl<T: IntoT<u8> + Copy> Identify for T {
    type Ident = us;

    fn ident(&self) -> us {
        let c = self.into_t();
        if b'0' <= c && c <= b'9'      { (c - b'0').us() }
        else if b'a' <= c && c <= b'z' { (c - b'a').us() }
        else if b'A' <= c && c <= b'Z' { (c - b'A').us() }
        else { 0 }
    }

    fn ident_by(&self, s: &str) -> us {
        let b = self.into_t();
        s.chars().position(|c|c==b.char()).expect("error")
    }
}
impl<T: Identify> Identify for [T] {
    type Ident = Vec<T::Ident>;
    fn ident(&self) -> Self::Ident { self.iter().map(|x|x.ident()).collect_vec() }
    fn ident_by(&self, s: &str) -> Self::Ident { self.iter().map(|x|x.ident_by(s)).collect_vec() }
}
impl Identify for &str {
    type Ident = Vec<us>;
    fn ident(&self) -> Self::Ident { self.as_bytes().ident() }
    fn ident_by(&self, s: &str) -> Self::Ident { self.as_bytes().ident_by(s) }
}
impl Identify for String {
    type Ident = Vec<us>;
    fn ident(&self) -> Self::Ident { self.as_bytes().ident() }
    fn ident_by(&self, s: &str) -> Self::Ident { self.as_bytes().ident_by(s) }
}

pub trait ToC: IntoT<u8> + Copy {
    fn to_c_by(self, ini: char) -> char { (ini.u8() + self.u8()).char() }
}
impl<T: IntoT<u8>+Copy> ToC for T {}

trait Joiner { fn join(self, sep: &str) -> String; }
impl<It: Iterator<Item=String>> Joiner for It { fn join(self, sep: &str) -> String { self.collect::<Vec<_>>().join(sep) } }

pub trait Fmt { fn fmt(&self) -> String; }
macro_rules! fmt_primitive { ($($t:ty),*) => { $(impl Fmt for $t { fn fmt(&self) -> String { self.to_string() }})* } }

fmt_primitive! {
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128,
    usize, isize, f32, f64, char, &str, String, bool
}

impl<T: Fmt> Fmt for [T]         { fn fmt(&self) -> String { self.iter().map(|e| e.fmt()).join(" ") } }
impl<T: Fmt> Fmt for VecDeque<T> { fn fmt(&self) -> String { self.iter().map(|e| e.fmt()).join(" ") } }
impl<T: Fmt> Fmt for set<T>      { fn fmt(&self) -> String { self.iter().map(|e| e.fmt()).join(" ") } }
impl<T: Fmt> Fmt for bset<T>     { fn fmt(&self) -> String { self.iter().map(|e| e.fmt()).join(" ") } }

#[macro_export] macro_rules! fmt {
    ($a:expr, $($b:expr),*) => {{ format!("{} {}", fmt!(($a)), fmt!($($b),*)) }};
    ($a:expr)               => {{ ($a).fmt() }};

    (@debug $a:expr, $($b:expr),*) => {{ format!("{} {}", fmt!(@debug ($a)), fmt!(@debug $($b),*)) }};
    (@debug $a:expr)               => {{ format!("{:?}", ($a)) }};
}

#[macro_export] macro_rules! vprintln {
    ($a:expr) => { for x in &($a) { println!("{}", x); } };
}
#[macro_export] macro_rules! vprintsp {
    ($a:expr) => {
        {
            use itertools::Itertools;
            println!("{}", ($a).iter().join(" "));
        }
    }
}
#[macro_export] macro_rules! print_grid {
    ($a:expr) => { for v in &($a) { println!("{}", v.iter().collect::<Str>()); } };
}

#[macro_export]#[cfg(feature="local")] macro_rules! debug {
    ($($a:expr),*)    => { eprintln!("{}", fmt!(@debug  $($a),*)); };
}
#[macro_export]#[cfg(not(feature="local"))] macro_rules! debug {
    ($($a:expr),*)    => { };
}
#[macro_export]#[cfg(feature="local")] macro_rules! debug2d {
    ($a:expr) => { for v in &($a) { eprintln!("{:?}", v); } };
}
#[macro_export]#[cfg(not(feature="local"))] macro_rules! debug2d {
    ($a:expr) => { };
}

pub fn yes(b: bool) -> &'static str { if b { "yes" } else { "no" } }
pub fn Yes(b: bool) -> &'static str { if b { "Yes" } else { "No" } }
pub fn YES(b: bool) -> &'static str { if b { "YES" } else { "NO" } }
pub fn no(b: bool) -> &'static str { yes(!b) }
pub fn No(b: bool) -> &'static str { Yes(!b) }
pub fn NO(b: bool) -> &'static str { YES(!b) }


}
