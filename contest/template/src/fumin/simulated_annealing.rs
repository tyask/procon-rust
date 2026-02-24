#![allow(dead_code)]
use std::cmp::Ordering;

use rand::Rng;
use rand_core::SeedableRng;
use crate::common::*;

const M: us = 10;

pub struct SimulatedAnnealing {
    pub t0: f64, // 初期温度. 一度の遷移で改善されるscoreの最大値くらいを設定する
    pub t1: f64, // 最終温度. 一度の遷移で改善されるscoreの最小値くらいを設定する
    pub limit: f64,
    pub score: i64,
    pub ordering: Ordering,
    pub rng: rand_pcg::Pcg64Mcg,
    pub temp: f64,
    pub iter: us,
    pub updated: [us; M],
}

impl SimulatedAnnealing {
    pub fn new(t0: f64, t1: f64, limit: f64) -> Self {
        Self {
            t0,
            t1,
            limit,
            score: i64::MINF,
            ordering: Ordering::Greater,
            rng: rand_pcg::Pcg64Mcg::from_os_rng(),
            temp: 0.,
            iter: 0,
            updated: [0; M],
        }
    }

    pub fn initial_score(mut self, s: i64) -> Self {
        self.score = s;
        self
    }

    pub fn ordering(mut self, ordering: Ordering) -> Self {
        self.ordering = ordering;
        self
    }

    pub fn update(&mut self, next_score: i64, elapsed: f64) -> bool {
        self.update_delta(next_score - self.score, elapsed)
    }

    pub fn update_delta(&mut self, delta: i64, elapsed: f64) -> bool {
        if self.iter % 100 == 0 { self.temp = self.temp(elapsed); }
        self.iter += 1;
        // スコアが改善されていれば更新する
        // 改善されていない場合でも時間と差分に応じた確率で遷移させる (焼きなまし)
        // scoreを最大化するように実装してるので、最小化したい場合は負にする必要あり.
        let improved = delta.cmp(&0) == self.ordering;
        if improved || self.rng.random_bool(f64::exp(delta.f64() / self.temp)) {
            self.score += delta;
            if improved { self.updated[(elapsed/self.limit*M.f64()).us()] += 1; }
            true
        } else {
            false
        }
    }

    fn temp(&self, elapsed: f64) -> f64 {
        // 時間の経過に応じてt0 -> t1まで徐々に冷やされていく
        let (t0, t1, t) = (self.t0, self.t1, elapsed / self.limit);
        t0.powf(1. - t) * t1.powf(t)
    }

    pub fn stats(&self) -> String {
        format!("iter={} updated={:?}", self.iter, self.updated)
    }
}

// CAP(IGNORE_BELOW)
#[cfg(test)]
mod tests {
    use std::time::Instant;
    use super::*;

    #[test]
    fn template() {
        let eval = || -> i64 { 10 }; // 評価関数
        let start = Instant::now();
        let mut an = SimulatedAnnealing::new(30000., 10., 1900.);
        loop {
            let elapsed = start.elapsed().as_millis() as f64;
            if elapsed >= an.limit { break; }

            let next_score = eval();
            if an.update(next_score, elapsed) {
                // 更新結果を反映
            } else {
                // もとにもどす
            }
        }

        println!("{}", an.score);
    }
}