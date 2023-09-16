#![allow(dead_code)]
use rand::{rngs::SmallRng, Rng};
use rand_core::SeedableRng;
use crate::common::*;

pub struct SimulatedAnnealing {
    pub start_temp: i64, // 初期温度. 一度の遷移で改善されるscoreの最大値くらいを設定する
    pub end_temp: i64,   // 最終温度. 一度の遷移で改善されるscoreの最小値くらいを設定する
    pub limit_ms: us,
    pub score: i64,
    pub rng: SmallRng,
}

impl SimulatedAnnealing {
    pub fn new(start_temp: i64, end_temp: i64, limit_ms: us) -> Self {
        Self {
            start_temp,
            end_temp,
            limit_ms,
            score: i64::MINF,
            rng: SmallRng::from_entropy(),
        }
    }

    pub fn update(&mut self, next_score: i64, elapsed: us) -> bool {
        let delta = next_score - self.score;
        // スコアが改善されていれば更新する
        // 改善されていない場合でも時間と差分に応じた確率で遷移させる (焼きなまし)
        // scoreを最大化するように実装してるので、最小化したい場合は負にする必要あり.
        if delta > 0 || self.prob(delta, elapsed) > self.rng.gen_range(0f64..1f64) {
            self.score = next_score;
            true
        } else {
            false
        }
    }

    fn prob(&self, delta: i64, elapsed: us) -> f64 {
        let (st, et, lim) = (self.start_temp.f64(), self.end_temp.f64(), self.limit_ms.f64());
        let temp = st + (et - st) * elapsed.f64() / lim;
        f64::exp(delta.f64() / temp)
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
        let mut an = SimulatedAnnealing::new(30000, 10, 1900);
        let mut _cnt = 0;
        loop {
            let elapsed = start.elapsed().as_millis() as us;
            if elapsed >= an.limit_ms { break; }

            let next_score = eval();
            if an.update(next_score, elapsed) {
                // 更新結果を反映
            } else {
                // もとにもどす
            }
            _cnt += 1;
        }

        println!("{}", an.score);
    }
}