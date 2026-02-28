#![allow(dead_code)]
use rand::Rng;
use rand_core::SeedableRng;
use crate::common::*;

/// 改善タイミングを時間方向で粗く可視化するための分割数。
/// `improved_bins[i]` は、探索時間の i 番目区間で改善が何回起きたかを表す。
const M: us = 10;
/// 温度の下限値。0 除算や極端なオーバーフロー/アンダーフローを避けるために使う。
const TEMP_FLOOR: f64 = 1e-12;
/// `exp(x)` がほぼ 0 にアンダーフローする境界の近似値。
const EXP_UNDERFLOW_THRESHOLD: f64 = -745.0;

/// 最適化の方向を型で表現する。
/// - `Maximize`: スコアを大きくする方向が改善
/// - `Minimize`: スコアを小さくする方向が改善
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(unused)]
pub enum Objective {
    /// 大きいスコアほど良い（最大化）
    Maximize,
    /// 小さいスコアほど良い（最小化）
    Minimize,
}

/// 焼きなましの統計情報。
/// 探索中に更新され、`stats()` で参照として取得できる。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaStats {
    /// `update_delta` が呼ばれた回数。
    pub iter: us,
    /// 統計更新を行うかどうか。本番高速化したい場合に false を指定する。
    pub track_stats: bool,
    /// 改善/悪化問わず、遷移を受理した回数。
    pub accepted: us,
    /// 受理のうち「改善だった」回数。
    pub improved: us,
    /// 時間帯ごとの改善回数ヒストグラム。
    pub improved_bins: [us; M],
}

impl SaStats {
    fn new(track_stats: bool) -> Self {
        Self {
            iter: 0,
            track_stats,
            accepted: 0,
            improved: 0,
            improved_bins: [0; M],
        }
    }

    #[inline(always)]
    fn set_track_stats(&mut self, track_stats: bool) {
        self.track_stats = track_stats;
    }

    #[cfg(feature = "local")]
    #[inline(always)]
    fn on_transition_accepted(&mut self, improved: bool, elapsed: f64, limit: f64) {
        if !self.track_stats {
            return;
        }
        self.accepted += 1;
        if improved {
            self.improved += 1;
            let progress = if !elapsed.is_finite() {
                1.0
            } else {
                (elapsed / limit).clamp(0.0, 1.0)
            };
            let idx = ((progress * M as f64) as us).min(M - 1);
            self.improved_bins[idx] += 1;
        }
    }

    #[cfg(not(feature = "local"))]
    #[inline(always)]
    fn on_transition_accepted(&mut self, _improved: bool, _elapsed: f64, _limit: f64) {}
}

/// 焼きなまし本体。
/// 1 ステップごとに差分スコア `delta` を与えて受理/棄却を判定する。
pub struct SimulatedAnnealing {
    /// 初期温度。探索初期のランダム性を決める。
    /// 一回の遷移で起きる「大きめの悪化」もある程度受け入れる値を目安にする。
    t0: f64,
    /// 最終温度。探索終盤のランダム性を決める。
    /// 一回の遷移で起きる「小さめの悪化」をたまに受け入れる程度を目安にする。
    t1: f64,
    /// 許容時間（呼び出し側が与える `elapsed` と同じ単位で扱う）。
    limit: f64,
    /// 現在のスコア。
    score: i64,
    /// 最大化/最小化の方向。
    objective: Objective,
    /// 受理判定に使う乱数生成器。
    rng: rand_pcg::Pcg64Mcg,
    /// 最小化問題かどうか。ホットパスでの `match` を避けるために使う。
    is_minimize: bool,
    /// 現在温度（一定ステップごとに再計算）。
    temp: f64,
    /// `1 / temp` のキャッシュ。`update_delta` での除算を避ける。
    inv_temp: f64,
    /// 統計情報をひとまとめで保持する。
    stats: SaStats,
}

impl SimulatedAnnealing {
    /// OS 由来シードで初期化する。
    /// 再現性が必要な場合は `with_seed` を使う。
    pub fn new(t0: f64, t1: f64, limit: f64, objective: Objective, initial_score: i64) -> Self {
        Self::validate_params(t0, t1, limit);
        Self::build(t0, t1, limit, objective, initial_score, rand_pcg::Pcg64Mcg::from_os_rng())
    }

    /// 指定シードで初期化する（再現実験向け）。
    pub fn with_seed(t0: f64, t1: f64, limit: f64, objective: Objective, initial_score: i64, seed: u64) -> Self {
        Self::validate_params(t0, t1, limit);
        Self::build(t0, t1, limit, objective, initial_score, rand_pcg::Pcg64Mcg::seed_from_u64(seed))
    }

    /// 統計更新の有効/無効を初期化時に指定するための builder 風 API。
    /// `false` を指定すると `accepted/improved/improved_bins` の更新コストを省ける。
    pub fn with_stats_tracking(mut self, track_stats: bool) -> Self {
        self.stats.set_track_stats(track_stats);
        self
    }

    /// 途中で統計更新の有効/無効を切り替える。
    pub fn set_stats_tracking(&mut self, track_stats: bool) {
        self.stats.set_track_stats(track_stats);
    }

    pub fn iter(&self) -> us { self.stats.iter }

    fn build(
        t0: f64,
        t1: f64,
        limit: f64,
        objective: Objective,
        initial_score: i64,
        rng: rand_pcg::Pcg64Mcg,
    ) -> Self {
        let is_minimize = matches!(objective, Objective::Minimize);
        let temp = t0.max(TEMP_FLOOR);
        Self {
            t0,
            t1,
            limit,
            score: initial_score,
            objective,
            rng,
            is_minimize,
            // 初回 `update_delta` までに使われる可能性があるので、温度は安全な下限つきで初期化。
            temp,
            inv_temp: 1.0 / temp,
            stats: SaStats::new(true),
        }
    }

    /// パラメータ検証。
    /// 温度と制限時間は「正の有限値」である必要がある。
    fn validate_params(t0: f64, t1: f64, limit: f64) {
        assert!(t0.is_finite() && t0 > 0.0, "t0 must be finite and > 0, got {}", t0);
        assert!(t1.is_finite() && t1 > 0.0, "t1 must be finite and > 0, got {}", t1);
        assert!(limit.is_finite() && limit > 0.0, "limit must be finite and > 0, got {}", limit);
    }

    /// 経過時間を 0.0..=1.0 の進捗率に正規化する。
    /// 不正値（NaN/Inf）は「探索終盤」とみなして 1.0 を返す。
    fn progress(&self, elapsed: f64) -> f64 {
        if !elapsed.is_finite() {
            return 1.0;
        }
        (elapsed / self.limit).clamp(0.0, 1.0)
    }

    /// 幾何補間による温度スケジュール。
    /// `t=0` で `t0`, `t=1` で `t1` になり、下限 `TEMP_FLOOR` を保証する。
    fn temp(&self, elapsed: f64) -> f64 {
        let t = self.progress(elapsed);
        (self.t0.powf(1.0 - t) * self.t1.powf(t)).max(TEMP_FLOOR)
    }

    /// 温度と逆温度キャッシュを同時に更新する。
    #[inline(always)]
    fn refresh_temp(&mut self, elapsed: f64) {
        self.temp = self.temp(elapsed);
        self.inv_temp = 1.0 / self.temp;
    }

    /// 最適化方向を吸収した差分。
    /// ここで「改善なら正、悪化なら負」に揃えることで、
    /// 以後の判定ロジックを最大化/最小化で共通化できる。
    /// `i64::MIN` を安全に扱うため、最小化時の符号反転は `saturating_neg` を使う。
    #[inline(always)]
    fn normalized_delta(&self, delta: i64) -> i64 {
        if self.is_minimize {
            delta.saturating_neg()
        } else {
            delta
        }
    }

    /// 改善でない遷移（`normalized_delta <= 0`）の受理確率を返す。
    /// 返り値は常に `0.0..=1.0` に収まる。
    #[inline(always)]
    fn acceptance_probability_non_improving(&self, normalized_delta: i64) -> f64 {
        if normalized_delta >= 0 {
            return 1.0;
        }
        let x = (normalized_delta as f64) * self.inv_temp;
        if x < EXP_UNDERFLOW_THRESHOLD {
            return 0.0;
        }
        f64::exp(x)
    }

    /// 改善でない遷移（`normalized_delta <= 0`）を受理するか判定する。
    /// - 0 差分は `exp(0)=1` なので即受理
    /// - `x` が十分小さいときは `exp(x)` が 0 扱いになるため即棄却
    /// - それ以外は `U(0,1) < p` で判定する
    #[inline(always)]
    fn accept_non_improving(&mut self, normalized_delta: i64) -> bool {
        if normalized_delta == 0 {
            return true;
        }
        let p = self.acceptance_probability_non_improving(normalized_delta);
        if p <= 0.0 {
            return false;
        }
        self.rng.random::<f64>() < p
    }

    /// 「次状態の絶対スコア」を与える API。
    /// 内部では差分に変換して `update_delta` に委譲する。
    pub fn update(&mut self, next_score: i64, elapsed: f64) -> bool {
        self.update_delta(next_score - self.score, elapsed)
    }

    /// 「差分スコア」を与える本体 API。
    /// 戻り値が `true` のとき、その遷移は受理されて `self.score` が更新済み。
    #[inline(always)]
    pub fn update_delta(&mut self, delta: i64, elapsed: f64) -> bool {
        // 温度再計算は毎回だと重いので、一定間隔で更新する。
        if self.stats.iter & 0x7F == 0 {
            self.refresh_temp(elapsed);
        }
        self.stats.iter += 1;

        // 方向を吸収して「正なら改善」に揃える。
        let normalized_delta = self.normalized_delta(delta);
        let improved = normalized_delta > 0;

        // 改善は常に受理。悪化は確率受理。
        let accepted = if improved {
            true
        } else {
            self.accept_non_improving(normalized_delta)
        };

        if accepted {
            // 受理されたのでスコアを確定反映する。
            self.score += delta;
            self.stats.on_transition_accepted(improved, elapsed, self.limit);
            true
        } else {
            false
        }
    }

    /// 現在の統計情報への参照を返す。
    /// 呼び出し側で clone せずに読み取り可能。
    pub fn stats(&self) -> &SaStats {
        &self.stats
    }
}

// CAP(IGNORE_BELOW)
#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn new_test_sa(objective: Objective, seed: u64) -> SimulatedAnnealing {
        SimulatedAnnealing::with_seed(30000.0, 10.0, 1900.0, objective, 0, seed)
    }

    #[test]
    fn maximize_accepts_improvement() {
        let mut an = new_test_sa(Objective::Maximize, 1);
        assert!(an.update_delta(1, 0.0));
        assert_eq!(an.score, 1);
        let stats = an.stats();
        assert_eq!(stats.accepted, 1);
        assert_eq!(stats.improved, 1);
    }

    #[test]
    fn minimize_worse_move_no_panic() {
        let mut an = new_test_sa(Objective::Minimize, 2);
        let result = catch_unwind(AssertUnwindSafe(|| an.update_delta(1, 0.0)));
        assert!(result.is_ok());
    }

    #[test]
    fn acceptance_probability_bounded() {
        let an = new_test_sa(Objective::Maximize, 3);
        let deltas = [-1_000_000_000_i64, -100, -1, 0, 1, 100, 1_000_000_000];
        for &delta in &deltas {
            let p = an.acceptance_probability_non_improving(an.normalized_delta(delta));
            assert!(p.is_finite(), "p must be finite, got {}", p);
            assert!((0.0..=1.0).contains(&p), "p must be in [0, 1], got {}", p);
        }
    }

    #[test]
    fn elapsed_at_limit_no_oob() {
        let mut an = new_test_sa(Objective::Maximize, 4);
        assert!(an.update_delta(1, an.limit));
        let stats = an.stats();
        assert_eq!(stats.improved_bins.len(), M);
        assert_eq!(stats.improved_bins.iter().sum::<us>(), 1);
        assert_eq!(stats.improved_bins[M - 1], 1);
    }

    #[test]
    fn invalid_params_rejected() {
        let cases = [
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (f64::NAN, 1.0, 1.0),
            (1.0, f64::INFINITY, 1.0),
            (1.0, 1.0, f64::NEG_INFINITY),
        ];
        for &(t0, t1, limit) in &cases {
            let result = catch_unwind(|| {
                SimulatedAnnealing::with_seed(t0, t1, limit, Objective::Maximize, 0, 0)
            });
            assert!(result.is_err(), "params ({}, {}, {}) should panic", t0, t1, limit);
        }
    }

    #[test]
    fn seed_reproducibility() {
        let mut a = new_test_sa(Objective::Maximize, 42);
        let mut b = new_test_sa(Objective::Maximize, 42);
        let deltas = [-12, -3, -7, 5, -8, 2, -1, -10, 3, -4, 6, -9, 1, -6, 4];

        let mut a_accepted = Vec::new();
        let mut b_accepted = Vec::new();
        for (i, &delta) in deltas.iter().enumerate() {
            let elapsed = (i as f64) * 37.0;
            a_accepted.push(a.update_delta(delta, elapsed));
            b_accepted.push(b.update_delta(delta, elapsed));
        }

        assert_eq!(a_accepted, b_accepted);
        assert_eq!(a.score, b.score);
        assert_eq!(a.stats(), b.stats());
    }

    #[test]
    fn stats_tracking_can_be_disabled() {
        let mut an = new_test_sa(Objective::Maximize, 7).with_stats_tracking(false);
        assert!(an.update_delta(5, 0.0));
        assert_eq!(an.stats().iter, 1);
        assert_eq!(an.stats().accepted, 0);
        assert_eq!(an.stats().improved, 0);
        assert_eq!(an.stats().improved_bins.iter().sum::<us>(), 0);
    }
}
