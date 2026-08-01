"""
test_guardian_score.py — Unit tests for Guardian Score Engine
=============================================================
Validates Bayesian update correctness, anti-manipulation safeguards,
and weighted consensus logic.
"""

import pytest
import sys
import os
import time
from unittest.mock import patch

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering.guardian_score import (
    UserProfile, UserRank, GuardianScoreEngine,
    INSTANT_BLACKLIST_THRESHOLD, REVIEW_QUEUE_THRESHOLD,
)


class TestUserProfile:
    """Tests for individual user reputation scoring."""

    def test_initial_guardian_score(self):
        """New user should start with moderate score from Beta(2,2) prior."""
        user = UserProfile(user_id="test", device_fingerprint="fp", registered_country="TW")
        # Beta(2,2) mean = 0.5, score = 0.5 * 0.60 + 0 - 0 = 0.30
        assert 0.25 <= user.guardian_score <= 0.35

    def test_initial_rank_is_civilian(self):
        """New user should be CIVILIAN rank."""
        user = UserProfile(user_id="test", device_fingerprint="fp", registered_country="TW")
        assert user.rank == UserRank.CIVILIAN

    def test_bayesian_update_correct(self):
        """Correct reports should increase alpha and score."""
        user = UserProfile(user_id="test", device_fingerprint="fp", registered_country="TW")
        initial_score = user.guardian_score

        for _ in range(20):
            user.update_bayesian(was_correct=True)

        assert user.guardian_score > initial_score
        assert user.alpha == 22.0  # 2.0 initial + 20 correct
        assert user.beta == 2.0    # Unchanged

    def test_bayesian_update_wrong(self):
        """Wrong reports should increase beta and lower score."""
        user = UserProfile(user_id="test", device_fingerprint="fp", registered_country="TW")

        for _ in range(10):
            user.update_bayesian(was_correct=False)

        assert user.beta == 12.0  # 2.0 initial + 10 wrong
        assert user.accuracy_rate < 0.5

    def test_rank_progression(self):
        """User should rank up as score increases."""
        user = UserProfile(user_id="test", device_fingerprint="fp", registered_country="TW")

        # Simulate 50 correct reports
        user.alpha = 47.0
        user.beta = 5.0
        user.total_reports = 50

        assert user.rank in (UserRank.GUARDIAN, UserRank.ARCHANGEL)
        assert user.report_weight >= 0.65

    def test_burst_penalty(self):
        """Burst reporting should lower score."""
        user = UserProfile(user_id="test", device_fingerprint="fp", registered_country="TW")
        user.alpha = 20.0
        user.total_reports = 20
        normal_score = user.guardian_score

        user.report_burst_count = 5
        assert user.guardian_score < normal_score

    def test_geo_inconsistency_penalty(self):
        """Geo inconsistency flag should lower score."""
        user = UserProfile(user_id="test", device_fingerprint="fp", registered_country="TW")
        user.alpha = 20.0
        user.total_reports = 20
        normal_score = user.guardian_score

        user.geo_inconsistency_flag = True
        assert user.guardian_score < normal_score


class TestGuardianScoreEngine:
    """Tests for the full engine lifecycle."""

    def setup_method(self):
        self.engine = GuardianScoreEngine()

    def test_register_user(self):
        profile = self.engine.register_user("u1", "fp1", "TW")
        assert profile.user_id == "u1"
        assert "u1" in self.engine.users

    def test_submit_report_unknown_user(self):
        result = self.engine.submit_report("nonexistent", "+886-000", "investment")
        assert result["status"] == "error"

    def test_submit_report_creates_risk_profile(self):
        self.engine.register_user("u1", "fp1", "TW")
        result = self.engine.submit_report("u1", "+886-SCAM", "investment")
        assert "+886-SCAM" in self.engine.phone_risk
        assert result["decision"] in ("SAFE", "SUSPECT", "REVIEW", "BLOCK")

    def test_consensus_required_for_block(self):
        """BLOCK requires minimum consensus (3 reports)."""
        # Register a high-reputation user
        self.engine.register_user("alice", "fp1", "TW")
        self.engine.users["alice"].alpha = 50.0
        self.engine.users["alice"].beta = 3.0
        self.engine.users["alice"].total_reports = 50

        # Single report shouldn't trigger BLOCK even from high-rep user
        result = self.engine.submit_report("alice", "+886-TEST", "investment")
        assert result["decision"] != "BLOCK"

    def test_multiple_reports_increase_score(self):
        """More reports should increase weighted scam score."""
        for i in range(5):
            uid = f"user_{i}"
            self.engine.register_user(uid, f"fp_{i}", "TW")
            self.engine.users[uid].alpha = 20.0
            self.engine.users[uid].total_reports = 20

        scores = []
        for i in range(5):
            result = self.engine.submit_report(f"user_{i}", "+886-MULTI", "investment")
            scores.append(result["weighted_scam_score"])

        # Score should be non-decreasing
        for j in range(1, len(scores)):
            assert scores[j] >= scores[j - 1]

    def test_geo_inconsistency_flagged(self):
        """Reports from inconsistent geography should flag user."""
        self.engine.register_user("dave", "fp1", "TW")
        self.engine.submit_report("dave", "+886-TEST", "investment", report_country="RU")
        assert self.engine.users["dave"].geo_inconsistency_flag is True

    def test_burst_throttling(self):
        """Excessive reporting should be throttled."""
        self.engine.register_user("spammer", "fp1", "TW")
        self.engine.users["spammer"].report_burst_count = 10  # At limit
        self.engine.users["spammer"].last_report_time = time.time()  # Keep in window

        result = self.engine.submit_report("spammer", "+886-SPAM", "investment")
        assert result["status"] == "throttled"

    def test_burst_throttling_window_reset(self):
        """測試爆量回報限制在時間視窗過期後，是否會正確重置。"""
        self.engine.register_user("active_user", "fp1", "TW")
        
        # 模擬快速回報 10 次，達到上限
        for _ in range(10):
            res = self.engine.submit_report("active_user", "+886-SPAM", "investment")
            assert res.get("status") != "throttled"
            
        # 第 11 次回報應該被阻擋
        res_blocked = self.engine.submit_report("active_user", "+886-SPAM", "investment")
        assert res_blocked.get("status") == "throttled"
        
        # 模擬時間推進了 301 秒 (超過 BURST_WINDOW_SEC 300秒)
        future_time = time.time() + 301
        
        # 這裡我們需要 mock time.time 讓 submit_report 認為時間已經過了
        with patch('time.time', return_value=future_time):
            res_recovered = self.engine.submit_report("active_user", "+886-SPAM", "investment")
            
            # 測試是否成功重置並允許回報
            assert res_recovered.get("status") != "throttled", "過了時間視窗後應該要解除封鎖"
            assert self.engine.users["active_user"].report_burst_count == 1, "計數器應該被重置為 1"

    def test_leaderboard(self):
        """Leaderboard should return sorted users."""
        self.engine.register_user("low", "fp1", "TW")
        self.engine.register_user("high", "fp2", "TW")
        self.engine.users["high"].alpha = 30.0
        self.engine.users["high"].total_reports = 30

        board = self.engine.get_leaderboard()
        assert board[0]["user_id"] == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
