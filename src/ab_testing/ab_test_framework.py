"""
A/B Testing Framework for Recommendation Engine
Provides statistical testing and significance analysis for model comparison
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B Test status enumeration"""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    STOPPED = "stopped"


class MetricType(Enum):
    """Metric types for A/B testing"""
    CTR = "ctr"  # Click-Through Rate
    CONVERSION_RATE = "conversion_rate"
    REVENUE_PER_USER = "revenue_per_user"
    AVERAGE_ORDER_VALUE = "average_order_value"
    SESSION_DURATION = "session_duration"
    BOUNCE_RATE = "bounce_rate"


@dataclass
class TestConfig:
    """Configuration for A/B test"""
    name: str
    description: str
    traffic_split: float  # Percentage of traffic to test group (0.0-1.0)
    minimum_sample_size: int
    confidence_level: float = 0.95
    power: float = 0.8
    expected_effect_size: float = 0.1
    test_duration_days: int = 14
    metrics: List[MetricType] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [MetricType.CTR]
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class TestResult:
    """Results of A/B test"""
    test_id: str
    control_group: Dict[str, Any]
    test_group: Dict[str, Any]
    statistical_significance: bool
    confidence_interval: Tuple[float, float]
    effect_size: float
    p_value: float
    test_statistic: float
    sample_size_control: int
    sample_size_test: int
    metrics_results: Dict[MetricType, Dict[str, Any]]
    recommendation: str
    completed_at: datetime


class ABTestFramework:
    """
    Comprehensive A/B Testing Framework for recommendation models
    """
    
    def __init__(self):
        self.active_tests = {}
        self.completed_tests = {}
        self.test_results = {}
        
    def create_test(self, config: TestConfig) -> str:
        """
        Create a new A/B test
        Args:
            config: Test configuration
        Returns:
            Test ID
        """
        test_id = str(uuid.uuid4())
        
        test_data = {
            "id": test_id,
            "config": config,
            "status": TestStatus.DRAFT,
            "created_at": datetime.now(),
            "started_at": None,
            "ended_at": None,
            "control_group_data": [],
            "test_group_data": [],
            "metrics_collected": {metric: [] for metric in config.metrics}
        }
        
        self.active_tests[test_id] = test_data
        
        logger.info(f"Created A/B test '{config.name}' with ID: {test_id}")
        return test_id
    
    def start_test(self, test_id: str) -> bool:
        """
        Start an A/B test
        """
        if test_id not in self.active_tests:
            logger.error(f"Test {test_id} not found")
            return False
        
        test_data = self.active_tests[test_id]
        test_data["status"] = TestStatus.RUNNING
        test_data["started_at"] = datetime.now()
        
        logger.info(f"Started A/B test {test_id}")
        return True
    
    def assign_user_to_group(self, test_id: str, user_id: str) -> str:
        """
        Assign user to control or test group
        Args:
            test_id: Test identifier
            user_id: User identifier
        Returns:
            Group assignment ('control' or 'test')
        """
        if test_id not in self.active_tests:
            return "control"
        
        test_data = self.active_tests[test_id]
        traffic_split = test_data["config"].traffic_split
        
        # Use consistent hashing for user assignment
        hash_value = int(hash(user_id)) % 100
        
        if hash_value < traffic_split * 100:
            return "test"
        else:
            return "control"
    
    def record_interaction(self, 
                          test_id: str, 
                          user_id: str, 
                          interaction_data: Dict[str, Any]) -> bool:
        """
        Record user interaction data
        Args:
            test_id: Test identifier
            user_id: User identifier
            interaction_data: Dictionary with interaction metrics
        """
        if test_id not in self.active_tests:
            return False
        
        test_data = self.active_tests[test_id]
        group = self.assign_user_to_group(test_id, user_id)
        
        # Add group assignment and timestamp to interaction data
        interaction_data["group"] = group
        interaction_data["user_id"] = user_id
        interaction_data["timestamp"] = datetime.now()
        
        # Store interaction data
        if group == "control":
            test_data["control_group_data"].append(interaction_data)
        else:
            test_data["test_group_data"].append(interaction_data)
        
        # Collect specific metrics
        for metric in test_data["config"].metrics:
            if metric.value in interaction_data:
                test_data["metrics_collected"][metric].append({
                    "group": group,
                    "value": interaction_data[metric.value],
                    "timestamp": interaction_data["timestamp"]
                })
        
        return True
    
    def calculate_sample_size(self, 
                            metric_type: MetricType,
                            baseline_rate: float,
                            expected_improvement: float,
                            confidence_level: float = 0.95,
                            power: float = 0.8) -> int:
        """
        Calculate required sample size for A/B test
        """
        # Effect size (Cohen's d for continuous, proportion difference for binary)
        if metric_type in [MetricType.CTR, MetricType.CONVERSION_RATE, MetricType.BOUNCE_RATE]:
            # For proportions
            effect_size = expected_improvement
            alpha = 1 - confidence_level
            
            # Sample size calculation for two proportions
            p1 = baseline_rate
            p2 = baseline_rate * (1 + expected_improvement)
            
            # Pooled proportion
            p_pooled = (p1 + p2) / 2
            
            # Z-scores
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            # Sample size per group
            n_per_group = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / (p2 - p1)**2
            
            return int(np.ceil(n_per_group))
        
        else:
            # For continuous metrics (revenue, session duration, etc.)
            effect_size = expected_improvement  # Standardized effect size
            alpha = 1 - confidence_level
            
            # Use power analysis
            power_analysis = TTestIndPower()
            sample_size = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative='two-sided'
            )
            
            return int(np.ceil(sample_size))
    
    def analyze_test(self, test_id: str) -> Optional[TestResult]:
        """
        Analyze completed A/B test results
        """
        if test_id not in self.active_tests:
            logger.error(f"Test {test_id} not found")
            return None
        
        test_data = self.active_tests[test_id]
        
        if len(test_data["control_group_data"]) == 0 or len(test_data["test_group_data"]) == 0:
            logger.warning(f"Insufficient data for test {test_id}")
            return None
        
        metrics_results = {}
        overall_significance = False
        
        for metric in test_data["config"].metrics:
            metric_data = test_data["metrics_collected"][metric]
            
            if not metric_data:
                continue
            
            # Separate control and test group data
            control_values = [d["value"] for d in metric_data if d["group"] == "control"]
            test_values = [d["value"] for d in metric_data if d["group"] == "test"]
            
            if len(control_values) == 0 or len(test_values) == 0:
                continue
            
            # Perform statistical test
            if metric in [MetricType.CTR, MetricType.CONVERSION_RATE, MetricType.BOUNCE_RATE]:
                # Proportion test
                result = self._analyze_proportions(control_values, test_values, test_data["config"].confidence_level)
            else:
                # T-test for continuous metrics
                result = self._analyze_continuous(control_values, test_values, test_data["config"].confidence_level)
            
            metrics_results[metric] = result
            
            if result["significant"]:
                overall_significance = True
        
        # Calculate overall effect size and recommendation
        control_mean = np.mean([d["value"] for d in test_data["control_group_data"] if "value" in d])
        test_mean = np.mean([d["value"] for d in test_data["test_group_data"] if "value" in d])
        
        effect_size = (test_mean - control_mean) / control_mean if control_mean != 0 else 0
        
        # Generate recommendation
        if overall_significance and effect_size > 0:
            recommendation = "Implement test group - shows significant improvement"
        elif overall_significance and effect_size < 0:
            recommendation = "Keep control group - test group performs worse"
        else:
            recommendation = "No significant difference detected - continue testing or keep control"
        
        # Create test result
        test_result = TestResult(
            test_id=test_id,
            control_group={
                "sample_size": len(test_data["control_group_data"]),
                "mean": control_mean,
                "std": np.std([d["value"] for d in test_data["control_group_data"] if "value" in d])
            },
            test_group={
                "sample_size": len(test_data["test_group_data"]),
                "mean": test_mean,
                "std": np.std([d["value"] for d in test_data["test_group_data"] if "value" in d])
            },
            statistical_significance=overall_significance,
            confidence_interval=(0, 0),  # Will be calculated per metric
            effect_size=effect_size,
            p_value=0,  # Will be calculated per metric
            test_statistic=0,  # Will be calculated per metric
            sample_size_control=len(test_data["control_group_data"]),
            sample_size_test=len(test_data["test_group_data"]),
            metrics_results=metrics_results,
            recommendation=recommendation,
            completed_at=datetime.now()
        )
        
        self.test_results[test_id] = test_result
        
        logger.info(f"Analysis completed for test {test_id}")
        return test_result
    
    def _analyze_proportions(self, 
                           control_values: List[float], 
                           test_values: List[float],
                           confidence_level: float) -> Dict[str, Any]:
        """
        Analyze proportion metrics (CTR, conversion rate, etc.)
        """
        control_successes = sum(control_values)
        control_total = len(control_values)
        test_successes = sum(test_values)
        test_total = len(test_values)
        
        # Two-proportion z-test
        count = np.array([control_successes, test_successes])
        nobs = np.array([control_total, test_total])
        
        z_stat, p_value = proportions_ztest(count, nobs)
        
        # Calculate confidence interval for difference in proportions
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        p1 = control_successes / control_total
        p2 = test_successes / test_total
        
        se_diff = np.sqrt((p1 * (1 - p1) / control_total) + (p2 * (1 - p2) / test_total))
        diff = p2 - p1
        
        ci_lower = diff - z_critical * se_diff
        ci_upper = diff + z_critical * se_diff
        
        return {
            "significant": p_value < (1 - confidence_level),
            "p_value": p_value,
            "test_statistic": z_stat,
            "control_rate": p1,
            "test_rate": p2,
            "difference": diff,
            "confidence_interval": (ci_lower, ci_upper),
            "relative_improvement": (diff / p1) if p1 > 0 else 0
        }
    
    def _analyze_continuous(self, 
                          control_values: List[float], 
                          test_values: List[float],
                          confidence_level: float) -> Dict[str, Any]:
        """
        Analyze continuous metrics (revenue, session duration, etc.)
        """
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(test_values, control_values)
        
        # Calculate confidence interval for difference in means
        alpha = 1 - confidence_level
        df = len(control_values) + len(test_values) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        mean_control = np.mean(control_values)
        mean_test = np.mean(test_values)
        std_control = np.std(control_values, ddof=1)
        std_test = np.std(test_values, ddof=1)
        
        # Pooled standard error
        n1, n2 = len(control_values), len(test_values)
        se_diff = np.sqrt((std_control**2 / n1) + (std_test**2 / n2))
        
        diff = mean_test - mean_control
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff
        
        return {
            "significant": p_value < (1 - confidence_level),
            "p_value": p_value,
            "test_statistic": t_stat,
            "control_mean": mean_control,
            "test_mean": mean_test,
            "difference": diff,
            "confidence_interval": (ci_lower, ci_upper),
            "relative_improvement": (diff / mean_control) if mean_control != 0 else 0
        }
    
    def stop_test(self, test_id: str) -> bool:
        """
        Stop an A/B test
        """
        if test_id not in self.active_tests:
            return False
        
        test_data = self.active_tests[test_id]
        test_data["status"] = TestStatus.STOPPED
        test_data["ended_at"] = datetime.now()
        
        # Move to completed tests
        self.completed_tests[test_id] = test_data
        del self.active_tests[test_id]
        
        logger.info(f"Stopped A/B test {test_id}")
        return True
    
    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of A/B test
        """
        if test_id in self.active_tests:
            test_data = self.active_tests[test_id]
        elif test_id in self.completed_tests:
            test_data = self.completed_tests[test_id]
        else:
            return None
        
        return {
            "test_id": test_id,
            "name": test_data["config"].name,
            "status": test_data["status"].value,
            "created_at": test_data["created_at"],
            "started_at": test_data.get("started_at"),
            "ended_at": test_data.get("ended_at"),
            "control_group_size": len(test_data["control_group_data"]),
            "test_group_size": len(test_data["test_group_data"]),
            "traffic_split": test_data["config"].traffic_split,
            "minimum_sample_size": test_data["config"].minimum_sample_size
        }
    
    def get_test_summary(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of test results
        """
        if test_id in self.test_results:
            result = self.test_results[test_id]
            return asdict(result)
        
        return None
    
    def list_active_tests(self) -> List[Dict[str, Any]]:
        """
        List all active tests
        """
        return [self.get_test_status(test_id) for test_id in self.active_tests.keys()]
    
    def list_completed_tests(self) -> List[Dict[str, Any]]:
        """
        List all completed tests
        """
        return [self.get_test_status(test_id) for test_id in self.completed_tests.keys()]
    
    def export_test_results(self, test_id: str, format: str = "json") -> Optional[str]:
        """
        Export test results to specified format
        """
        if test_id not in self.test_results:
            return None
        
        result = self.test_results[test_id]
        
        if format == "json":
            return json.dumps(asdict(result), default=str, indent=2)
        elif format == "csv":
            # Convert to CSV format
            df = pd.DataFrame([asdict(result)])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def calculate_power_analysis(self, 
                                baseline_rate: float,
                                expected_improvement: float,
                                sample_size: int,
                                alpha: float = 0.05) -> Dict[str, float]:
        """
        Calculate statistical power for given parameters
        """
        # Effect size
        effect_size = expected_improvement
        
        # Calculate power
        power_analysis = TTestIndPower()
        power = power_analysis.power(
            effect_size=effect_size,
            nobs1=sample_size,
            alpha=alpha,
            alternative='two-sided'
        )
        
        return {
            "power": power,
            "effect_size": effect_size,
            "sample_size": sample_size,
            "alpha": alpha,
            "adequate": power >= 0.8
        }


class ABTestDashboard:
    """
    Dashboard utilities for A/B test visualization and monitoring
    """
    
    def __init__(self, ab_framework: ABTestFramework):
        self.framework = ab_framework
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for A/B test dashboard
        """
        active_tests = self.framework.list_active_tests()
        completed_tests = self.framework.list_completed_tests()
        
        # Calculate summary statistics
        total_tests = len(active_tests) + len(completed_tests)
        running_tests = len([t for t in active_tests if t["status"] == "running"])
        
        # Get recent results
        recent_results = []
        for test_id in list(self.framework.test_results.keys())[-5:]:
            result = self.framework.test_results[test_id]
            recent_results.append({
                "test_id": test_id,
                "name": result.test_id,
                "effect_size": result.effect_size,
                "significant": result.statistical_significance,
                "recommendation": result.recommendation,
                "completed_at": result.completed_at
            })
        
        return {
            "summary": {
                "total_tests": total_tests,
                "active_tests": len(active_tests),
                "running_tests": running_tests,
                "completed_tests": len(completed_tests)
            },
            "active_tests": active_tests,
            "completed_tests": completed_tests,
            "recent_results": recent_results
        }
    
    def generate_test_report(self, test_id: str) -> Dict[str, Any]:
        """
        Generate detailed report for a specific test
        """
        test_status = self.framework.get_test_status(test_id)
        test_result = self.framework.get_test_summary(test_id)
        
        if not test_status:
            return {"error": "Test not found"}
        
        report = {
            "test_info": test_status,
            "results": test_result,
            "recommendations": []
        }
        
        if test_result:
            # Add specific recommendations based on results
            if test_result["statistical_significance"]:
                if test_result["effect_size"] > 0:
                    report["recommendations"].append("Implement the test variant - shows significant improvement")
                else:
                    report["recommendations"].append("Keep control variant - test variant performs worse")
            else:
                report["recommendations"].append("No significant difference detected")
                
                # Check if sample size is adequate
                if test_result["sample_size_control"] + test_result["sample_size_test"] < test_status["minimum_sample_size"]:
                    report["recommendations"].append("Consider increasing sample size")
                else:
                    report["recommendations"].append("Test has adequate sample size - effects may be too small to detect")
        
        return report
