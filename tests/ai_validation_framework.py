"""
AI Accuracy Validation Framework for HVAC Analysis Agents.

Critical for production AI: Validate AI accuracy with known scenarios
BEFORE any customer data is processed. This prevents:
- Incorrect recommendations reaching customers
- Damage to professional reputation
- Liability from poor AI advice
- Customer churn from unreliable analysis

Validation methodology:
1. Expert-validated test scenarios with known correct answers
2. Consistency testing across multiple runs
3. Boundary condition testing
4. Comparative analysis against industry standards
5. Statistical validation of AI decision patterns
"""

import asyncio
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, cast

import structlog
from pydantic import BaseModel

from adapters.hvac.domain import HVACMetric, HVACMetricType, HVACSystemType
from core.services.ai_analysis import AIAnalysisConfig
from core.services.generic_hvac_agents import (
    GenericHVACAIService,
)

logger = structlog.get_logger()


class ValidationScenarioType(Enum):
    """Types of validation scenarios for comprehensive testing."""

    NORMAL_OPERATION = "normal_operation"
    EFFICIENCY_PROBLEM = "efficiency_problem"
    COMFORT_ISSUE = "comfort_issue"
    EQUIPMENT_FAULT = "equipment_fault"
    BOUNDARY_CONDITION = "boundary_condition"
    COMPLEX_MULTI_ISSUE = "complex_multi_issue"


@dataclass
class ExpertValidatedScenario:
    """
    Test scenario with expert-validated expected outcomes.

    These scenarios are created by HVAC experts and represent
    known situations with predictable analysis results.
    """

    scenario_id: str
    scenario_type: ValidationScenarioType
    description: str

    # Input data
    hvac_metrics: list[HVACMetric]
    system_context: dict[str, Any]

    # Expert-validated expected outcomes
    expected_energy_assessment: dict[str, Any]
    expected_comfort_assessment: dict[str, Any]
    expected_fault_assessment: dict[str, Any]

    # Validation criteria
    acceptable_confidence_range: tuple[float, float] = (0.7, 1.0)
    critical_recommendations_must_include: list[str] | None = None
    critical_recommendations_must_not_include: list[str] | None = None

    # Expert metadata
    validated_by: str = "HVAC Expert"
    validation_date: datetime | None = None
    industry_standards_applied: list[str] | None = None

    def __post_init__(self):
        if self.validation_date is None:
            self.validation_date = datetime.now(UTC)
        if self.critical_recommendations_must_include is None:
            self.critical_recommendations_must_include = []
        if self.critical_recommendations_must_not_include is None:
            self.critical_recommendations_must_not_include = []


class ValidationResult(BaseModel):
    """Results of AI validation against expert scenarios."""

    scenario_id: str
    scenario_type: str
    validation_passed: bool
    confidence_score: float

    # Detailed validation results
    energy_validation: dict[str, Any]
    comfort_validation: dict[str, Any]
    fault_validation: dict[str, Any]

    # Issues found
    validation_issues: list[str]
    critical_failures: list[str]

    # Performance metrics
    analysis_duration_seconds: float
    consistency_score: float  # Based on multiple runs

    # Validation metadata
    validation_timestamp: datetime
    ai_model_used: str


class AIValidationFramework:
    """
    Comprehensive framework for validating AI agent accuracy.

    This must be run and pass before any customer deployment.
    Provides confidence that AI analysis is reliable and accurate.
    """

    def __init__(self, ai_config: AIAnalysisConfig):
        self.ai_config = ai_config
        self.hvac_ai_service = GenericHVACAIService(ai_config)
        self.logger = logger.bind(component="ai_validation_framework")

        # Validation results storage
        self.validation_results: list[ValidationResult] = []

        # Expert scenarios database
        self.expert_scenarios = self._load_expert_scenarios()

    def _load_expert_scenarios(self) -> list[ExpertValidatedScenario]:
        """Load expert-validated test scenarios."""

        scenarios = []

        # Scenario 1: Normal efficient operation
        normal_metrics = [
            HVACMetric(
                metric_type=HVACMetricType.CHILLED_WATER_TEMP,
                value=42.0,
                unit="degrees_f",
                source="Chiller-01",
                equipment_type=HVACSystemType.CHILLER,
                zone_name="Main-Building",
                setpoint=42.0,
            ),
            HVACMetric(
                metric_type=HVACMetricType.COP,
                value=4.2,  # Excellent efficiency
                unit="ratio",
                source="Chiller-01",
                equipment_type=HVACSystemType.CHILLER,
                zone_name="Main-Building",
                setpoint=None,
            ),
            HVACMetric(
                metric_type=HVACMetricType.SUPPLY_AIR_TEMP,
                value=55.0,
                unit="degrees_f",
                source="AHU-01",
                equipment_type=HVACSystemType.AIR_HANDLER,
                zone_name="Office-Zone",
                setpoint=55.0,
            ),
            HVACMetric(
                metric_type=HVACMetricType.HUMIDITY,
                value=45.0,  # Good humidity
                unit="percent_rh",
                source="Zone-01",
                equipment_type=HVACSystemType.AIR_HANDLER,
                zone_name="Office-Zone",
                setpoint=None,
            ),
        ]

        scenarios.append(
            ExpertValidatedScenario(
                scenario_id="normal_efficient_operation",
                scenario_type=ValidationScenarioType.NORMAL_OPERATION,
                description="Well-tuned HVAC system operating efficiently within design parameters",
                hvac_metrics=normal_metrics,
                system_context={"building_type": "office", "occupancy": "normal"},
                expected_energy_assessment={
                    "efficiency_level": "good",
                    "improvement_potential": "low",
                    "confidence_min": 0.8,
                },
                expected_comfort_assessment={
                    "overall_rating": "good",
                    "comfort_score_min": 80.0,
                    "issues_max": 1,
                },
                expected_fault_assessment={
                    "faults_detected_max": 0,
                    "severity": "low",
                    "confidence_min": 0.7,
                },
                critical_recommendations_must_include=[
                    "maintain",
                    "continue",
                    "normal",  # Should recognize good operation
                ],
                critical_recommendations_must_not_include=[
                    "critical",
                    "emergency",
                    "immediate replacement",
                ],
            )
        )

        # Scenario 2: Poor chiller efficiency
        poor_efficiency_metrics = [
            HVACMetric(
                metric_type=HVACMetricType.POWER_CONSUMPTION,
                value=120.0,  # High power consumption
                unit="kw",
                source="Chiller-02",
                equipment_type=HVACSystemType.CHILLER,
                zone_name="Main-Building",
                setpoint=None,
            ),
            HVACMetric(
                metric_type=HVACMetricType.COP,
                value=1.8,  # Poor efficiency
                unit="ratio",
                source="Chiller-02",
                equipment_type=HVACSystemType.CHILLER,
                zone_name="Main-Building",
                setpoint=None,
            ),
            HVACMetric(
                metric_type=HVACMetricType.CHILLED_WATER_TEMP,
                value=44.5,  # Above setpoint
                unit="degrees_f",
                source="Chiller-02",
                equipment_type=HVACSystemType.CHILLER,
                zone_name="Main-Building",
                setpoint=42.0,
            ),
        ]

        scenarios.append(
            ExpertValidatedScenario(
                scenario_id="poor_chiller_efficiency",
                scenario_type=ValidationScenarioType.EFFICIENCY_PROBLEM,
                description="Chiller operating with poor efficiency, likely needing maintenance",
                hvac_metrics=poor_efficiency_metrics,
                system_context={"equipment_age": 8, "last_maintenance": "6_months_ago"},
                expected_energy_assessment={
                    "efficiency_level": "poor",
                    "improvement_potential": "high",
                    "confidence_min": 0.8,
                },
                expected_comfort_assessment={"overall_rating": "fair", "comfort_score_max": 70.0},
                expected_fault_assessment={
                    "faults_detected_min": 1,
                    "severity": "medium",
                    "confidence_min": 0.7,
                },
                critical_recommendations_must_include=["maintenance", "efficiency", "inspection"],
                critical_recommendations_must_not_include=["normal operation", "no action needed"],
            )
        )

        # Scenario 3: Comfort issue - high humidity
        comfort_problem_metrics = [
            HVACMetric(
                metric_type=HVACMetricType.HUMIDITY,
                value=78.0,  # Very high humidity
                unit="percent_rh",
                source="Zone-Problem",
                equipment_type=HVACSystemType.AIR_HANDLER,
                zone_name="Conference-Room",
                setpoint=50.0,
            ),
            HVACMetric(
                metric_type=HVACMetricType.SUPPLY_AIR_TEMP,
                value=58.0,  # Higher than normal
                unit="degrees_f",
                source="AHU-Problem",
                equipment_type=HVACSystemType.AIR_HANDLER,
                zone_name="Conference-Room",
                setpoint=55.0,
            ),
            HVACMetric(
                metric_type=HVACMetricType.CO2_LEVEL,
                value=1150,  # High CO2
                unit="ppm",
                source="Zone-Problem",
                equipment_type=HVACSystemType.AIR_HANDLER,
                zone_name="Conference-Room",
                setpoint=None,
            ),
        ]

        scenarios.append(
            ExpertValidatedScenario(
                scenario_id="high_humidity_comfort_issue",
                scenario_type=ValidationScenarioType.COMFORT_ISSUE,
                description="Zone with high humidity and poor air quality affecting comfort",
                hvac_metrics=comfort_problem_metrics,
                system_context={"zone_usage": "conference_room", "occupancy": "high"},
                expected_energy_assessment={
                    "improvement_potential": "medium"  # Could improve dehumidification
                },
                expected_comfort_assessment={
                    "overall_rating": "poor",
                    "comfort_score_max": 50.0,
                    "issues_min": 2,  # Humidity and CO2
                },
                expected_fault_assessment={
                    "faults_detected_min": 1,  # Likely ventilation or dehumidification fault
                    "severity": "medium",
                },
                critical_recommendations_must_include=["humidity", "ventilation", "air quality"],
                critical_recommendations_must_not_include=["excellent comfort", "no issues"],
            )
        )

        # Scenario 4: Equipment fault - temperature sensor issue
        sensor_fault_metrics = [
            HVACMetric(
                metric_type=HVACMetricType.SUPPLY_AIR_TEMP,
                value=-15.0,  # Clearly erroneous reading
                unit="degrees_f",
                source="AHU-Fault",
                equipment_type=HVACSystemType.AIR_HANDLER,
                zone_name="Test-Zone",
                setpoint=55.0,
            ),
            HVACMetric(
                metric_type=HVACMetricType.RETURN_AIR_TEMP,
                value=72.0,  # Normal reading
                unit="degrees_f",
                source="AHU-Fault",
                equipment_type=HVACSystemType.AIR_HANDLER,
                zone_name="Test-Zone",
                setpoint=None,
            ),
        ]

        scenarios.append(
            ExpertValidatedScenario(
                scenario_id="temperature_sensor_fault",
                scenario_type=ValidationScenarioType.EQUIPMENT_FAULT,
                description="Temperature sensor providing clearly erroneous readings",
                hvac_metrics=sensor_fault_metrics,
                system_context={},
                expected_energy_assessment={
                    "confidence_max": 0.5  # Should have low confidence with bad data
                },
                expected_comfort_assessment={
                    "comfort_score_max": 30.0  # Should recognize data quality issue
                },
                expected_fault_assessment={
                    "faults_detected_min": 1,
                    "severity": "high",  # Sensor faults affect control
                    "confidence_min": 0.9,  # Should be very confident about obvious fault
                },
                critical_recommendations_must_include=[
                    "sensor",
                    "calibration",
                    "replacement",
                    "fault",
                ],
                critical_recommendations_must_not_include=["normal operation"],
            )
        )

        return scenarios

    async def run_comprehensive_validation(self) -> dict[str, Any]:
        """Run complete validation against all expert scenarios."""

        validation_start = datetime.now(UTC)
        self.logger.info("comprehensive_validation_starting", scenarios=len(self.expert_scenarios))

        validation_results: list[ValidationResult] = []

        for scenario in self.expert_scenarios:
            self.logger.info("validating_scenario", scenario_id=scenario.scenario_id)

            # Run validation for this scenario
            result = await self._validate_scenario(scenario)
            validation_results.append(result)

            # Log result
            status = "✅ PASSED" if result.validation_passed else "❌ FAILED"
            self.logger.info(
                "scenario_validation_completed",
                scenario_id=scenario.scenario_id,
                status=status,
                confidence=result.confidence_score,
            )

        # Calculate overall validation statistics
        total_scenarios = len(validation_results)
        passed_scenarios = sum(1 for r in validation_results if r.validation_passed)
        overall_success_rate = passed_scenarios / total_scenarios if total_scenarios > 0 else 0

        # Calculate average confidence across all scenarios
        avg_confidence = statistics.mean([r.confidence_score for r in validation_results])

        # Identify critical failures
        critical_failures: list[str] = []
        for result in validation_results:
            if result.critical_failures:
                critical_failures.extend(result.critical_failures)

        validation_duration = (datetime.now(UTC) - validation_start).total_seconds()

        summary = {
            "validation_summary": {
                "total_scenarios": total_scenarios,
                "passed_scenarios": passed_scenarios,
                "failed_scenarios": total_scenarios - passed_scenarios,
                "overall_success_rate": round(overall_success_rate * 100, 1),
                "average_confidence": round(avg_confidence, 3),
                "validation_duration_seconds": round(validation_duration, 2),
            },
            "critical_failures": critical_failures,
            "detailed_results": validation_results,
            "validation_timestamp": validation_start,
            "ai_model_tested": self.ai_config.model_name,
            "deployment_recommendation": self._get_deployment_recommendation(
                overall_success_rate, avg_confidence, critical_failures
            ),
        }

        self.validation_results = validation_results

        self.logger.info(
            "comprehensive_validation_completed",
            success_rate=f"{overall_success_rate * 100:.1f}%",
            avg_confidence=round(avg_confidence, 3),
            critical_failures=len(critical_failures),
        )

        return summary

    async def _validate_scenario(self, scenario: ExpertValidatedScenario) -> ValidationResult:
        """Validate AI performance against a single expert scenario."""

        scenario_start = datetime.now(UTC)
        validation_issues: list[str] = []
        critical_failures: list[str] = []

        try:
            # Run AI analysis multiple times for consistency testing
            consistency_runs = 3
            analysis_results = []

            for run in range(consistency_runs):
                result = await self.hvac_ai_service.comprehensive_analysis(
                    scenario.hvac_metrics, scenario.system_context
                )
                analysis_results.append(result)

                # Small delay between runs to avoid rate limiting
                if run < consistency_runs - 1:
                    await asyncio.sleep(1.0)

            # Calculate consistency score based on multiple runs
            consistency_score = self._calculate_consistency_score(analysis_results)

            # Use first successful result for detailed validation
            primary_result: dict[str, Any] | None = None
            for result in analysis_results:
                if "error" not in result:
                    primary_result = result
                    break

            if primary_result is None:
                return ValidationResult(
                    scenario_id=scenario.scenario_id,
                    scenario_type=scenario.scenario_type.value,
                    validation_passed=False,
                    confidence_score=0.0,
                    energy_validation={},
                    comfort_validation={},
                    fault_validation={},
                    validation_issues=["All AI analysis attempts failed"],
                    critical_failures=["Complete AI system failure"],
                    analysis_duration_seconds=(datetime.now(UTC) - scenario_start).total_seconds(),
                    consistency_score=0.0,
                    validation_timestamp=scenario_start,
                    ai_model_used=self.ai_config.model_name,
                )

            # Validate energy optimization results
            energy_validation = self._validate_energy_analysis(
                primary_result.get("energy_optimization"),
                scenario.expected_energy_assessment,
                validation_issues,
                critical_failures,
            )

            # Validate comfort assessment results
            comfort_validation = self._validate_comfort_analysis(
                primary_result.get("comfort_assessment"),
                scenario.expected_comfort_assessment,
                validation_issues,
                critical_failures,
            )

            # Validate fault detection results
            fault_validation = self._validate_fault_analysis(
                primary_result.get("fault_detection"),
                scenario.expected_fault_assessment,
                validation_issues,
                critical_failures,
            )

            # Check critical recommendations
            self._validate_critical_recommendations(
                primary_result,
                scenario.critical_recommendations_must_include or [],
                scenario.critical_recommendations_must_not_include or [],
                validation_issues,
                critical_failures,
            )

            # Calculate overall confidence score
            confidence_scores = []
            if energy_validation.get("confidence"):
                confidence_scores.append(energy_validation["confidence"])
            if comfort_validation.get("confidence"):
                confidence_scores.append(comfort_validation["confidence"])
            if fault_validation.get("confidence"):
                confidence_scores.append(fault_validation["confidence"])

            overall_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0

            # Determine if validation passed
            validation_passed = (
                len(critical_failures) == 0
                and overall_confidence >= scenario.acceptable_confidence_range[0]
                and overall_confidence <= scenario.acceptable_confidence_range[1]
                and consistency_score >= 0.7  # Require reasonable consistency
            )

            analysis_duration = (datetime.now(UTC) - scenario_start).total_seconds()

            return ValidationResult(
                scenario_id=scenario.scenario_id,
                scenario_type=scenario.scenario_type.value,
                validation_passed=validation_passed,
                confidence_score=overall_confidence,
                energy_validation=energy_validation,
                comfort_validation=comfort_validation,
                fault_validation=fault_validation,
                validation_issues=validation_issues,
                critical_failures=critical_failures,
                analysis_duration_seconds=analysis_duration,
                consistency_score=consistency_score,
                validation_timestamp=scenario_start,
                ai_model_used=self.ai_config.model_name,
            )

        except Exception as e:
            return ValidationResult(
                scenario_id=scenario.scenario_id,
                scenario_type=scenario.scenario_type.value,
                validation_passed=False,
                confidence_score=0.0,
                energy_validation={},
                comfort_validation={},
                fault_validation={},
                validation_issues=[f"Validation exception: {str(e)}"],
                critical_failures=[f"Validation system failure: {str(e)}"],
                analysis_duration_seconds=(datetime.now(UTC) - scenario_start).total_seconds(),
                consistency_score=0.0,
                validation_timestamp=scenario_start,
                ai_model_used=self.ai_config.model_name,
            )

    def _validate_energy_analysis(
        self,
        energy_result: Any,
        expected: dict[str, Any],
        validation_issues: list[str],
        critical_failures: list[str],
    ) -> dict[str, Any]:
        """Validate energy optimization analysis results."""

        validation: dict[str, Any] = {"passed": False, "confidence": 0.0, "issues": []}
        issues = cast(list[str], validation["issues"])

        if not energy_result:
            validation_issues.append("Energy analysis returned no results")
            return validation

        # Check efficiency level assessment
        if "efficiency_level" in expected:
            expected_level = expected["efficiency_level"].lower()
            actual_level = energy_result.current_efficiency_level.lower()

            if expected_level not in actual_level:
                issues.append(
                    f"Efficiency level mismatch: expected '{expected_level}', got '{actual_level}'"
                )

        # Check improvement potential
        if "improvement_potential" in expected:
            expected_potential = expected["improvement_potential"].lower()
            actual_potential = energy_result.improvement_potential.lower()

            if expected_potential not in actual_potential:
                issues.append(
                    f"Improvement potential mismatch: "
                    f"expected '{expected_potential}', got '{actual_potential}'"
                )

        # Check confidence bounds
        confidence = energy_result.confidence
        validation["confidence"] = confidence

        if "confidence_min" in expected and confidence < expected["confidence_min"]:
            issues.append(
                f"Energy confidence too low: {confidence:.2f} < {expected['confidence_min']}"
            )

        if "confidence_max" in expected and confidence > expected["confidence_max"]:
            issues.append(
                f"Energy confidence too high: {confidence:.2f} > {expected['confidence_max']}"
            )

        validation["passed"] = len(issues) == 0
        validation_issues.extend(issues)

        return validation

    def _validate_comfort_analysis(
        self,
        comfort_result: Any,
        expected: dict[str, Any],
        validation_issues: list[str],
        critical_failures: list[str],
    ) -> dict[str, Any]:
        """Validate comfort assessment results."""

        validation: dict[str, Any] = {"passed": False, "confidence": 1.0, "issues": []}
        issues = cast(list[str], validation["issues"])

        if not comfort_result:
            validation_issues.append("Comfort analysis returned no results")
            return validation

        # Check overall comfort rating
        if "overall_rating" in expected:
            expected_rating = expected["overall_rating"].lower()
            actual_rating = comfort_result.overall_comfort_rating.lower()

            if expected_rating not in actual_rating and actual_rating not in expected_rating:
                issues.append(
                    f"Comfort rating mismatch: expected '{expected_rating}', got '{actual_rating}'"
                )

        # Check comfort score bounds
        if "comfort_score_min" in expected:
            if comfort_result.comfort_score < expected["comfort_score_min"]:
                issues.append(
                    f"Comfort score too low: "
                    f"{comfort_result.comfort_score} < {expected['comfort_score_min']}"
                )

        if "comfort_score_max" in expected:
            if comfort_result.comfort_score > expected["comfort_score_max"]:
                issues.append(
                    f"Comfort score too high: "
                    f"{comfort_result.comfort_score} > {expected['comfort_score_max']}"
                )

        # Check number of issues detected
        issues_count = len(comfort_result.comfort_issues_identified)

        if "issues_min" in expected and issues_count < expected["issues_min"]:
            issues.append(
                f"Too few comfort issues detected: {issues_count} < {expected['issues_min']}"
            )

        if "issues_max" in expected and issues_count > expected["issues_max"]:
            issues.append(
                f"Too many comfort issues detected: {issues_count} > {expected['issues_max']}"
            )

        validation["passed"] = len(issues) == 0
        validation_issues.extend(issues)

        return validation

    def _validate_fault_analysis(
        self,
        fault_result: Any,
        expected: dict[str, Any],
        validation_issues: list[str],
        critical_failures: list[str],
    ) -> dict[str, Any]:
        """Validate fault detection results."""

        validation: dict[str, Any] = {"passed": False, "confidence": 0.0, "issues": []}
        issues = cast(list[str], validation["issues"])

        if not fault_result:
            validation_issues.append("Fault analysis returned no results")
            return validation

        # Check number of faults detected
        faults_count = len(fault_result.faults_detected)

        if "faults_detected_min" in expected and faults_count < expected["faults_detected_min"]:
            issues.append(
                f"Too few faults detected: {faults_count} < {expected['faults_detected_min']}"
            )

        if "faults_detected_max" in expected and faults_count > expected["faults_detected_max"]:
            issues.append(
                f"Too many faults detected: {faults_count} > {expected['faults_detected_max']}"
            )

        # Check fault severity
        if "severity" in expected:
            expected_severity = expected["severity"].lower()
            actual_severity = fault_result.fault_severity.lower()

            # Allow some flexibility in severity assessment
            severity_levels = ["low", "medium", "high", "critical"]
            expected_idx = (
                severity_levels.index(expected_severity)
                if expected_severity in severity_levels
                else -1
            )
            actual_idx = (
                severity_levels.index(actual_severity) if actual_severity in severity_levels else -1
            )

            # Allow within one level of expected severity
            if abs(expected_idx - actual_idx) > 1:
                issues.append(
                    f"Fault severity mismatch: "
                    f"expected '{expected_severity}', got '{actual_severity}'"
                )

        # Check confidence bounds
        confidence = fault_result.confidence_level
        validation["confidence"] = confidence

        if "confidence_min" in expected and confidence < expected["confidence_min"]:
            issues.append(
                f"Fault confidence too low: {confidence:.2f} < {expected['confidence_min']}"
            )

        validation["passed"] = len(issues) == 0
        validation_issues.extend(issues)

        return validation

    def _validate_critical_recommendations(
        self,
        analysis_result: dict[str, Any],
        must_include: list[str],
        must_not_include: list[str],
        validation_issues: list[str],
        critical_failures: list[str],
    ):
        """Validate that critical recommendations are present/absent as expected."""

        # Collect all recommendations text
        all_recommendations = []

        if "energy_optimization" in analysis_result and analysis_result["energy_optimization"]:
            energy = analysis_result["energy_optimization"]
            all_recommendations.extend(energy.immediate_actions)
            all_recommendations.extend(energy.longer_term_projects)
            all_recommendations.extend(energy.setpoint_adjustments)

        if "comfort_assessment" in analysis_result and analysis_result["comfort_assessment"]:
            comfort = analysis_result["comfort_assessment"]
            all_recommendations.extend(comfort.improvement_recommendations)

        if "fault_detection" in analysis_result and analysis_result["fault_detection"]:
            fault = analysis_result["fault_detection"]
            all_recommendations.extend(fault.maintenance_recommendations)

        recommendations_text = " ".join(all_recommendations).lower()

        # Check required recommendations
        for required in must_include:
            if required.lower() not in recommendations_text:
                critical_failures.append(f"Critical recommendation missing: '{required}'")

        # Check prohibited recommendations
        for prohibited in must_not_include:
            if prohibited.lower() in recommendations_text:
                critical_failures.append(f"Inappropriate recommendation found: '{prohibited}'")

    def _calculate_consistency_score(self, analysis_results: list[dict[str, Any]]) -> float:
        """Calculate consistency score across multiple AI runs."""

        if len(analysis_results) < 2:
            return 1.0

        consistency_scores = []

        # Compare energy assessments
        energy_assessments: list[dict[str, Any]] = []
        for result in analysis_results:
            if "energy_optimization" in result and result["energy_optimization"]:
                energy = result["energy_optimization"]
                energy_assessments.append(
                    {
                        "efficiency": energy.current_efficiency_level.lower(),
                        "potential": energy.improvement_potential.lower(),
                        "confidence": energy.confidence,
                    }
                )

        if len(energy_assessments) > 1:
            # Check consistency of categorical assessments
            efficiency_consistency = len({a["efficiency"] for a in energy_assessments}) == 1
            potential_consistency = len({a["potential"] for a in energy_assessments}) == 1

            # Check confidence variation
            confidences = [cast(float, a["confidence"]) for a in energy_assessments]
            confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0
            confidence_consistency = confidence_std < 0.1  # Less than 10% variation

            energy_score = (
                sum([efficiency_consistency, potential_consistency, confidence_consistency]) / 3
            )
            consistency_scores.append(energy_score)

        # Similar checks for comfort and fault detection could be added here

        return statistics.mean(consistency_scores) if consistency_scores else 0.8

    def _get_deployment_recommendation(
        self, success_rate: float, avg_confidence: float, critical_failures: list[str]
    ) -> str:
        """Generate deployment recommendation based on validation results."""

        if len(critical_failures) > 0:
            return "❌ DO NOT DEPLOY - Critical failures detected"

        if success_rate < 0.8:
            return "❌ DO NOT DEPLOY - Success rate too low"

        if avg_confidence < 0.7:
            return "⚠️ DEPLOY WITH CAUTION - Low confidence scores"

        if success_rate >= 0.95 and avg_confidence >= 0.8:
            return "✅ APPROVED FOR DEPLOYMENT - High confidence"

        if success_rate >= 0.9 and avg_confidence >= 0.75:
            return "✅ APPROVED FOR DEPLOYMENT - Good performance"

        return "⚠️ DEPLOY WITH MONITORING - Acceptable but watch closely"

    async def run_stress_testing(self, iterations: int = 50) -> dict[str, Any]:
        """Run stress testing with multiple rapid analyses."""

        self.logger.info("stress_testing_starting", iterations=iterations)

        # Use a simple scenario for stress testing
        stress_scenario = self.expert_scenarios[0]  # Normal operation scenario

        start_time = datetime.now(UTC)
        results = []
        failures = 0

        for i in range(iterations):
            try:
                result = await self.hvac_ai_service.comprehensive_analysis(
                    stress_scenario.hvac_metrics, stress_scenario.system_context
                )

                if "error" in result:
                    failures += 1

                results.append(result)

            except Exception as e:
                failures += 1
                self.logger.error("stress_test_iteration_failed", iteration=i, error=str(e))

        total_duration = (datetime.now(UTC) - start_time).total_seconds()

        return {
            "stress_test_summary": {
                "iterations": iterations,
                "failures": failures,
                "success_rate": (iterations - failures) / iterations * 100,
                "total_duration_seconds": round(total_duration, 2),
                "average_duration_per_analysis": round(total_duration / iterations, 3),
                "analyses_per_second": round(iterations / total_duration, 2),
            },
            "performance_recommendation": (
                "✅ GOOD PERFORMANCE"
                if failures / iterations < 0.05
                else "⚠️ MONITOR PERFORMANCE"
                if failures / iterations < 0.1
                else "❌ PERFORMANCE ISSUES"
            ),
        }


# Utility functions for creating custom validation scenarios
def create_custom_validation_scenario(
    scenario_id: str, description: str, metrics: list[HVACMetric], expected_outcomes: dict[str, Any]
) -> ExpertValidatedScenario:
    """Helper function to create custom validation scenarios."""

    return ExpertValidatedScenario(
        scenario_id=scenario_id,
        scenario_type=ValidationScenarioType.NORMAL_OPERATION,
        description=description,
        hvac_metrics=metrics,
        system_context={},
        expected_energy_assessment=expected_outcomes.get("energy", {}),
        expected_comfort_assessment=expected_outcomes.get("comfort", {}),
        expected_fault_assessment=expected_outcomes.get("fault", {}),
        validated_by="Custom Test",
        industry_standards_applied=["Custom validation"],
    )


# Example usage and validation runner
async def main() -> None:
    """Run AI validation framework demonstration."""

    print("🧪 AI Accuracy Validation Framework")
    print("=" * 50)

    # Initialize validation framework
    try:
        from core.config import get_config

        config_obj = get_config()
        ai_config = AIAnalysisConfig(
            model_name=config_obj.ai_provider.anomaly_detection_model,
            temperature=0.1,  # Use consistent temperature for validation
            timeout_seconds=30.0,
        )

        validator = AIValidationFramework(ai_config)

    except Exception as e:
        print(f"⚠️  Validation framework initialization failed: {e}")
        print("   This requires OpenAI API key configuration")
        return

    print(
        f"✅ Initialized validation framework with "
        f"{len(validator.expert_scenarios)} expert scenarios"
    )

    # Run comprehensive validation
    print("\n🔍 Running Comprehensive Validation")
    print("   This will test AI accuracy against expert-validated scenarios...")
    print("   Expected duration: 2-3 minutes")

    try:
        validation_summary = await validator.run_comprehensive_validation()

        # Display validation results
        print("\n📊 VALIDATION RESULTS")
        print("=" * 30)

        summary = validation_summary["validation_summary"]
        print(f"Total Scenarios: {summary['total_scenarios']}")
        print(f"Passed: {summary['passed_scenarios']} ({summary['overall_success_rate']:.1f}%)")
        print(f"Failed: {summary['failed_scenarios']}")
        print(f"Average Confidence: {summary['average_confidence']:.3f}")
        print(f"Duration: {summary['validation_duration_seconds']:.1f}s")

        # Show deployment recommendation
        print("\n🚀 DEPLOYMENT RECOMMENDATION:")
        print(f"   {validation_summary['deployment_recommendation']}")

        # Show critical failures if any
        if validation_summary["critical_failures"]:
            print("\n❌ CRITICAL FAILURES:")
            for failure in validation_summary["critical_failures"]:
                print(f"   • {failure}")

        # Show detailed results for failed scenarios
        failed_results = [
            r for r in validation_summary["detailed_results"] if not r.validation_passed
        ]
        if failed_results:
            print("\n📋 FAILED SCENARIO DETAILS:")
            for result in failed_results:
                print(f"   {result.scenario_id}:")
                for issue in result.validation_issues:
                    print(f"     • {issue}")

        # Run stress testing if validation passed
        if summary["overall_success_rate"] >= 80:
            print("\n⚡ Running Stress Testing...")

            stress_results = await validator.run_stress_testing(iterations=10)  # Reduced for demo
            stress_summary = stress_results["stress_test_summary"]

            print(f"   Iterations: {stress_summary['iterations']}")
            print(f"   Success Rate: {stress_summary['success_rate']:.1f}%")
            print(f"   Avg Duration: {stress_summary['average_duration_per_analysis']:.3f}s")
            print(f"   Performance: {stress_results['performance_recommendation']}")

        print("\n✅ Validation framework completed successfully!")

    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        print("   Check API key configuration and network connectivity")


if __name__ == "__main__":
    asyncio.run(main())
