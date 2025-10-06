"""
Generic HVAC AI agents suitable for open source publication.

These agents demonstrate:
1. Multi-agent architecture patterns
2. Domain-specific AI prompting techniques
3. Structured output validation
4. Error handling and fallback strategies

Strategic positioning: Show technical competence and HVAC knowledge
without revealing proprietary optimization algorithms or customer-specific IP.
"""

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from statistics import mean
from typing import Any

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from adapters.hvac.domain import HVACMetric, HVACMetricType, HVACSystemType
from core.services.ai_analysis import AIAnalysisConfig

logger = structlog.get_logger()


class EnergyOptimizationSuggestion(BaseModel):
    """Generic energy optimization recommendations."""

    # Overall assessment
    current_efficiency_level: str = Field(description="Current system efficiency rating")
    improvement_potential: str = Field(description="Low/Medium/High improvement potential")
    confidence: float = Field(ge=0.0, le=1.0, description="Analysis confidence")

    # Specific recommendations (generic patterns)
    setpoint_adjustments: list[str] = Field(description="Suggested setpoint modifications")
    scheduling_optimizations: list[str] = Field(description="Operating schedule improvements")
    control_strategy_improvements: list[str] = Field(description="Control logic optimizations")

    # Educational insights (safe to share publicly)
    energy_efficiency_principles: list[str] = Field(
        description="General HVAC efficiency principles applied"
    )
    estimated_impact: str = Field(description="Qualitative impact assessment")

    # Implementation guidance
    immediate_actions: list[str] = Field(description="Actions that can be taken immediately")
    longer_term_projects: list[str] = Field(description="Projects requiring planning or investment")


class ComfortAssessmentResult(BaseModel):
    """Generic occupant comfort assessment."""

    # Overall comfort rating
    overall_comfort_rating: str = Field(description="Excellent/Good/Fair/Poor")
    comfort_score: float = Field(ge=0.0, le=100.0, description="Numeric comfort score")

    # Component assessments
    temperature_assessment: str = Field(description="Temperature comfort evaluation")
    humidity_assessment: str = Field(description="Humidity comfort evaluation")
    air_quality_assessment: str = Field(description="Air quality evaluation")

    # Issues and recommendations
    comfort_issues_identified: list[str] = Field(description="Specific comfort problems found")
    improvement_recommendations: list[str] = Field(description="Suggestions to improve comfort")

    # Zone-specific insights
    zone_analysis: dict[str, str] = Field(
        default_factory=dict, description="Comfort assessment by zone"
    )

    # Educational content
    comfort_principles: list[str] = Field(description="ASHRAE comfort principles applied")


class FaultDetectionResult(BaseModel):
    """Generic fault detection and diagnostics."""

    # Fault analysis
    faults_detected: list[str] = Field(description="Potential equipment faults identified")
    fault_severity: str = Field(description="Overall fault severity: Low/Medium/High/Critical")
    confidence_level: float = Field(ge=0.0, le=1.0, description="Confidence in fault detection")

    # Diagnostic insights
    performance_degradation_indicators: list[str] = Field(
        description="Signs of equipment performance decline"
    )
    maintenance_recommendations: list[str] = Field(description="Suggested maintenance actions")

    # Pattern analysis
    trend_analysis: str = Field(description="Equipment performance trend assessment")
    anomaly_indicators: list[str] = Field(description="Unusual patterns in equipment behavior")

    # Risk assessment
    risk_factors: list[str] = Field(description="Factors that increase failure risk")
    monitoring_priorities: list[str] = Field(description="Key metrics to monitor closely")


class GenericEnergyOptimizationAgent:
    """
    Generic energy optimization agent for HVAC systems.

    Focus: General efficiency principles and common optimization patterns
    without revealing proprietary algorithms or customer-specific strategies.
    """

    def __init__(self, config: AIAnalysisConfig):
        self.config = config
        self.logger = logger.bind(component="generic_energy_agent")

        self.agent = Agent(
            model=config.model_name,
            result_type=EnergyOptimizationSuggestion,
            system_prompt=self._build_system_prompt(),
        )

    def _build_system_prompt(self) -> str:
        return """You are an HVAC Energy Efficiency Consultant with expertise in
commercial building systems optimization.

Your role is to analyze HVAC performance data and provide general energy efficiency
recommendations based on established industry best practices.

Key principles to apply:
1. EQUIPMENT EFFICIENCY: Operate equipment in optimal efficiency ranges
2. LOAD MATCHING: Match system capacity to actual load requirements
3. FREE COOLING: Maximize use of favorable outdoor conditions
4. SCHEDULING: Optimize operating schedules based on occupancy
5. SETPOINT OPTIMIZATION: Adjust setpoints within comfort ranges for efficiency
6. SYSTEM INTEGRATION: Coordinate multiple systems for overall efficiency

Focus on:
- Industry-standard optimization strategies
- General HVAC efficiency principles
- Common control improvements
- Basic maintenance practices that improve efficiency
- Educational insights about HVAC system operation

Avoid:
- Specific proprietary optimization algorithms
- Customer-specific strategies
- Detailed equipment-specific tuning parameters
- Advanced control sequences requiring specialized knowledge

Provide practical, implementable recommendations that demonstrate
understanding of HVAC systems without revealing competitive advantages."""

    async def analyze_energy_efficiency(
        self, hvac_metrics: list[HVACMetric], system_context: dict[str, Any] | None = None
    ) -> EnergyOptimizationSuggestion:
        """Analyze HVAC metrics for general energy optimization opportunities."""

        try:
            # Prepare analysis context
            context = self._prepare_analysis_context(hvac_metrics, system_context)

            # Build analysis prompt
            prompt = self._build_analysis_prompt(hvac_metrics, context)

            # Run AI analysis
            result = await self.agent.run(prompt)

            self.logger.info(
                "energy_optimization_analysis_completed",
                improvement_potential=result.data.improvement_potential,
                confidence=result.data.confidence,
            )

            return result.data

        except Exception as e:
            self.logger.error("energy_optimization_analysis_failed", error=str(e))

            # Return educational fallback
            return EnergyOptimizationSuggestion(
                current_efficiency_level="Unable to determine",
                improvement_potential="Unknown",
                confidence=0.1,
                setpoint_adjustments=["Verify all setpoints are within design ranges"],
                scheduling_optimizations=[
                    "Review operating schedules for optimization opportunities"
                ],
                control_strategy_improvements=[
                    "Ensure all control systems are functioning properly"
                ],
                energy_efficiency_principles=[
                    "Match system output to actual load requirements",
                    "Maximize free cooling opportunities when available",
                    "Optimize equipment staging and sequencing",
                ],
                estimated_impact="Unable to assess without successful analysis",
                immediate_actions=["Schedule comprehensive energy audit"],
                longer_term_projects=["Consider building automation system upgrades"],
            )

    def _prepare_analysis_context(
        self, metrics: list[HVACMetric], system_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Prepare context for AI analysis."""

        context = {
            "metrics_summary": {},
            "efficiency_indicators": {},
            "system_characteristics": system_context or {},
        }

        # Group metrics by type and equipment
        metrics_by_type = defaultdict(list)
        metrics_by_equipment = defaultdict(list)

        for metric in metrics:
            metrics_by_type[metric.metric_type].append(metric.value)
            metrics_by_equipment[metric.equipment_type].append(metric)

        # Calculate efficiency indicators
        for metric_type, values in metrics_by_type.items():
            if values:
                context["metrics_summary"][metric_type.value] = {
                    "average": mean(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        # Equipment-specific analysis
        for equipment_type, equipment_metrics in metrics_by_equipment.items():
            power_metrics = [
                m for m in equipment_metrics if m.metric_type == HVACMetricType.POWER_CONSUMPTION
            ]
            efficiency_metrics = [
                m for m in equipment_metrics if m.metric_type == HVACMetricType.COP
            ]

            if power_metrics:
                avg_power = mean([m.value for m in power_metrics])
                context["efficiency_indicators"][f"{equipment_type.value}_power"] = avg_power

            if efficiency_metrics:
                avg_efficiency = mean([m.value for m in efficiency_metrics])
                context["efficiency_indicators"][f"{equipment_type.value}_efficiency"] = (
                    avg_efficiency
                )

        return context

    def _build_analysis_prompt(self, metrics: list[HVACMetric], context: dict[str, Any]) -> str:
        """Build specific analysis prompt."""

        # Summarize current performance
        performance_summary = []
        for metric_type, stats in context["metrics_summary"].items():
            line = (
                f"- {metric_type}: avg {stats['average']:.1f}, "
                f"range {stats['min']:.1f}-{stats['max']:.1f}"
            )
            performance_summary.append(line)

        # Equipment efficiency summary
        efficiency_summary = []
        for indicator, value in context["efficiency_indicators"].items():
            efficiency_summary.append(f"- {indicator}: {value:.2f}")

        return f"""Analyze this HVAC system for energy optimization opportunities:

PERFORMANCE METRICS:
{chr(10).join(performance_summary)}

EFFICIENCY INDICATORS:
{chr(10).join(efficiency_summary) if efficiency_summary else "No efficiency metrics available"}

SYSTEM CONTEXT:
- Total data points: {len(metrics)}
- Equipment types: {len({m.equipment_type for m in metrics})}
- Time span: Recent operational data
- Zones: {len({m.zone_name for m in metrics})}

Please provide:
1. Current efficiency assessment using industry standards
2. General optimization opportunities based on HVAC best practices
3. Setpoint and scheduling recommendations within standard ranges
4. Control strategy improvements using common techniques
5. Educational insights about HVAC efficiency principles
6. Implementation guidance for immediate and longer-term actions

Focus on proven, industry-standard optimization strategies that demonstrate
HVAC expertise without revealing proprietary methods."""


class GenericComfortAssessmentAgent:
    """
    Generic comfort assessment agent based on ASHRAE standards.

    Focus: Standard comfort criteria and general assessment principles
    without customer-specific comfort models or proprietary algorithms.
    """

    def __init__(self, config: AIAnalysisConfig):
        self.config = config
        self.logger = logger.bind(component="generic_comfort_agent")

        self.agent = Agent(
            model=config.model_name,
            result_type=ComfortAssessmentResult,
            system_prompt=self._build_system_prompt(),
        )

    def _build_system_prompt(self) -> str:
        return """You are an Indoor Environmental Quality (IEQ) Specialist with expertise
in occupant comfort assessment based on industry standards.

Your role is to evaluate HVAC system performance against established comfort standards
and provide recommendations for comfort improvement.

Apply these standards:
- ASHRAE Standard 55 (Thermal Comfort): 68-76°F operative temperature range
- ASHRAE Standard 62.1 (Ventilation): Fresh air requirements, CO2 levels
- Relative Humidity: 30-60% for optimal comfort
- Air Movement: 15-50 FPM in occupied spaces

Assessment criteria:
1. TEMPERATURE: Within comfort range, minimal variation
2. HUMIDITY: Controlled within healthy ranges
3. AIR QUALITY: Adequate ventilation, low pollutant levels
4. UNIFORMITY: Consistent conditions across zones
5. SEASONAL ADAPTATION: Appropriate for climate and clothing

Focus on:
- Standard comfort metrics and ranges
- General HVAC comfort principles
- Common comfort problems and solutions
- ASHRAE guideline applications
- Basic troubleshooting approaches

Provide educational insights about comfort standards and practical
recommendations that any HVAC professional could implement."""

    async def assess_comfort(
        self, hvac_metrics: list[HVACMetric], occupancy_context: dict[str, Any] | None = None
    ) -> ComfortAssessmentResult:
        """Assess occupant comfort based on HVAC metrics."""

        try:
            # Analyze comfort metrics
            comfort_analysis = self._analyze_comfort_metrics(hvac_metrics)

            # Build assessment prompt
            prompt = self._build_assessment_prompt(
                hvac_metrics, comfort_analysis, occupancy_context
            )

            # Run AI analysis
            result = await self.agent.run(prompt)

            self.logger.info(
                "comfort_assessment_completed",
                comfort_rating=result.data.overall_comfort_rating,
                comfort_score=result.data.comfort_score,
            )

            return result.data

        except Exception as e:
            self.logger.error("comfort_assessment_failed", error=str(e))

            # Return basic fallback assessment
            return ComfortAssessmentResult(
                overall_comfort_rating="Unknown",
                comfort_score=50.0,
                temperature_assessment="Unable to assess",
                humidity_assessment="Unable to assess",
                air_quality_assessment="Unable to assess",
                comfort_issues_identified=["Assessment failed - check system operation"],
                improvement_recommendations=[
                    "Verify HVAC system is operating properly",
                    "Check temperature and humidity sensors",
                    "Review space thermostat settings",
                ],
                comfort_principles=[
                    "Maintain temperature between 68-76°F for optimal comfort",
                    "Control humidity between 30-60% relative humidity",
                    "Ensure adequate fresh air ventilation",
                ],
            )

    def _analyze_comfort_metrics(self, metrics: list[HVACMetric]) -> dict[str, Any]:
        """Analyze metrics for comfort-related patterns."""

        temperature_zones: dict[str, dict[str, float | bool]] = {}
        humidity_readings: list[dict[str, Any]] = []
        air_quality_indicators: dict[str, dict[str, float | bool]] = {}
        comfort_violations: list[str] = []

        # Group by zone for zone-level analysis
        zone_metrics = defaultdict(list)
        for metric in metrics:
            zone_metrics[metric.zone_name].append(metric)

        # Analyze each zone
        for zone_name, zone_metric_list in zone_metrics.items():
            # Temperature analysis
            temp_metrics = [
                m
                for m in zone_metric_list
                if m.metric_type in [HVACMetricType.SUPPLY_AIR_TEMP, HVACMetricType.RETURN_AIR_TEMP]
            ]

            if temp_metrics:
                temps = [m.value for m in temp_metrics]
                avg_temp = mean(temps)
                temp_variation = max(temps) - min(temps) if len(temps) > 1 else 0

                temperature_zones[zone_name] = {
                    "average_temperature": avg_temp,
                    "temperature_variation": temp_variation,
                    "comfort_range": 68 <= avg_temp <= 76,
                }

                # Check for comfort violations
                if avg_temp < 68 or avg_temp > 76:
                    comfort_violations.append(
                        f"{zone_name}: Temperature {avg_temp:.1f}°F outside comfort range"
                    )

                if temp_variation > 4:  # More than 4°F variation
                    comfort_violations.append(
                        f"{zone_name}: High temperature variation ({temp_variation:.1f}°F)"
                    )

            # Humidity analysis
            humidity_metrics = [
                m for m in zone_metric_list if m.metric_type == HVACMetricType.HUMIDITY
            ]
            if humidity_metrics:
                avg_humidity = mean([m.value for m in humidity_metrics])
                humidity_readings.append(
                    {
                        "zone": zone_name,
                        "average_humidity": avg_humidity,
                        "in_comfort_range": 30 <= avg_humidity <= 60,
                    }
                )

                if avg_humidity < 30 or avg_humidity > 60:
                    comfort_violations.append(
                        f"{zone_name}: Humidity {avg_humidity:.1f}% outside comfort range"
                    )

            # Air quality analysis
            co2_metrics = [m for m in zone_metric_list if m.metric_type == HVACMetricType.CO2_LEVEL]
            if co2_metrics:
                avg_co2 = mean([m.value for m in co2_metrics])
                air_quality_indicators[zone_name] = {
                    "average_co2": avg_co2,
                    "ventilation_adequate": avg_co2 < 1000,
                }

                if avg_co2 > 1000:
                    msg = (
                        f"{zone_name}: High CO2 level ({avg_co2:.0f} ppm) "
                        "indicates inadequate ventilation"
                    )
                    comfort_violations.append(msg)

        return {
            "temperature_zones": temperature_zones,
            "humidity_readings": humidity_readings,
            "air_quality_indicators": air_quality_indicators,
            "comfort_violations": comfort_violations,
        }

    def _build_assessment_prompt(
        self,
        metrics: list[HVACMetric],
        analysis: dict[str, Any],
        occupancy_context: dict[str, Any] | None = None,
    ) -> str:
        """Build comfort assessment prompt."""

        # Zone temperature summary
        temp_summary = []
        for zone, data in analysis["temperature_zones"].items():
            status = "✓" if data["comfort_range"] else "✗"
            temp_summary.append(
                f"- {zone}: {data['average_temperature']:.1f}°F {status} "
                f"(variation: {data['temperature_variation']:.1f}°F)"
            )

        # Humidity summary
        humidity_summary = []
        for reading in analysis["humidity_readings"]:
            status = "✓" if reading["in_comfort_range"] else "✗"
            humidity_summary.append(
                f"- {reading['zone']}: {reading['average_humidity']:.1f}% RH {status}"
            )

        # Air quality summary
        air_quality_summary = []
        for zone, data in analysis["air_quality_indicators"].items():
            status = "✓" if data["ventilation_adequate"] else "✗"
            air_quality_summary.append(f"- {zone}: {data['average_co2']:.0f} ppm CO2 {status}")

        # Comfort violations
        violations_text = (
            "\n".join(analysis["comfort_violations"])
            if analysis["comfort_violations"]
            else "None detected"
        )

        return f"""Assess occupant comfort conditions based on HVAC performance data:

TEMPERATURE CONDITIONS BY ZONE:
{chr(10).join(temp_summary) if temp_summary else "No temperature data available"}

HUMIDITY CONDITIONS:
{chr(10).join(humidity_summary) if humidity_summary else "No humidity data available"}

AIR QUALITY INDICATORS:
{chr(10).join(air_quality_summary) if air_quality_summary else "No air quality data available"}

COMFORT VIOLATIONS DETECTED:
{violations_text}

SYSTEM OVERVIEW:
- Total zones analyzed: {len({m.zone_name for m in metrics})}
- Data points: {len(metrics)}
- Equipment types: {len({m.equipment_type for m in metrics})}

{f"OCCUPANCY CONTEXT: {occupancy_context}" if occupancy_context else ""}

Please provide:
1. Overall comfort rating and numeric score (0-100)
2. Assessment of temperature, humidity, and air quality comfort
3. Specific comfort issues identified
4. Practical recommendations for comfort improvement
5. Zone-specific analysis where applicable
6. Educational insights about ASHRAE comfort standards

Base assessment on established comfort standards (ASHRAE 55, 62.1) and
provide actionable recommendations for HVAC professionals."""


class GenericFaultDetectionAgent:
    """
    Generic fault detection agent for HVAC equipment.

    Focus: Common fault patterns and diagnostic principles
    without proprietary fault detection algorithms or models.
    """

    def __init__(self, config: AIAnalysisConfig):
        self.config = config
        self.logger = logger.bind(component="generic_fault_agent")

        self.agent = Agent(
            model=config.model_name,
            result_type=FaultDetectionResult,
            system_prompt=self._build_system_prompt(),
        )

    def _build_system_prompt(self) -> str:
        return """You are an HVAC Diagnostic Specialist with expertise in equipment
fault detection and performance analysis.

Your role is to analyze equipment performance data for signs of faults,
degradation, or maintenance needs using established diagnostic principles.

Common HVAC fault patterns to recognize:
1. TEMPERATURE FAULTS: Sensors out of calibration, heat transfer issues
2. PRESSURE FAULTS: Filter clogs, fan issues, duct problems
3. EFFICIENCY DEGRADATION: Gradual performance decline over time
4. CYCLING ISSUES: Short cycling, failure to start/stop properly
5. CONTROL FAULTS: Setpoint deviations, sensor failures

Diagnostic principles:
- Compare current performance to typical operating ranges
- Look for gradual trends indicating degradation
- Identify sudden changes suggesting equipment failures
- Consider seasonal and load-related performance variations
- Apply basic HVAC troubleshooting logic

Focus on:
- Common equipment fault symptoms
- Standard diagnostic approaches
- General maintenance recommendations
- Performance trend analysis
- Basic troubleshooting guidance

Provide educational insights about HVAC fault detection and practical
recommendations that any qualified technician could follow."""

    async def detect_faults(
        self, hvac_metrics: list[HVACMetric], equipment_history: dict[str, Any] | None = None
    ) -> FaultDetectionResult:
        """Analyze HVAC metrics for potential equipment faults."""

        try:
            # Analyze metrics for fault indicators
            fault_analysis = self._analyze_fault_indicators(hvac_metrics)

            # Build diagnostic prompt
            prompt = self._build_diagnostic_prompt(hvac_metrics, fault_analysis, equipment_history)

            # Run AI analysis
            result = await self.agent.run(prompt)

            self.logger.info(
                "fault_detection_completed",
                faults_detected=len(result.data.faults_detected),
                severity=result.data.fault_severity,
                confidence=result.data.confidence_level,
            )

            return result.data

        except Exception as e:
            self.logger.error("fault_detection_failed", error=str(e))

            # Return basic fallback analysis
            return FaultDetectionResult(
                faults_detected=["Unable to complete fault analysis"],
                fault_severity="Unknown",
                confidence_level=0.1,
                performance_degradation_indicators=[
                    "Analysis incomplete - check system data collection"
                ],
                maintenance_recommendations=[
                    "Schedule professional diagnostic assessment",
                    "Verify all sensors are functioning properly",
                    "Review equipment operating logs",
                ],
                trend_analysis="Unable to assess trends without successful analysis",
                anomaly_indicators=["Insufficient data for anomaly detection"],
                risk_factors=["Unknown due to analysis failure"],
                monitoring_priorities=[
                    "Ensure continuous data collection",
                    "Monitor critical system parameters",
                    "Track equipment runtime and cycles",
                ],
            )

    def _analyze_fault_indicators(self, metrics: list[HVACMetric]) -> dict[str, Any]:
        """Analyze metrics for potential fault indicators."""

        out_of_range_conditions: list[str] = []
        trend_indicators: dict[str, Any] = {}
        performance_anomalies: list[str] = []
        equipment_analysis: dict[str, dict[str, Any]] = {}

        # Group metrics by equipment and type
        equipment_metrics = defaultdict(list)
        for metric in metrics:
            equipment_metrics[metric.source].append(metric)

        # Analyze each piece of equipment
        for equipment_id, equip_metrics in equipment_metrics.items():
            equipment_summary: dict[str, Any] = {
                "out_of_range_count": 0,
                "efficiency_metrics": [],
                "temperature_metrics": [],
                "pressure_metrics": [],
                "power_metrics": [],
            }

            for metric in equip_metrics:
                # Check for out-of-range conditions
                if metric.is_out_of_range is True:
                    part1 = (
                        f"{equipment_id}: {metric.metric_type.value} = "
                        f"{metric.value} {metric.unit} "
                    )
                    part2 = f"(setpoint: {metric.setpoint})"
                    out_of_range_conditions.append(part1 + part2)
                    equipment_summary["out_of_range_count"] += 1

                # Categorize metrics for analysis
                if metric.metric_type in [HVACMetricType.COP, HVACMetricType.EER]:
                    equipment_summary["efficiency_metrics"].append(metric.value)
                elif "temp" in metric.metric_type.value:
                    equipment_summary["temperature_metrics"].append(metric.value)
                elif "pressure" in metric.metric_type.value:
                    equipment_summary["pressure_metrics"].append(metric.value)
                elif metric.metric_type == HVACMetricType.POWER_CONSUMPTION:
                    equipment_summary["power_metrics"].append(metric.value)

            # Analyze efficiency trends
            if len(equipment_summary["efficiency_metrics"]) > 3:
                efficiency_values = equipment_summary["efficiency_metrics"]
                recent_efficiency = mean(efficiency_values[-3:])
                earlier_efficiency = mean(efficiency_values[:3])

                if recent_efficiency < earlier_efficiency * 0.9:  # 10% decline
                    msg = (
                        f"{equipment_id}: Efficiency declining (was {earlier_efficiency:.2f}, "
                        f"now {recent_efficiency:.2f})"
                    )
                    performance_anomalies.append(msg)

            # Temperature spread analysis
            if len(equipment_summary["temperature_metrics"]) > 1:
                temp_range = max(equipment_summary["temperature_metrics"]) - min(
                    equipment_summary["temperature_metrics"]
                )
                if temp_range > 10:  # Large temperature spread might indicate issues
                    performance_anomalies.append(
                        f"{equipment_id}: Large temperature variation ({temp_range:.1f}°F)"
                    )

            equipment_analysis[equipment_id] = equipment_summary

        return {
            "out_of_range_conditions": out_of_range_conditions,
            "trend_indicators": trend_indicators,
            "performance_anomalies": performance_anomalies,
            "equipment_analysis": equipment_analysis,
        }

    def _build_diagnostic_prompt(
        self,
        metrics: list[HVACMetric],
        analysis: dict[str, Any],
        equipment_history: dict[str, Any] | None = None,
    ) -> str:
        """Build fault detection diagnostic prompt."""

        # Out-of-range conditions summary
        out_of_range_text = (
            "\n".join(analysis["out_of_range_conditions"])
            if analysis["out_of_range_conditions"]
            else "None detected"
        )

        # Performance anomalies summary
        anomalies_text = (
            "\n".join(analysis["performance_anomalies"])
            if analysis["performance_anomalies"]
            else "None detected"
        )

        # Equipment summary
        equipment_summary = []
        for equipment_id, equip_data in analysis["equipment_analysis"].items():
            equipment_summary.append(
                f"- {equipment_id}: {equip_data['out_of_range_count']} out-of-range conditions, "
                f"{len(equip_data['efficiency_metrics'])} efficiency readings"
            )

        return f"""Analyze this HVAC system for potential equipment faults and maintenance needs:

OUT-OF-RANGE CONDITIONS DETECTED:
{out_of_range_text}

PERFORMANCE ANOMALIES:
{anomalies_text}

EQUIPMENT SUMMARY:
{chr(10).join(equipment_summary)}

SYSTEM OVERVIEW:
- Total equipment: {len(analysis["equipment_analysis"])}
- Total data points: {len(metrics)}
- Analysis period: Recent operational data
- Equipment types: {", ".join({m.equipment_type.value for m in metrics})}

{f"EQUIPMENT HISTORY: {equipment_history}" if equipment_history else "No history provided"}

Please provide:
1. Potential faults detected with severity assessment
2. Performance degradation indicators and their significance
3. Maintenance recommendations based on findings
4. Equipment performance trend analysis
5. Specific anomaly indicators requiring attention
6. Risk factors that could lead to equipment failure
7. Key metrics to monitor for early fault detection

Apply standard HVAC diagnostic principles and provide actionable
recommendations that qualified technicians can implement."""


# Integration service for coordinating all generic agents
class GenericHVACAIService:
    """
    Coordinates generic HVAC AI agents for comprehensive system analysis.

    Demonstrates multi-agent architecture without revealing proprietary
    coordination strategies or customer-specific optimization logic.
    """

    def __init__(self, config: AIAnalysisConfig):
        self.config = config
        self.energy_agent = GenericEnergyOptimizationAgent(config)
        self.comfort_agent = GenericComfortAssessmentAgent(config)
        self.fault_agent = GenericFaultDetectionAgent(config)
        self.logger = logger.bind(component="generic_hvac_ai_service")

    async def comprehensive_analysis(
        self, hvac_metrics: list[HVACMetric], system_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run comprehensive analysis using all generic agents."""

        analysis_start = datetime.now(UTC)
        results: dict[str, Any] = {}

        try:
            self.logger.info("comprehensive_analysis_starting", metrics_count=len(hvac_metrics))

            # Run all agents concurrently for efficiency
            energy_task = asyncio.create_task(
                self.energy_agent.analyze_energy_efficiency(hvac_metrics, system_context)
            )

            comfort_task = asyncio.create_task(
                self.comfort_agent.assess_comfort(hvac_metrics, system_context)
            )

            fault_task = asyncio.create_task(
                self.fault_agent.detect_faults(hvac_metrics, system_context)
            )

            # Wait for all analyses to complete
            energy_result, comfort_result, fault_result = await asyncio.gather(
                energy_task, comfort_task, fault_task, return_exceptions=True
            )

            # Process results
            results["energy_optimization"] = (
                energy_result if not isinstance(energy_result, Exception) else None
            )
            results["comfort_assessment"] = (
                comfort_result if not isinstance(comfort_result, Exception) else None
            )
            results["fault_detection"] = (
                fault_result if not isinstance(fault_result, Exception) else None
            )

            # Log any exceptions
            for analysis_type, result in [
                ("energy", energy_result),
                ("comfort", comfort_result),
                ("fault", fault_result),
            ]:
                if isinstance(result, Exception):
                    self.logger.error(f"{analysis_type}_analysis_exception", error=str(result))

            analysis_duration = (datetime.now(UTC) - analysis_start).total_seconds()

            results["analysis_metadata"] = {
                "analysis_duration_seconds": analysis_duration,
                "metrics_analyzed": len(hvac_metrics),
                "agents_used": ["energy_optimization", "comfort_assessment", "fault_detection"],
                "analysis_timestamp": analysis_start,
                "success_count": sum(
                    1
                    for r in [energy_result, comfort_result, fault_result]
                    if not isinstance(r, Exception)
                ),
            }

            self.logger.info(
                "comprehensive_analysis_completed",
                duration_seconds=round(analysis_duration, 3),
                successful_analyses=results["analysis_metadata"]["success_count"],
            )

            return results

        except Exception as e:
            self.logger.error("comprehensive_analysis_failed", error=str(e))

            return {
                "error": f"Comprehensive analysis failed: {str(e)}",
                "analysis_metadata": {
                    "analysis_duration_seconds": (
                        datetime.now(UTC) - analysis_start
                    ).total_seconds(),
                    "metrics_analyzed": len(hvac_metrics),
                    "analysis_timestamp": analysis_start,
                    "success_count": 0,
                },
            }

    def get_agent_capabilities(self) -> dict[str, Any]:
        """Get information about available agent capabilities."""

        return {
            "energy_optimization": {
                "description": (
                    "Analyzes HVAC energy efficiency and provides optimization recommendations"
                ),
                "focus_areas": [
                    "Equipment efficiency assessment",
                    "Setpoint optimization within comfort ranges",
                    "Operating schedule improvements",
                    "Control strategy enhancements",
                ],
                "output_type": "EnergyOptimizationSuggestion",
                "standards_applied": ["ASHRAE standards", "Industry best practices"],
            },
            "comfort_assessment": {
                "description": (
                    "Evaluates occupant comfort based on temperature, humidity, and air quality"
                ),
                "focus_areas": [
                    "ASHRAE 55 thermal comfort compliance",
                    "ASHRAE 62.1 ventilation adequacy",
                    "Zone-level comfort analysis",
                    "Comfort improvement recommendations",
                ],
                "output_type": "ComfortAssessmentResult",
                "standards_applied": ["ASHRAE 55", "ASHRAE 62.1"],
            },
            "fault_detection": {
                "description": "Identifies potential equipment faults and maintenance needs",
                "focus_areas": [
                    "Equipment performance degradation",
                    "Out-of-range condition analysis",
                    "Maintenance recommendation generation",
                    "Performance trend assessment",
                ],
                "output_type": "FaultDetectionResult",
                "standards_applied": [
                    "Standard diagnostic principles",
                    "HVAC troubleshooting practices",
                ],
            },
            "integration_features": {
                "concurrent_analysis": "All agents run simultaneously for efficiency",
                "error_handling": "Graceful degradation when individual agents fail",
                "comprehensive_reporting": "Coordinated results from all agents",
                "metadata_tracking": "Performance and success metrics",
            },
        }


# Example usage and testing
async def main() -> None:
    """Demonstrate the generic HVAC AI agents."""

    print("🤖 Generic HVAC AI Agents Demo")
    print("=" * 50)

    # Initialize AI service
    try:
        from core.config import get_config

        config_obj = get_config()
        ai_config = AIAnalysisConfig(
            model_name=config_obj.ai_provider.anomaly_detection_model,
            temperature=0.1,
            timeout_seconds=30.0,
        )

        hvac_ai_service = GenericHVACAIService(ai_config)

    except Exception as e:
        print(f"⚠️  AI service initialization failed: {e}")
        print("   This demo requires OpenAI API key configuration")
        return

    # Create test HVAC metrics with various conditions
    print("\n📊 Creating Test HVAC Metrics")

    # Normal operation metrics
    normal_metrics = [
        HVACMetric(
            metric_type=HVACMetricType.SUPPLY_AIR_TEMP,
            value=55.2,
            unit="degrees_f",
            source="AHU-01",
            equipment_type=HVACSystemType.AIR_HANDLER,
            zone_name="Office-Floor-1",
            setpoint=55.0,
        ),
        HVACMetric(
            metric_type=HVACMetricType.RETURN_AIR_TEMP,
            value=74.8,
            unit="degrees_f",
            source="AHU-01",
            equipment_type=HVACSystemType.AIR_HANDLER,
            zone_name="Office-Floor-1",
            setpoint=None,
        ),
        HVACMetric(
            metric_type=HVACMetricType.POWER_CONSUMPTION,
            value=45.2,
            unit="kw",
            source="Chiller-01",
            equipment_type=HVACSystemType.CHILLER,
            zone_name="Main-Building",
            setpoint=None,
        ),
        HVACMetric(
            metric_type=HVACMetricType.COP,
            value=4.1,
            unit="ratio",
            source="Chiller-01",
            equipment_type=HVACSystemType.CHILLER,
            zone_name="Main-Building",
            setpoint=None,
        ),
    ]

    # Problematic metrics for testing
    problem_metrics = [
        HVACMetric(
            metric_type=HVACMetricType.SUPPLY_AIR_TEMP,
            value=62.5,  # Too high
            unit="degrees_f",
            source="AHU-02",
            equipment_type=HVACSystemType.AIR_HANDLER,
            zone_name="Office-Floor-2",
            setpoint=55.0,
        ),
        HVACMetric(
            metric_type=HVACMetricType.HUMIDITY,
            value=75.0,  # Too high
            unit="percent_rh",
            source="AHU-02",
            equipment_type=HVACSystemType.AIR_HANDLER,
            zone_name="Office-Floor-2",
            setpoint=50.0,
        ),
        HVACMetric(
            metric_type=HVACMetricType.COP,
            value=2.1,  # Poor efficiency
            unit="ratio",
            source="Chiller-02",
            equipment_type=HVACSystemType.CHILLER,
            zone_name="Main-Building",
            setpoint=None,
        ),
        HVACMetric(
            metric_type=HVACMetricType.CO2_LEVEL,
            value=1250,  # High CO2
            unit="ppm",
            source="Zone-Sensor-1",
            equipment_type=HVACSystemType.AIR_HANDLER,
            zone_name="Conference-Room",
            setpoint=None,
        ),
    ]

    all_test_metrics = normal_metrics + problem_metrics

    print(f"   Created {len(all_test_metrics)} test metrics")
    print(f"   Normal conditions: {len(normal_metrics)} metrics")
    print(f"   Problem conditions: {len(problem_metrics)} metrics")

    # System context
    system_context = {
        "building_type": "office",
        "total_square_feet": 50000,
        "occupancy_schedule": "business_hours",
        "climate_zone": "mixed_humid",
    }

    # Run comprehensive analysis
    print("\n🔍 Running Comprehensive AI Analysis")
    print("   This may take 30-60 seconds...")

    try:
        analysis_results = await hvac_ai_service.comprehensive_analysis(
            all_test_metrics, system_context
        )

        # Display results
        print("\n📈 ANALYSIS RESULTS")
        print("=" * 30)

        metadata = analysis_results.get("analysis_metadata", {})
        print(f"Duration: {metadata.get('analysis_duration_seconds', 0):.2f}s")
        print(f"Successful Analyses: {metadata.get('success_count', 0)}/3")

        # Energy Optimization Results
        if "energy_optimization" in analysis_results and analysis_results["energy_optimization"]:
            energy = analysis_results["energy_optimization"]
            print("\n⚡ ENERGY OPTIMIZATION")
            print(f"   Current Efficiency: {energy.current_efficiency_level}")
            print(f"   Improvement Potential: {energy.improvement_potential}")
            print(f"   Confidence: {energy.confidence:.1%}")
            print("   Key Recommendations:")
            for rec in energy.immediate_actions[:2]:
                print(f"     • {rec}")

        # Comfort Assessment Results
        if "comfort_assessment" in analysis_results and analysis_results["comfort_assessment"]:
            comfort = analysis_results["comfort_assessment"]
            print("\n🌡️ COMFORT ASSESSMENT")
            print(f"   Overall Rating: {comfort.overall_comfort_rating}")
            print(f"   Comfort Score: {comfort.comfort_score:.1f}/100")
            print(f"   Temperature: {comfort.temperature_assessment}")
            print(f"   Issues Found: {len(comfort.comfort_issues_identified)}")
            if comfort.comfort_issues_identified:
                print(f"     • {comfort.comfort_issues_identified[0]}")

        # Fault Detection Results
        if "fault_detection" in analysis_results and analysis_results["fault_detection"]:
            faults = analysis_results["fault_detection"]
            print("\n🔧 FAULT DETECTION")
            print(f"   Faults Detected: {len(faults.faults_detected)}")
            print(f"   Severity: {faults.fault_severity}")
            print(f"   Confidence: {faults.confidence_level:.1%}")
            if faults.faults_detected:
                print(f"   Primary Fault: {faults.faults_detected[0]}")

        # Agent Capabilities
        print("\n🛠️ AGENT CAPABILITIES")
        capabilities = hvac_ai_service.get_agent_capabilities()
        for agent_name, cap_info in capabilities.items():
            if agent_name != "integration_features":
                print(f"   {agent_name.replace('_', ' ').title()}:")
                print(f"     • {cap_info['description']}")

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        print("   Check API key configuration and network connectivity")


if __name__ == "__main__":
    asyncio.run(main())
