"""
HVAC-specific domain models extending the core monitoring framework.

This demonstrates domain-driven design:
- HVAC-specific metrics and calculations
- Energy efficiency models
- Maintenance prediction logic
- Building automation concepts

Key HVAC Concepts:
- COP (Coefficient of Performance): Energy efficiency ratio
- EER (Energy Efficiency Ratio): Cooling efficiency
- Static Pressure: Air handling system resistance
- Setpoint Deviation: How far from target temperature
"""

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field, validator

# Extend core domain models
from core.domain.models import Severity


class HVACSystemType(str, Enum):
    """Types of HVAC equipment we can monitor."""

    CHILLER = "chiller"
    BOILER = "boiler"
    AIR_HANDLER = "air_handler"
    VAV_BOX = "vav_box"  # Variable Air Volume
    FAN_COIL = "fan_coil"
    HEAT_PUMP = "heat_pump"
    ROOFTOP_UNIT = "rooftop_unit"
    COOLING_TOWER = "cooling_tower"
    EXHAUST_FAN = "exhaust_fan"


class HVACMetricType(str, Enum):
    """HVAC-specific metrics extending the base metrics."""

    # Temperature measurements
    SUPPLY_AIR_TEMP = "supply_air_temp"
    RETURN_AIR_TEMP = "return_air_temp"
    OUTDOOR_AIR_TEMP = "outdoor_air_temp"
    CHILLED_WATER_TEMP = "chilled_water_temp"
    HOT_WATER_TEMP = "hot_water_temp"

    # Pressure measurements
    STATIC_PRESSURE = "static_pressure"
    REFRIGERANT_PRESSURE = "refrigerant_pressure"

    # Flow measurements
    AIR_FLOW_RATE = "air_flow_rate"
    WATER_FLOW_RATE = "water_flow_rate"

    # Energy and efficiency
    POWER_CONSUMPTION = "power_consumption"
    COP = "coefficient_of_performance"  # Energy efficiency
    EER = "energy_efficiency_ratio"

    # Control and operation
    DAMPER_POSITION = "damper_position"
    VALVE_POSITION = "valve_position"
    FAN_SPEED = "fan_speed"
    SETPOINT_DEVIATION = "setpoint_deviation"

    # Air quality
    CO2_LEVEL = "co2_level"
    HUMIDITY = "humidity"

    # Status indicators
    RUNTIME_HOURS = "runtime_hours"
    CYCLE_COUNT = "cycle_count"


class HVACMetric(BaseModel):
    """
    HVAC-specific metric extending SystemMetric with domain knowledge.

    This adds HVAC-specific validation and computed properties.
    """

    # Core metric data (same as SystemMetric)
    metric_type: HVACMetricType
    value: float
    unit: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str
    tags: dict[str, str] = Field(default_factory=dict)

    # HVAC-specific metadata
    equipment_type: HVACSystemType
    zone_name: str = Field(description="Building zone or area served")
    setpoint: float | None = Field(None, description="Target value for controlled metrics")

    @validator("value")
    def validate_hvac_ranges(cls, v: float, values: dict) -> float:
        """Validate HVAC metrics are in realistic ranges."""
        metric_type = values.get("metric_type")

        # Temperature range validation (reasonable HVAC ranges)
        temp_metrics = [
            HVACMetricType.SUPPLY_AIR_TEMP,
            HVACMetricType.RETURN_AIR_TEMP,
            HVACMetricType.OUTDOOR_AIR_TEMP,
        ]

        if metric_type in temp_metrics:
            if v < -40 or v > 150:  # Fahrenheit range
                raise ValueError(f"Temperature {v} outside realistic range (-40°F to 150°F)")

        # Humidity validation
        if metric_type == HVACMetricType.HUMIDITY:
            if v < 0 or v > 100:
                raise ValueError(f"Humidity {v}% must be 0-100%")

        # Pressure validation (positive values only)
        if isinstance(metric_type, HVACMetricType) and "pressure" in metric_type.value:
            if v < 0:
                raise ValueError(f"Pressure {v} cannot be negative")

        return v

    @computed_field(return_type=bool)
    def is_out_of_range(self) -> bool:
        """Check if metric is significantly outside normal operating range."""
        if not self.setpoint:
            return False

        deviation = abs(self.value - self.setpoint)

        # Temperature tolerance (±3°F typical)
        if "temp" in self.metric_type.value:
            return deviation > 3.0

        # Humidity tolerance (±10% typical)
        if self.metric_type == HVACMetricType.HUMIDITY:
            return deviation > 10.0

        # Pressure tolerance (±20% typical)
        if "pressure" in self.metric_type.value and self.setpoint > 0:
            return (deviation / self.setpoint) > 0.20

        return False

    @computed_field(return_type=str | None)
    def energy_efficiency_rating(self) -> str | None:
        """Rate energy efficiency for applicable metrics."""
        if self.metric_type == HVACMetricType.COP:
            # Coefficient of Performance ratings
            if self.value >= 4.0:
                return "excellent"
            elif self.value >= 3.0:
                return "good"
            elif self.value >= 2.0:
                return "fair"
            else:
                return "poor"

        elif self.metric_type == HVACMetricType.EER:
            # Energy Efficiency Ratio ratings
            if self.value >= 12.0:
                return "excellent"
            elif self.value >= 10.0:
                return "good"
            elif self.value >= 8.0:
                return "fair"
            else:
                return "poor"

        return None


class EnergyAnalysis(BaseModel):
    """Analysis of energy consumption and efficiency patterns."""

    total_power_kw: float = Field(ge=0.0)
    average_cop: float | None = Field(None, ge=0.0)
    average_eer: float | None = Field(None, ge=0.0)

    # Cost analysis
    estimated_hourly_cost: float = Field(ge=0.0, description="USD per hour")
    potential_savings_percent: float = Field(ge=0.0, le=100.0)

    # Efficiency trends
    efficiency_trend: Literal["improving", "stable", "degrading"]
    baseline_comparison_percent: float = Field(description="% change from baseline efficiency")

    # Recommendations
    efficiency_recommendations: list[str] = Field(max_length=5)

    @computed_field(return_type=float)
    def annual_cost_estimate(self) -> float:
        """Estimate annual energy cost based on current consumption."""
        return self.estimated_hourly_cost * 24 * 365


class MaintenancePrediction(BaseModel):
    """Predictive maintenance analysis for HVAC equipment."""

    equipment_id: str
    equipment_type: HVACSystemType

    # Health indicators
    overall_health_score: float = Field(ge=0.0, le=100.0)
    wear_indicators: dict[str, float] = Field(description="Component wear scores (0-100)")

    # Predictions
    estimated_remaining_life_days: int = Field(ge=0)
    maintenance_urgency: Severity
    recommended_actions: list[str]

    # Supporting data
    runtime_hours_total: float = Field(ge=0.0)
    cycle_count_total: int = Field(ge=0)
    last_maintenance_date: datetime | None = None

    @computed_field(return_type=datetime)
    def maintenance_due_date(self) -> datetime:
        """Calculate when maintenance is due."""
        return datetime.now(UTC) + timedelta(days=self.estimated_remaining_life_days)

    @computed_field(return_type=bool)
    def is_maintenance_overdue(self) -> bool:
        """Check if maintenance is overdue."""
        return self.estimated_remaining_life_days <= 0


class HVACSystemStatus(BaseModel):
    """Overall status of an HVAC system with AI analysis."""

    system_id: str
    system_type: HVACSystemType
    zone_name: str

    # Current metrics summary
    current_metrics: list[HVACMetric]
    last_update: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # AI analysis results
    operational_status: Literal["optimal", "suboptimal", "attention_needed", "critical"]
    comfort_level: Literal["excellent", "good", "fair", "poor"]
    energy_efficiency: Literal["excellent", "good", "fair", "poor"]

    # Detailed analysis
    energy_analysis: EnergyAnalysis | None = None
    maintenance_prediction: MaintenancePrediction | None = None

    # Issues and recommendations
    active_alarms: list[str] = Field(default_factory=list)
    optimization_opportunities: list[str] = Field(default_factory=list)

    @computed_field(return_type=bool)
    def requires_immediate_attention(self) -> bool:
        """Check if system requires immediate attention."""
        return (
            self.operational_status == "critical"
            or any("critical" in alarm.lower() for alarm in self.active_alarms)
            or (
                self.maintenance_prediction is not None
                and self.maintenance_prediction.maintenance_urgency == Severity.CRITICAL
            )
        )


# HVAC-specific calculations and utilities
class HVACCalculations:
    """Utility class for HVAC-specific calculations."""

    @staticmethod
    def calculate_cop(cooling_output_btuh: float, power_input_watts: float) -> float:
        """
        Calculate Coefficient of Performance.

        COP = Cooling Output (BTU/h) ÷ Power Input (Watts × 3.412)
        Higher is better (typical range 2.0-6.0)
        """
        if power_input_watts <= 0:
            return 0.0

        power_input_btuh = power_input_watts * 3.412  # Convert watts to BTU/h
        return cooling_output_btuh / power_input_btuh

    @staticmethod
    def calculate_eer(cooling_output_btuh: float, power_input_watts: float) -> float:
        """
        Calculate Energy Efficiency Ratio.

        EER = Cooling Output (BTU/h) ÷ Power Input (Watts)
        Higher is better (typical range 8-15)
        """
        if power_input_watts <= 0:
            return 0.0

        return cooling_output_btuh / power_input_watts

    @staticmethod
    def calculate_air_flow_cfm(velocity_fpm: float, duct_area_sqft: float) -> float:
        """
        Calculate air flow in CFM (Cubic Feet per Minute).

        CFM = Velocity (FPM) × Duct Area (sq ft)
        """
        return velocity_fpm * duct_area_sqft

    @staticmethod
    def estimate_energy_cost(
        power_kw: float, runtime_hours: float, rate_per_kwh: float = 0.12
    ) -> float:
        """
        Estimate energy cost for given usage.

        Default rate: $0.12/kWh (US commercial average)
        """
        kwh_consumed = power_kw * runtime_hours
        return kwh_consumed * rate_per_kwh

    @staticmethod
    def comfort_index(
        temperature_f: float, humidity_percent: float, setpoint_f: float = 72.0
    ) -> tuple[float, str]:
        """
        Calculate comfort index based on temperature and humidity.

        Returns (score 0-100, description)
        Based on ASHRAE comfort standards.
        """
        # Temperature comfort (optimal 68-76°F)
        temp_deviation = abs(temperature_f - setpoint_f)
        temp_score = max(0, 100 - (temp_deviation * 10))

        # Humidity comfort (optimal 30-60%)
        if 30 <= humidity_percent <= 60:
            humidity_score = 100.0
        elif humidity_percent < 30:
            humidity_score = max(0, 100 - ((30 - humidity_percent) * 3))
        else:  # > 60%
            humidity_score = max(0, 100 - ((humidity_percent - 60) * 2))

        # Combined score (weighted average)
        combined_score = (temp_score * 0.7) + (humidity_score * 0.3)

        # Description
        if combined_score >= 85:
            description = "excellent"
        elif combined_score >= 70:
            description = "good"
        elif combined_score >= 50:
            description = "fair"
        else:
            description = "poor"

        return combined_score, description


# Factory functions for creating HVAC metrics
def create_chiller_metrics(
    chiller_id: str, zone_name: str, chilled_water_temp: float, power_kw: float, cooling_tons: float
) -> list[HVACMetric]:
    """Create typical chiller metrics."""

    now = datetime.now(UTC)

    # Calculate efficiency
    cooling_btuh = cooling_tons * 12000  # 1 ton = 12,000 BTU/h
    power_watts = power_kw * 1000
    cop = HVACCalculations.calculate_cop(cooling_btuh, power_watts)

    return [
        HVACMetric(
            metric_type=HVACMetricType.CHILLED_WATER_TEMP,
            value=chilled_water_temp,
            unit="degrees_f",
            timestamp=now,
            source=chiller_id,
            equipment_type=HVACSystemType.CHILLER,
            zone_name=zone_name,
            setpoint=42.0,  # Typical chilled water setpoint
            tags={"equipment_type": "centrifugal_chiller", "refrigerant": "R134a"},
        ),
        HVACMetric(
            metric_type=HVACMetricType.POWER_CONSUMPTION,
            value=power_kw,
            unit="kw",
            timestamp=now,
            source=chiller_id,
            equipment_type=HVACSystemType.CHILLER,
            zone_name=zone_name,
            setpoint=None,
            tags={"load": f"{cooling_tons}_tons"},
        ),
        HVACMetric(
            metric_type=HVACMetricType.COP,
            value=cop,
            unit="ratio",
            timestamp=now,
            source=chiller_id,
            equipment_type=HVACSystemType.CHILLER,
            zone_name=zone_name,
            setpoint=None,
            tags={"efficiency_rating": "variable"},
        ),
    ]


def create_air_handler_metrics(
    ahu_id: str,
    zone_name: str,
    supply_temp: float,
    return_temp: float,
    static_pressure: float,
    fan_speed_percent: float,
) -> list[HVACMetric]:
    """Create typical air handling unit metrics."""

    now = datetime.now(UTC)

    return [
        HVACMetric(
            metric_type=HVACMetricType.SUPPLY_AIR_TEMP,
            value=supply_temp,
            unit="degrees_f",
            timestamp=now,
            source=ahu_id,
            equipment_type=HVACSystemType.AIR_HANDLER,
            zone_name=zone_name,
            setpoint=55.0,  # Typical supply air setpoint
            tags={"control_mode": "vav", "economizer": "enabled"},
        ),
        HVACMetric(
            metric_type=HVACMetricType.RETURN_AIR_TEMP,
            value=return_temp,
            unit="degrees_f",
            timestamp=now,
            source=ahu_id,
            equipment_type=HVACSystemType.AIR_HANDLER,
            zone_name=zone_name,
            setpoint=None,
            tags={"sensor_location": "return_duct"},
        ),
        HVACMetric(
            metric_type=HVACMetricType.STATIC_PRESSURE,
            value=static_pressure,
            unit="inches_wc",  # Inches of water column
            timestamp=now,
            source=ahu_id,
            equipment_type=HVACSystemType.AIR_HANDLER,
            zone_name=zone_name,
            setpoint=2.0,  # Typical static pressure setpoint
            tags={"measurement_point": "supply_duct"},
        ),
        HVACMetric(
            metric_type=HVACMetricType.FAN_SPEED,
            value=fan_speed_percent,
            unit="percent",
            timestamp=now,
            source=ahu_id,
            equipment_type=HVACSystemType.AIR_HANDLER,
            zone_name=zone_name,
            setpoint=None,
            tags={"vfd_controlled": "true", "motor_hp": "25"},
        ),
    ]


# Example usage and testing
if __name__ == "__main__":
    # Test HVAC metric creation and validation
    print("🏭 Testing HVAC Domain Models")

    # Create chiller metrics
    chiller_metrics = create_chiller_metrics(
        chiller_id="CH-01",
        zone_name="Main Building",
        chilled_water_temp=42.5,
        power_kw=85.2,
        cooling_tons=300,
    )

    print(f"\n❄️  CHILLER METRICS ({len(chiller_metrics)} metrics)")
    for metric in chiller_metrics:
        print(f"  {metric.metric_type.value}: {metric.value} {metric.unit}")
        rating = metric.energy_efficiency_rating
        if rating is not None:
            print(f"    Efficiency: {rating}")
        if metric.is_out_of_range is True:
            print(f"    ⚠️  OUT OF RANGE (setpoint: {metric.setpoint})")

    # Create air handler metrics
    ahu_metrics = create_air_handler_metrics(
        ahu_id="AHU-01",
        zone_name="Office Floor 1",
        supply_temp=54.2,
        return_temp=75.8,
        static_pressure=2.1,
        fan_speed_percent=67.5,
    )

    print(f"\n🌬️  AIR HANDLER METRICS ({len(ahu_metrics)} metrics)")
    for metric in ahu_metrics:
        print(f"  {metric.metric_type.value}: {metric.value} {metric.unit}")
        if metric.is_out_of_range is True:
            print(f"    ⚠️  OUT OF RANGE (setpoint: {metric.setpoint})")

    # Test calculations
    print("\n🧮 HVAC CALCULATIONS")
    cop = HVACCalculations.calculate_cop(3600000, 85200)  # 300 tons, 85.2kW
    eer = HVACCalculations.calculate_eer(3600000, 85200)
    comfort_score, comfort_desc = HVACCalculations.comfort_index(74.5, 45.0)

    print(f"  COP: {cop:.2f}")
    print(f"  EER: {eer:.1f}")
    print(f"  Comfort: {comfort_score:.1f}% ({comfort_desc})")
    print(f"  Daily energy cost: ${HVACCalculations.estimate_energy_cost(85.2, 24):.2f}")
