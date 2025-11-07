# Model-Agnostic AI Architecture

This document explains the model-agnostic architecture implemented in the Intelligent Health Monitor system, allowing seamless switching between AI providers (OpenAI, Anthropic, Google Gemini, etc.) without code changes.

## Architecture Overview

The system uses **Pydantic AI** as the foundation, which provides a unified interface across multiple LLM providers. The key innovation is the **Verbal Confidence Pattern**, which enables provider-agnostic confidence scoring.

### Key Components

1. **Pydantic AI Foundation**: Unified interface for multiple providers
2. **Verbal Confidence Pattern**: LLM self-assessment of certainty
3. **Provider-Agnostic Configuration**: Single source of truth for model selection
4. **Structured Output Validation**: Type-safe AI responses via Pydantic models
5. **Automatic Retry Logic**: Built-in validation and retry mechanisms

## Verbal Confidence Pattern

### What is it?

The Verbal Confidence Pattern is a provider-agnostic approach to confidence scoring where the LLM explicitly self-assesses its certainty and populates a `confidence_score` field (0.0-1.0) in the structured output.

### Why Use It?

**Traditional Approach (Provider-Specific)**:
```python
# ❌ Tightly coupled to OpenAI's API
response = openai.chat.completions.create(...)
confidence = response.choices[0].logprobs.top_logprobs[0]  # OpenAI-specific
```

**Verbal Confidence Approach (Provider-Agnostic)**:
```python
# ✅ Works with any provider
class AnomalyDetection(BaseModel):
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Self-assessed confidence (0.0-1.0)..."
    )

# LLM populates this field based on explicit instructions
```

### How It Works

1. **Pydantic Model Definition**: Include `confidence_score` field with detailed description
2. **System Prompt Instructions**: Explicitly instruct the LLM to self-assess
3. **Validation**: Pydantic ensures the field is populated correctly
4. **Retry Logic**: Auto-retry if validation fails

### Implementation Example

```python
from pydantic import BaseModel, Field

class AnomalyDetection(BaseModel):
    """AI analysis with verbal confidence pattern."""

    severity: Severity
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Self-assessed confidence level (0.0-1.0). "
            "Consider: data quality, pattern clarity, "
            "correlation strength, historical context."
        ),
    )
    # ... other fields
```

**System Prompt**:
```python
system_prompt = """
CRITICAL - Confidence Score Assessment:
You MUST self-assess your confidence and populate the confidence_score field (0.0-1.0).

Consider these factors:
- Data Quality: Sufficient data? Gaps or inconsistencies?
- Pattern Clarity: Clear and unambiguous patterns?
- Correlation Strength: Multiple metrics support conclusion?
- Historical Context: Baseline data available?

Confidence Guidelines:
- 0.9-1.0: Very high confidence - clear patterns, strong evidence
- 0.7-0.9: High confidence - solid evidence, minor uncertainties
- 0.5-0.7: Moderate confidence - some evidence, notable uncertainties
- Below 0.5: Low confidence - weak evidence, significant gaps
"""
```

## Switching Between Providers

### Step 1: Install Provider Dependencies

The system uses `pydantic-ai-slim` with provider-specific extras:

```bash
# For OpenAI
uv add "pydantic-ai-slim[openai]"

# For Anthropic
uv add "pydantic-ai-slim[anthropic]"

# For Google Gemini
uv add "pydantic-ai-slim[google]"

# For multiple providers
uv add "pydantic-ai-slim[openai,anthropic,google]"
```

**Current `pyproject.toml`**:
```toml
dependencies = [
    "pydantic-ai-slim[anthropic,evals,logfire,openai,google]>=0.8.1",
    # ... other dependencies
]
```

### Step 2: Set API Keys

Set the appropriate environment variable for your chosen provider:

```bash
# OpenAI
export OPENAI_AI_API_KEY=sk-your-key-here

# Anthropic
export ANTHROPIC_AI_API_KEY=sk-ant-your-key-here

# Google Gemini
export GEMINI_AI_API_KEY=your-key-here
```

### Step 3: Configure Model Name

Update your configuration to use the desired model:

```python
from core.services.ai_analysis import AIAnalysisConfig

# OpenAI
config = AIAnalysisConfig(
    model_name="openai:gpt-4o-mini",
    temperature=0.1,
    max_retries=3,
)

# Anthropic
config = AIAnalysisConfig(
    model_name="anthropic:claude-3-5-sonnet-20241022",
    temperature=0.1,
    max_retries=3,
)

# Google Gemini
config = AIAnalysisConfig(
    model_name="gemini-2.5-flash-lite",
    temperature=0.1,
    max_retries=3,
)
```

**That's it!** No code changes needed beyond configuration.

## Model Name Format

### OpenAI Models
- Format: `openai:model-name`
- Examples:
  - `openai:gpt-4o-mini` (fast, cost-effective)
  - `openai:gpt-4o` (most capable)
  - `openai:gpt-4-turbo`

### Anthropic Models
- Format: `anthropic:model-name`
- Examples:
  - `anthropic:claude-3-5-sonnet-20241022` (balanced)
  - `anthropic:claude-3-5-haiku-20241022` (fast)
  - `anthropic:claude-3-opus-20240229` (most capable)

### Google Gemini Models
- Format: `gemini-model-name` (no prefix)
- Examples:
  - `gemini-2.5-flash-lite` (fastest)
  - `gemini-2.5-flash` (fast)
  - `gemini-2.5-pro` (balanced)
  - `gemini-1.5-pro` (most capable)

## Configuration Management

### Centralized Configuration

All AI configuration is centralized in `AIAnalysisConfig`:

```python
class AIAnalysisConfig(BaseModel):
    """Provider-agnostic AI configuration."""

    model_name: str  # Required - no defaults
    max_tokens: int = Field(default=1000, gt=100)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, description="Auto-retry on validation failures")

    # Analysis-specific
    anomaly_threshold: float = Field(
        default=0.7,
        description="Minimum confidence_score threshold"
    )
```

### Environment-Based Configuration

Load configuration from environment variables:

```python
from core.config import get_config

config = get_config()
ai_config = AIAnalysisConfig(
    model_name=config.ai_provider.anomaly_detection_model,
    temperature=config.ai_provider.default_temperature,
    max_retries=config.ai_provider.default_max_retries,
)
```

## Automatic Retry Logic

Pydantic AI includes built-in retry logic for validation failures:

```python
config = AIAnalysisConfig(
    model_name="gemini-2.5-flash-lite",
    max_retries=3,  # Retry up to 3 times on validation failures
)
```

**How it works**:
1. LLM generates response
2. Pydantic validates the structured output
3. If validation fails (e.g., missing `confidence_score`):
   - System automatically re-prompts with validation error
   - LLM corrects the issue
   - Process repeats up to `max_retries` times

## Benefits of This Architecture

### 1. True Provider Agnosticism
- No provider-specific code in business logic
- Switch providers by changing configuration only
- Works identically across all providers

### 2. Type Safety
- All AI outputs validated with Pydantic
- Compile-time type checking with mypy
- Runtime validation ensures data integrity

### 3. Robustness
- Automatic retry on validation failures
- Graceful degradation on errors
- Comprehensive error logging

### 4. Maintainability
- Single source of truth for configuration
- Clear separation of concerns
- Easy to test and debug

### 5. Cost Optimization
- Mix and match models by task
- Use cheaper models for frequent operations
- Use powerful models for complex analysis

## Example: Multi-Provider Setup

```python
# Different models for different tasks
anomaly_config = AIAnalysisConfig(
    model_name="gemini-2.5-flash-lite",  # Fast, cheap for frequent checks
    temperature=0.1,
    anomaly_threshold=0.75,
)

root_cause_config = AIAnalysisConfig(
    model_name="anthropic:claude-3-5-sonnet-20241022",  # Powerful for deep analysis
    temperature=0.2,
    max_tokens=2000,
)

# Both use the same verbal confidence pattern!
```

## Testing Across Providers

The system includes tests that work across all providers:

```python
import pytest
from core.services.ai_analysis import AIAnalysisConfig, AnomalyDetectionAgent

@pytest.mark.parametrize("model_name", [
    "openai:gpt-4o-mini",
    "anthropic:claude-3-5-haiku-20241022",
    "gemini-2.5-flash-lite",
])
async def test_anomaly_detection_provider_agnostic(model_name):
    config = AIAnalysisConfig(model_name=model_name)
    agent = AnomalyDetectionAgent(config)

    result = await agent.analyze_metrics(test_metrics, test_context)

    # Verbal confidence pattern works across all providers
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.severity in Severity
```

## Migration Guide

### From OpenAI-Specific Code

**Before**:
```python
import openai

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
)
# Provider-specific parsing
```

**After**:
```python
from pydantic_ai import Agent
from core.services.ai_analysis import AIAnalysisConfig

config = AIAnalysisConfig(model_name="openai:gpt-4o-mini")
agent = Agent(
    model=config.model_name,
    output_type=AnomalyDetection,
    system_prompt=system_prompt,
)
result = await agent.run(user_prompt)
# Works with any provider!
```

### From Anthropic-Specific Code

**Before**:
```python
import anthropic

client = anthropic.Anthropic(api_key=...)
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[...],
)
# Provider-specific parsing
```

**After**:
```python
config = AIAnalysisConfig(model_name="anthropic:claude-3-5-sonnet-20241022")
agent = Agent(
    model=config.model_name,
    output_type=AnomalyDetection,
    system_prompt=system_prompt,
)
result = await agent.run(user_prompt)
# Same code, different provider!
```

## Best Practices

### 1. Always Use Verbal Confidence
- Include `confidence_score` in all AI output models
- Provide clear guidelines in system prompts
- Validate confidence thresholds in business logic

### 2. Centralize Configuration
- Use `AIAnalysisConfig` for all AI settings
- Load from environment variables in production
- Document model selection rationale

### 3. Implement Retry Logic
- Set `max_retries >= 3` for production
- Log retry attempts for monitoring
- Handle final failures gracefully

### 4. Test Across Providers
- Parametrize tests with multiple providers
- Verify verbal confidence works consistently
- Monitor performance and cost differences

### 5. Monitor and Optimize
- Track confidence scores over time
- Compare provider performance
- Optimize model selection by task

## Troubleshooting

### Issue: "Model not found"
**Cause**: Provider extra not installed or wrong model name format

**Solution**:
```bash
# Install provider extra
uv add "pydantic-ai-slim[google]"

# Verify model name format
# OpenAI: "openai:gpt-4o-mini"
# Anthropic: "anthropic:claude-3-5-sonnet-20241022"
# Gemini: "gemini-2.5-flash-lite"
```

### Issue: "API key not found"
**Cause**: Environment variable not set

**Solution**:
```bash
# Set the appropriate key
export OPENAI_AI_API_KEY=sk-...
export ANTHROPIC_AI_API_KEY=sk-ant-...
export GEMINI_AI_API_KEY=...
```

### Issue: "Validation failed: confidence_score missing"
**Cause**: LLM didn't populate confidence_score field

**Solution**: This should auto-retry. If it persists:
1. Check system prompt includes confidence instructions
2. Verify Pydantic model has confidence_score field
3. Increase `max_retries` in configuration

### Issue: "Inconsistent confidence scores"
**Cause**: Different providers may interpret guidelines differently

**Solution**:
1. Refine confidence guidelines in system prompt
2. Add examples of different confidence levels
3. Monitor and calibrate thresholds per provider

## Further Reading

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Verbal Confidence Pattern Research](https://arxiv.org/abs/2305.14975)
- [Provider Comparison Guide](./PROVIDER_COMPARISON.md)
- [Cost Optimization Strategies](./COST_OPTIMIZATION.md)
