# AI Provider Setup Guide

This project is **provider-agnostic** and supports multiple AI providers through Pydantic AI. You can use OpenAI, Anthropic, Google Gemini, or any other provider supported by Pydantic AI.

## How It Works

The system uses environment variables to configure both:
1. **API Keys** - Authentication credentials for AI providers
2. **Model Selection** - Which models to use for different tasks

Pydantic AI automatically detects the provider from the model name prefix and uses the corresponding API key from environment variables.

## Setup Steps

### 1. Choose Your AI Provider(s)

You need **at least one** AI provider configured. You can configure multiple providers and switch between them by changing the model configuration.

#### OpenAI
- **API Key Variable:** `OPENAI_AI_API_KEY`
- **Get Key From:** <https://platform.openai.com/api-keys>
- **Model Format:** `openai:model-name`
- **Example Models:**
  - `openai:gpt-4o-mini` (fast, cost-effective)
  - `openai:gpt-4o` (most capable)
  - `openai:gpt-4-turbo`

#### Anthropic (Claude)
- **API Key Variable:** `ANTHROPIC_AI_API_KEY`
- **Get Key From:** <https://console.anthropic.com/>
- **Model Format:** `anthropic:model-name`
- **Example Models:**
  - `anthropic:claude-3-5-sonnet-20241022` (balanced)
  - `anthropic:claude-3-5-haiku-20241022` (fast)
  - `anthropic:claude-3-opus-20240229` (most capable)

#### Google Gemini
- **API Key Variable:** `GEMINI_AI_API_KEY`
- **Get Key From:** <https://aistudio.google.com/app/apikey>
- **Model Format:** `gemini-model-name` (no prefix needed)
- **Example Models:**
  - `gemini-1.5-flash` (fast)
  - `gemini-1.5-pro` (balanced)
  - `gemini-2.0-flash-exp` (experimental)

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and add your API key(s):

```bash
# Add at least one API key
OPENAI_AI_API_KEY=sk-your-openai-key-here
ANTHROPIC_AI_API_KEY=sk-ant-your-anthropic-key-here
GEMINI_AI_API_KEY=your-gemini-key-here

# Configure which models to use (REQUIRED)
ANOMALY_MODEL=openai:gpt-4o-mini
ROOT_CAUSE_MODEL=openai:gpt-4o
```

### 3. Select Models

You **must** configure both `ANOMALY_MODEL` and `ROOT_CAUSE_MODEL`. There are no defaults to avoid provider bias.

**Example configurations:**

```bash
# All OpenAI
ANOMALY_MODEL=openai:gpt-4o-mini
ROOT_CAUSE_MODEL=openai:gpt-4o

# All Anthropic
ANOMALY_MODEL=anthropic:claude-3-5-haiku-20241022
ROOT_CAUSE_MODEL=anthropic:claude-3-5-sonnet-20241022

# All Gemini
ANOMALY_MODEL=gemini-1.5-flash
ROOT_CAUSE_MODEL=gemini-1.5-pro

# Mixed providers (requires multiple API keys)
ANOMALY_MODEL=openai:gpt-4o-mini
ROOT_CAUSE_MODEL=anthropic:claude-3-5-sonnet-20241022
```

## Validation

The system validates your configuration at startup:

1. **At least one API key must be set**
2. **Both model configurations must be provided**
3. **The selected models must have corresponding API keys**

For example, if you set `ANOMALY_MODEL=openai:gpt-4o-mini`, you must also have `OPENAI_AI_API_KEY` configured.

## Testing Your Configuration

Run the configuration test:

```bash
uv run python -m core.config
```

Or run the full system test:

```bash
uv run python test_system.py
```

## Docker Deployment

When deploying with Docker, pass the environment variables:

```bash
# Set in your shell or .env file
export OPENAI_AI_API_KEY=sk-your-key
export ANOMALY_MODEL=openai:gpt-4o-mini
export ROOT_CAUSE_MODEL=openai:gpt-4o

# Run with docker-compose
docker-compose up
```

The `docker-compose.yml` automatically passes through all AI provider environment variables.

## Switching Providers

To switch providers, simply update your `.env` file:

```bash
# Before (OpenAI)
ANOMALY_MODEL=openai:gpt-4o-mini
ROOT_CAUSE_MODEL=openai:gpt-4o

# After (Anthropic)
ANOMALY_MODEL=anthropic:claude-3-5-haiku-20241022
ROOT_CAUSE_MODEL=anthropic:claude-3-5-sonnet-20241022
```

No code changes required! The system automatically uses the correct API key based on the model prefix.

## Cost Optimization

Different models have different costs and capabilities. Consider:

- **Development:** Use faster, cheaper models (`gpt-4o-mini`, `claude-3-5-haiku`, `gemini-1.5-flash`)
- **Production:** Balance cost and quality based on your needs
- **Mixed approach:** Use cheaper models for frequent tasks (anomaly detection) and more capable models for complex analysis (root cause)

## Troubleshooting

### "At least one AI provider API key must be set"
- Check that you have at least one `*_AI_API_KEY` environment variable set
- Verify the `.env` file is in the project root
- Ensure the key format is correct (usually starts with `sk-` for OpenAI/Anthropic)

### "Missing API keys for selected models"
- The model you configured requires an API key that isn't set
- For example, `openai:gpt-4o` requires `OPENAI_AI_API_KEY`
- Add the missing API key or switch to a different model

### "ANOMALY_MODEL must be set in environment"
- Both `ANOMALY_MODEL` and `ROOT_CAUSE_MODEL` are required
- Add them to your `.env` file
- No defaults are provided to keep the system provider-agnostic

## Legacy Environment Variables

For backward compatibility, the system also checks:
- `OPENAI_API_KEY` (in addition to `OPENAI_AI_API_KEY`)
- `ANTHROPIC_API_KEY` (in addition to `ANTHROPIC_AI_API_KEY`)
- `GOOGLE_API_KEY` (in addition to `GEMINI_AI_API_KEY`)
- `ANOMALY_DETECTION_MODEL` (in addition to `ANOMALY_MODEL`)

However, we recommend using the new `*_AI_API_KEY` format for clarity.
