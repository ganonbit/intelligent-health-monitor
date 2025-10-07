# Mypy + Pydantic Configuration Guide

This document explains the mypy configuration for this Pydantic-heavy project and common quirks you might encounter.

## Configuration Overview

Our `mypy.ini` is optimized for:
- ✅ Pydantic v2 models
- ✅ Pydantic AI agents
- ✅ Strict type checking
- ✅ Production-ready code quality

## Pydantic Plugin Settings

### Core Plugin Configuration

```ini
[pydantic-mypy]
init_forbid_extra = True          # Catch typos in model initialization
init_typed = True                 # Type-check __init__ methods
warn_required_dynamic_aliases = True  # Warn about Field(alias=...)
warn_untyped_fields = True        # Ensure all fields have types
```

### What Each Setting Does

#### `init_forbid_extra = True`
**Catches typos in model instantiation:**

```python
class User(BaseModel):
    name: str
    age: int

# ❌ Mypy catches this typo
user = User(name="Alice", agee=30)  # Error: Unexpected keyword argument "agee"

# ✅ Correct
user = User(name="Alice", age=30)
```

#### `init_typed = True`
**Type-checks model initialization:**

```python
class Config(BaseModel):
    timeout: float
    retries: int

# ❌ Mypy catches type errors
config = Config(timeout="30", retries=3.5)  # Error: incompatible types

# ✅ Correct
config = Config(timeout=30.0, retries=3)
```

#### `warn_required_dynamic_aliases = True`
**Warns about Field aliases that might cause issues:**

```python
class APIResponse(BaseModel):
    user_id: int = Field(alias="userId")  # ⚠️ Warning: consider using alias_generator

# Better approach
class APIResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)
    user_id: int
```

#### `warn_untyped_fields = True`
**Ensures all fields have explicit types:**

```python
class Data(BaseModel):
    value = Field(default=10)  # ❌ Error: Field has no type annotation

# ✅ Correct
class Data(BaseModel):
    value: int = Field(default=10)
```

## Common Pydantic Quirks

### 1. Pydantic AI Agent Model Names

**Problem:** Pydantic AI uses strict `Literal` types for model names

```python
# Pydantic AI's signature
Agent(model: Literal['openai:gpt-4o', 'anthropic:claude-3-5-sonnet-20241022', ...])

# Your code
config.model_name: str = "gemini-2.5-flash-lite"
Agent(model=config.model_name)  # ❌ Error: str doesn't match Literal
```

**Solution:** Disable `call-overload` for files using dynamic model selection

```ini
[mypy-core.services.ai_analysis]
disable_error_code = call-overload
```

### 2. Complex Generic Types

**Problem:** Pydantic AI uses complex generics that confuse mypy

```python
Agent[AgentDepsT, OutputDataT]  # Complex nested generics
```

**Solution:**
```ini
[mypy-pydantic_ai.*]
disable_error_code = type-arg,no-any-return
```

### 3. Validators and Decorators

**Problem:** Pydantic validators use complex decorator magic

```python
class Model(BaseModel):
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        # Mypy might complain about decorator usage
        return v
```

**Solution:**
```ini
[mypy-pydantic.functional_validators]
disable_error_code = misc
```

### 4. Model Inheritance

**Problem:** Mypy sometimes struggles with BaseModel inheritance

```python
class Base(BaseModel):
    id: int

class Extended(Base):
    name: str

# Sometimes mypy gets confused about inherited fields
```

**Solution:** Usually works fine with `strict = True`, but if issues arise:
```python
# Use explicit type annotations
extended: Extended = Extended(id=1, name="test")
```

## Best Practices

### 1. Always Use Field Descriptions

```python
# ❌ Minimal
class Config(BaseModel):
    timeout: float

# ✅ Better - helps with documentation and validation
class Config(BaseModel):
    timeout: float = Field(
        gt=0.0,
        description="Request timeout in seconds"
    )
```

### 2. Use ConfigDict for Model Configuration

```python
# ✅ Modern Pydantic v2 style
class Model(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra='forbid'
    )
```

### 3. Explicit Type Annotations

```python
# ❌ Implicit
class Data(BaseModel):
    items = []  # Mypy doesn't know the type

# ✅ Explicit
class Data(BaseModel):
    items: list[str] = Field(default_factory=list)
```

### 4. Use Computed Fields Properly

```python
from pydantic import computed_field

class User(BaseModel):
    first_name: str
    last_name: str

    @computed_field  # ✅ Mypy understands this
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
```

## Common Errors and Solutions

### Error: "Incompatible types in assignment"

```python
# ❌ Problem
class Config(BaseModel):
    value: int

config = Config(value="10")  # Error: Expected int, got str

# ✅ Solution: Pydantic coerces at runtime, but mypy is strict
config = Config(value=int("10"))
```

### Error: "Unexpected keyword argument"

```python
# ❌ Problem
class Model(BaseModel):
    name: str

m = Model(name="test", extra="value")  # Error with init_forbid_extra

# ✅ Solution: Only pass defined fields
m = Model(name="test")
```

### Error: "Cannot determine type of field"

```python
# ❌ Problem
class Model(BaseModel):
    data = Field(default_factory=dict)  # No type annotation

# ✅ Solution
class Model(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)
```

## Testing with Pydantic

### Type-Safe Test Fixtures

```python
import pytest
from pydantic import BaseModel

class TestData(BaseModel):
    name: str
    value: int

@pytest.fixture
def test_data() -> TestData:  # ✅ Type-annotated fixture
    return TestData(name="test", value=42)

def test_something(test_data: TestData) -> None:
    assert test_data.name == "test"  # Mypy knows the type
```

## Integration with FastAPI

If you add FastAPI later:

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Request(BaseModel):
    query: str

app = FastAPI()

@app.post("/search")
async def search(request: Request) -> dict[str, str]:  # ✅ Fully typed
    return {"result": request.query}
```

Mypy will:
- ✅ Validate request/response types
- ✅ Catch missing fields
- ✅ Ensure type consistency

## Troubleshooting

### Mypy is too strict

If mypy is catching false positives:

```python
# Use type: ignore with a specific error code
result = some_complex_function()  # type: ignore[no-any-return]
```

### Pydantic model not recognized

Ensure the plugin is loaded:
```bash
uv run mypy --version
# Should show: pydantic.mypy plugin loaded
```

### Performance issues

For large projects, use mypy daemon:
```bash
uv run dmypy run -- .
```

## Summary

Your `mypy.ini` configuration:
- ✅ Strict type checking for production code
- ✅ Pydantic plugin with all safety features enabled
- ✅ Relaxed rules for tests (pragmatic)
- ✅ Specific overrides for known Pydantic AI quirks
- ✅ Industry-standard best practices

This gives you:
- 🛡️ Type safety without runtime overhead
- 🐛 Catch bugs at development time
- 📚 Better IDE autocomplete
- 🚀 Production-ready code quality

## Further Reading

- [Pydantic Mypy Plugin Docs](https://docs.pydantic.dev/latest/integrations/mypy/)
- [Mypy Configuration Reference](https://mypy.readthedocs.io/en/stable/config_file.html)
- [Pydantic AI Type Safety](https://ai.pydantic.dev/models/overview/)
