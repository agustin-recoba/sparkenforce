# :rocket: sparkenforce

**sparkenforce** is a type annotation system that lets you specify and validate PySpark DataFrame schemas using Python type hints. It validates both function arguments and return values, catching schema mismatches before they cause runtime errors.

## Why sparkenforce?

Working with PySpark DataFrames can be error-prone when schemas don't match expectations. sparkenforce helps by:

- **Preventing runtime errors**: Catch schema mismatches early with type validation
- **Improving code clarity**: Function signatures show exactly what DataFrame structure is expected
- **Enforcing contracts**: Ensure functions return DataFrames with the promised schema
- **Better debugging**: Clear error messages when validations fail

## Installation

Install sparkenforce using pip:

```bash
pip install sparkenforce
```

Or if you're using uv:

```bash
uv add sparkenforce
```

## Quick Start

### Validating Input DataFrames

```python
import sparkenforce
from pyspark.sql import functions as F

@sparkenforce.validate
def transform_data(df: sparkenforce.Dataset['firstname':str, ...]) -> sparkenforce.Dataset['name':str, 'length':int]:
    """Transform DataFrame with validated input and output schemas."""
    return df.select(
        df.firstname.alias('name'),
        F.length(df.firstname).alias('length')
    )

# If input DataFrame doesn't have 'firstname' column, validation fails
# If return DataFrame doesn't match expected schema, validation fails
```

### Flexible Schemas with Ellipsis

Use `...` to allow additional columns beyond the specified ones:

```python
@sparkenforce.validate
def process_names(df: sparkenforce.Dataset['firstname':str, 'lastname':str, ...]):
    """Requires firstname and lastname, but allows other columns too."""
    return df.filter(df.firstname != "")
```

### Return Value Validation

sparkenforce validates that your function returns exactly what you promise:

```python
@sparkenforce.validate
def get_summary(df: sparkenforce.Dataset['firstname':str, ...]) -> sparkenforce.Dataset['firstname':str, 'summary':str, ...]:
    return df.select(
        'firstname',
        F.lit('processed').alias('summary'),
        'lastname'  # Additional columns allowed with ...
    )
```

## Error Handling

When validation fails, sparkenforce provides clear error messages:

```python
# This will raise DatasetValidationError with detailed message:
# "return value columns mismatch. Expected exactly {'name', 'length'}, 
#  got {'lastname', 'firstname'}. missing columns: {'name', 'length'}, 
#  unexpected columns: {'lastname', 'firstname'}"

@sparkenforce.validate  
def bad_function(df: sparkenforce.Dataset['firstname':str, ...]) -> sparkenforce.Dataset['name':str, 'length':int]:
    return df.select('firstname', 'lastname')  # Wrong columns!
```


## Development Setup

**Step 1**: Create virtual environment
```bash
uv venv
```

**Step 2**: Activate environment
```bash
# Linux/Mac
source .venv/bin/activate

# Windows  
.venv\Scripts\activate
```

**Step 3**: Install dependencies
```bash
uv sync
```

## CLI Commands

```bash
# Run tests
task tests

# Type checking
task type

# Linting
task lint

# Format code
task format

# Coverage report
task coverage
```

## Inspiration

This project builds on [dataenforce](https://github.com/CedricFR/dataenforce), extending it with additional validation capabilities for PySpark DataFrame workflows.

## License

Apache Software License v2.0

## Contact

Created by [Agustín Recoba](https://github.com/agustin-recoba)
