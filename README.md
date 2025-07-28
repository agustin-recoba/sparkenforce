# :rocket: sparkenforce

**sparkenforce** is a type annotation system that lets you specify and validate PySpark DataFrame schemas using Python type hints. It validates both function arguments and return values, catching schema mismatches before they cause runtime errors.

## Why sparkenforce?

Working with PySpark DataFrames can be error-prone when schemas don't match expectations. sparkenforce helps by:

- **Preventing runtime errors**: Catch schema mismatches early with type validation
- **Improving code clarity**: Function signatures show exactly what DataFrame structure is expected
- **Enforcing contracts**: Ensure functions return DataFrames with the promised schema
- **Better debugging**: Clear error messages when validations fail

## Quick Start

### Validating Input DataFrames

```python
from sparkenforce import validate, Dataset
from pyspark.sql import functions as fn

@validate
def add_length(df: Dataset['firstname':str, ...]) -> Dataset['name':str, 'length':int]:
    return df.select(
        df.firstname.alias('name'),
        fn.length(df.firstname).alias('length')
    )

# If input DataFrame doesn't have 'firstname' column, validation fails
# If return DataFrame doesn't match expected schema, validation fails
```

### Flexible Schemas with Ellipsis

Use `...` to allow additional columns beyond the specified ones:

```python
@validate
def filter_names(df: Dataset['firstname':str, 'lastname':str, ...]):
    """Requires firstname and lastname, but allows other columns too."""
    return df.filter(df.firstname != "")
```

### Return Value Validation

sparkenforce validates that your function returns exactly what you promise:

```python
@validate
def get_summary(df: Dataset['firstname':str, ...]) -> Dataset['firstname':str, 'summary':str]:
    return df.select(
        'firstname',
        fn.lit('processed').alias('summary'),
    )
```

### Error Handling

When validation fails, sparkenforce provides clear error messages:

```python
# This will raise DatasetValidationError with detailed message:
# "return value columns mismatch. Expected exactly {'name', 'length'},
#  got {'lastname', 'firstname'}. missing columns: {'name', 'length'},
#  unexpected columns: {'lastname', 'firstname'}"

@validate
def bad_function(df: Dataset['firstname':str, ...]) -> Dataset['name':str, 'length':int]:
    return df.select('firstname', 'lastname')  # Wrong columns!
```

## Installation

Install sparkenforce using pip:

```bash
pip install sparkenforce
```

Or if you're using uv:

```bash
uv add sparkenforce
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
