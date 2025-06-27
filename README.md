# :rocket: sparkenforce

Type annotation system that allows you to specify and validate the schema of PySpark DataFrames using Python type hints for both function arguments and return values.

## Setup Dev Environment

Installation is using [UV](https://docs.astral.sh/uv/) to manage everything.

**Step 1**: Create a virtual environment

```
uv venv
```

**Step 2**: Activate your new environment

```
# on windows
.venv\Scripts\activate

# on mac / linux
source .venv/bin/activate
```

**Step 3**: Install all the cool dependencies

```
uv sync
```

## Github Repo Setup

To add your new project to its Github repository, firstly make sure you have created a project named **sparkenforce** on Github.
Follow these steps to push your new project.

```
git remote add origin git@github.com:agustin-recoba/sparkenforce.git
git branch -M main
git push -u origin main
```

## Built-in CLI Commands

We've included a bunch of useful CLI commands for common project tasks using [taskipy](https://github.com/taskipy/taskipy).

```
# run src/sparkenforce/sparkenforce.py
task run

# run all tests
task tests



# run test coverage and generate report
task coverage

# typechecking with Ty or Mypy
task type

# ruff linting
task lint

# format with ruff
task format
```

## Docker Setup

A Dockerfile optimized to reduce the image size has been included. To get it up and running follow these steps.

**Step 1**: Build your Docker image.

```
docker build --progress=plain -t "sparkenforce:Dockerfile" .
```

**Step 2**: Run your new image.

```
docker run --rm sparkenforce:Dockerfile
```

## PyPI Deployment

1. Register your project and create an API Token on [PyPI](https://pypi.org/).
2. Add the API Token to your projects secrets with the name `PYPI_TOKEN`
3. Create a new release on Github.
4. Create a new tag in the form `*.*.*`.

## Dependabot Setup

1. Go to the "Settings -> Advanced Security" tab in your repository.
2. Under the "Dependabot" section enable the options you want to monitor, we recommend the "Dependabot security updates" at the minimum.

Dependabot is configured to do _weekly_ scans of your dependencies, and pull requests will be prefixed with "DBOT". These settings can be adjusted in the `./.github/dependabot.yml` file.

## References

- [Cookiecutter Python Project](https://github.com/wyattferguson/pattern) - A modern cookiecutter template for your next Python project.

## License

Apache Software License v2.0

## Contact

Created by [Agustín Recoba](https://github.com/agustin-recoba)
