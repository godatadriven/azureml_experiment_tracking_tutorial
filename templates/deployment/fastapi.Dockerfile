FROM --platform=linux/amd64 python:3.8-slim


# Install poetry
RUN pip3 install poetry==1.3.2
ENV PYTHONIOENCODING=utf8
RUN poetry config virtualenvs.in-project true


# Copy the project dependencies
WORKDIR /app
ADD pyproject.toml poetry.lock ./
# First install project dependencies without root.
# this will speed up future builds when only the src has changed
RUN poetry env use $(which python3.8)
RUN poetry install --no-interaction --no-dev --no-root

# We add the src and model folders after the dependencies are installed
# This allows us to cache the dependency install layer.
COPY models models
COPY uvicorn_log_conf.yml log_conf.yml
COPY azureml_deployment/src src


## After copying the src and model folders, we can install our own package.
RUN poetry install --no-interaction --no-dev

# Clean caches after install
RUN poetry cache clear pypi --all --no-interaction

# Start application
ENV MODEL_DIR=models/
EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config",  "log_conf.yml"]
