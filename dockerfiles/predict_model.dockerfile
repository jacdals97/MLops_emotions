FROM python:3.11

EXPOSE 8000

RUN mkdir /code
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./emotions/ /code/emotions/ 
COPY pyproject.toml /code/pyproject.toml

WORKDIR /code
RUN pip install . --no-deps --no-cache-dir


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
