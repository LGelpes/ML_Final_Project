FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

COPY configs/ ${LAMBDA_TASK_ROOT}/configs/
COPY weights/ ${LAMBDA_TASK_ROOT}/weights/
COPY src/__init__.py ${LAMBDA_TASK_ROOT}/src/__init__.py
COPY src/config.py ${LAMBDA_TASK_ROOT}/src/config.py
COPY src/api/ ${LAMBDA_TASK_ROOT}/src/api/

CMD ["src.api.app.lambda_handler"]
