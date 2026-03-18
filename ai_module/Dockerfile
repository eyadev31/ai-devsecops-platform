FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn sqlalchemy

EXPOSE 8000

CMD ["uvicorn" , "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
