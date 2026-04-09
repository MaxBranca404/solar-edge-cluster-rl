FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir stable-baselines3 gymnasium shimmy fastapi uvicorn requests pydantic numpy streamlit pandas matplotlib altair
COPY . .