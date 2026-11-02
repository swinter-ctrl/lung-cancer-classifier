# Container for the Streamlit app (deployed on Render, see render.yaml).
FROM python:3.12-slim

# One CPU thread: the free instance has a fraction of a CPU and 512 MB RAM,
# and extra threads only add memory overhead.
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch first (~200 MB instead of several GB of CUDA libraries),
# then the rest of the runtime requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

# Render sets $PORT; default to 8501 when run locally with `docker run -p 8501:8501`.
EXPOSE 8501
CMD streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 \
    --server.headless=true --browser.gatherUsageStats=false
