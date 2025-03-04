# --- If using UV ---
uv venv
uv sync
echo "source ${UV_PROJECT_ENVIRONMENT}/bin/activate" >> ~/.bashrc
