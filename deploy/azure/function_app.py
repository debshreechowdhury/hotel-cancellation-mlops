"""Azure Functions entry point: serves the FastAPI app (api/main.py) unchanged.

The deploy script (scripts/deploy_azure.sh) copies this file, host.json,
requirements.txt, api/, src/ and models/ into one folder and publishes it.
"""
import azure.functions as func

from api.main import app as fastapi_app

app = func.AsgiFunctionApp(app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS)
