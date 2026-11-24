"""
Ethereum Guard - Application Package Root.
Exports FastAPI service and Streamlit dashboard modules.
"""

from src.app.engine import EthereumGuardEngine, get_engine
from src.app.service import app

__all__ = ["app", "get_engine", "EthereumGuardEngine"]
