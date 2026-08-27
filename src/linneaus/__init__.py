"""
Linneaus: AI-powered classification of Internet Autonomous Systems.

A tool for automatically classifying and analyzing Autonomous Systems (AS) that make up
the Internet using machine learning and large language models.
"""

__version__ = "0.1.0"
__author__ = "Esteban"
__email__ = "linneaus@example.com"

try:
    from .config import Config
    from .models.schemas import ASClassification, OrganizationData

    __all__ = ["Config", "ASClassification", "OrganizationData"]
except ImportError:
    # Handle import errors during development
    __all__ = []
