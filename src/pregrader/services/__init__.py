"""
Services package for the TCG Pre-Grader.
"""

from pregrader.services.grader import GraderService
from pregrader.services.ingestion import ImageIngestionService
from pregrader.services.preprocessing import PreprocessingService

__all__ = ["ImageIngestionService", "PreprocessingService", "GraderService"]
