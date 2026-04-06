"""
Services package for the TCG Pre-Grader.

Exposes the public service classes used by the API and CLI layers.
Each service is stateless (config injected at construction) so they
can be instantiated once and reused across requests.
"""

from pregrader.services.grader import GraderService
from pregrader.services.ingestion import ImageIngestionService

__all__ = ["ImageIngestionService", "GraderService"]
