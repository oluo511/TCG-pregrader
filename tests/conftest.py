"""
Pytest configuration and Hypothesis profile setup.

Why register a "ci" profile here instead of inline in each test?
- Centralizes the max_examples budget — one change here applies to all
  property tests, so tuning for CI speed vs. thoroughness is a single edit.
- HealthCheck.too_slow suppression prevents Hypothesis from aborting tests
  that generate large images or tensors, which are legitimately slow to build.
"""

from hypothesis import HealthCheck, settings

# Register the CI profile used by all property tests in this suite.
# @settings(max_examples=100) is the default; override per-test only when
# a property is computationally expensive and 100 examples is impractical.
settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile("ci")
