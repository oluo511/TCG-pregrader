"""
Training Data Pipeline — standalone async module for collecting and labeling
PSA slab photos for TCG Pre-Grader CNN training.

Data flow:
  eBay / CardLadder scrapers → Deduplicator → PSA Client → Manifest Builder
"""
