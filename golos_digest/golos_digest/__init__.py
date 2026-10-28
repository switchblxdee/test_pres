"""Генератор еженедельного дайджеста «Голос IT» (.pptx) из xlsx."""
from .config import Config, load_config
from .graph import build, build_graph

__all__ = ["Config", "load_config", "build", "build_graph"]
