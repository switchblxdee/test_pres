"""Генератор еженедельного дайджеста «Голос IT» (.pptx) из xlsx."""
from .agent import DigestAgent, DigestResult, build
from .config import Config, load_config

__all__ = ["DigestAgent", "DigestResult", "Config", "load_config", "build"]
