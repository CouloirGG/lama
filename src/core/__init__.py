"""
LAMA Core — game-agnostic pricing engine.

Usage:
    from core import PricingEngine, GameConfig
    from games.poe2 import create_poe2_config

    engine = PricingEngine(create_poe2_config())
    engine.initialize()
    result = engine.lookup(clipboard_text)
"""

from core.game_config import GameConfig
from core.pricing_engine import PricingEngine

# Re-export key types from existing modules for consumer convenience.
# These are lazy to avoid import errors when modules aren't on sys.path.

def _lazy_import(module_name, attr_name):
    """Create a lazy attribute that imports on first access."""
    def _get():
        import importlib
        mod = importlib.import_module(module_name)
        return getattr(mod, attr_name)
    return _get


# Accessors for key types — import when needed:
#   from item_parser import ParsedItem
#   from mod_parser import ParsedMod
#   from mod_database import Grade, ItemScore
#   from trade_client import RarePriceResult

__all__ = [
    "PricingEngine",
    "GameConfig",
]
