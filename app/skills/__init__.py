"""High-level skills built on top of Houdini actions.

Skills compose primitive Houdini actions (create geometry, wire nodes,
set parameters, etc.) into reusable, named capabilities exposed to the
LLM planner.
"""

from app.skills import procedural_rock

__all__ = ["procedural_rock"]
