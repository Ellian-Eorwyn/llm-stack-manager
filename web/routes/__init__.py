"""Flask blueprints, one per area of the manager's HTTP surface.

Each module registers its routes with the literal rules they had when they
lived in `app.py` — no `url_prefix` — so the URL map is unchanged by the move.
"""
