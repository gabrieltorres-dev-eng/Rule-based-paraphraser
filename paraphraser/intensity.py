"""Global structure-intensity switch. LEVEL 0 = match human base rates (gentle); LEVEL 1 = aggressive restructuring."""
LEVEL = 1
def aggressive() -> bool:
    return LEVEL >= 1

RETIRED: set = set()   # content words earlier passes removed from the text; lexical swaps must not bring them back
SEEN: set = set()      # restored sentence forms produced/seen in earlier passes; later passes must not revert to them

RECIPE: dict = {}      # per-candidate transform-family weights (structure.draw_recipe); empty = every family at weight 1
BOOST = 1.0

def set_recipe(recipe: dict | None):
    RECIPE.clear()
    global BOOST
    BOOST = 1.0
    if recipe:
        BOOST = float(recipe.get("_boost", 1.0))
        RECIPE.update({k: v for k, v in recipe.items() if k != "_boost"})

def p(family: str, base: float) -> float:
    """Probability for a transform family in the current candidate's recipe (clamped)."""
    w = RECIPE.get(family, 1.0)
    return float(min(0.98, max(0.0, base * w * BOOST)))
