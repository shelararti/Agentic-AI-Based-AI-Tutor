"""
Cheap hallucination check: how much of a generated text's vocabulary appears
in the retrieved notes? This is a heuristic (lexical overlap on word stems),
not a fact-checker -- it flags output the teacher should double-check.
It only makes sense when the output language matches the notes' language, so
callers skip it for translated output.
"""

import re

_STOP = set("""about above after again also because before being below between both could does doing down
during each from further have having hereherself himself into itself just more most myself once only other
ought over same should some such than that their theirs them then there these they this those through under
until very were what when where which while with would your yours yourself will been it's can't don't""".split())
_WORD = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
GROUNDED_AT = 0.5


def _stems(text: str) -> set:
    return {w.lower()[:6] for w in _WORD.findall(text or "") if w.lower() not in _STOP}


def support(text: str, context: str):
    """Fraction (0..1) of the text's content-word stems found in context; None if too little to judge."""
    words = _stems(text)
    if len(words) < 3:
        return None
    return round(len(words & _stems(context)) / len(words), 2)


def label(score) -> str:
    if score is None:
        return "not enough text to check"
    if score >= 0.6:
        return "well grounded in your notes"
    if score >= 0.4:
        return "partly grounded - skim it against your notes"
    return "weakly grounded - check carefully against your notes"
