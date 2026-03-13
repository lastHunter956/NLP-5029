"""
logic — Módulo de procesamiento de texto y extracción de features para NLP.
===========================================================================
Exporta:
  • TextProcessing              — pipeline completo (sklearn-compatible)
  • SentimentFeaturesTransformer — 26 features hand-crafted
  • FastTextFeaturesTransformer  — 300 features fastText (carga lazy)
  • strip_accents, pseudo_stem  — utilidades lingüísticas
  • Léxicos (7 conjuntos + aliases LEXICON_POS / LEXICON_NEG)
"""
from .text_processing import TextProcessing, pseudo_stem, strip_accents
from .feature_extraction import (
    SentimentFeaturesTransformer,
    FastTextFeaturesTransformer,
    SpaCyVectorTransformer,
    FEATURE_NAMES,
)
from .lexicon_es import (
    POSITIVE_WORDS, NEGATIVE_WORDS,
    NEGATORS, INTENSIFIERS, ATTENUATORS,
    POSITIVE_EMOJIS, NEGATIVE_EMOJIS,
    POSITIVE_BIGRAMS, NEGATIVE_BIGRAMS,
    SPELL_CORRECTIONS,
    LEXICON_POS, LEXICON_NEG,
)

__all__ = [
    # Transformadores
    'TextProcessing',
    'SentimentFeaturesTransformer',
    'FastTextFeaturesTransformer',
    'SpaCyVectorTransformer',
    'FEATURE_NAMES',
    # Utilidades
    'pseudo_stem',
    'strip_accents',
    # Léxicos
    'POSITIVE_WORDS', 'NEGATIVE_WORDS',
    'LEXICON_POS', 'LEXICON_NEG',
    'NEGATORS', 'INTENSIFIERS', 'ATTENUATORS',
    'POSITIVE_EMOJIS', 'NEGATIVE_EMOJIS',
    'POSITIVE_BIGRAMS', 'NEGATIVE_BIGRAMS',
    'SPELL_CORRECTIONS',
]
