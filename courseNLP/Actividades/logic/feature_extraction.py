"""
feature_extraction.py — Extracción de features manuales para sentimiento en ES.
=================================================================================
Dos transformadores sklearn para generar representaciones numéricas a partir
de texto procesado por TextProcessing:

  SentimentFeaturesTransformer   — 26 features hand-crafted (lexicón + sintaxis)
  FastTextFeaturesTransformer    — 300 features de embeddings fastText (opcional)

Pueden usarse juntos via sklearn FeatureUnion::

    from sklearn.pipeline import Pipeline, FeatureUnion
    from logic.text_processing import TextProcessing
    from logic.feature_extraction import SentimentFeaturesTransformer, FastTextFeaturesTransformer

    features = FeatureUnion([
        ('manual', SentimentFeaturesTransformer()),
        ('embed',  FastTextFeaturesTransformer()),
    ])
    pipe = Pipeline([
        ('clean',    TextProcessing()),
        ('features', features),
        ('clf',      LogisticRegression()),
    ])

Features manuales (26)
-----------------------
Índice  Nombre                Descripción
------  ──────────────────    ──────────────────────────────────────────────────
  0     pos_lexicon           Proporción de tokens positivos en léxico
  1     neg_lexicon           Proporción de tokens negativos en léxico
  2     polarity              pos_lexicon - neg_lexicon
  3     subjectivity          (pos + neg) / n — ~0 indica NONE/NEU
  4     negation_count        Proporción de tokens negadores
  5     negated_sentiment     Intensidad de tokens marcados neg_*
  6     intensifier_count     Proporción de intensificadores
  7     attenuator_count      Proporción de atenuadores
  8     emoji_pos_ratio       Proporción tokens emoji_pos
  9     emoji_neg_ratio       Proporción tokens emoji_neg
 10     laugh_ratio           Proporción tokens 'risa'
 11     pos_bigram_score      Proporción de bigramas positivos
 12     neg_bigram_score      Proporción de bigramas negativos
 13     has_url               ¿Contiene 'url'? (0/1)
 14     hashtag_ratio         Proporción de tokens 'hashtag'
 15     is_question           ¿Contiene '?' o '¿'? (0/1)
 16     first_person_ratio    Proporción de pronombres de 1ª persona
 17     all_caps_ratio        Proporción de palabras en mayúsculas (texto original)
 18     token_count_norm      len(tokens) / 50 — longitud relativa
 19     last_sentiment_pos    Posición normalizada del último token pos/neg (0-1)
 20     has_suspension        ¿Contiene 'suspensivo'? (señal de ironía/duda)
 21     has_intensif_marker   ¿Contiene 'marcador_intensidad'? (puto/puta/hostia)
 22     negation_position     Posición normalizada del primer negador (0-1)
 23     sentence_end_pos      ¿Último token es positivo? (1=pos, -1=neg, 0=neutro)
 24     cap_exclamation       Proporción caps + exclamaciones combinadas
 25     code_switch_ratio     Proporción de palabras en inglés frecuentes
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .lexicon_es import (
    ATTENUATORS,
    INTENSIFIERS,
    NEGATORS,
    NEGATIVE_BIGRAMS,
    NEGATIVE_WORDS,
    POSITIVE_BIGRAMS,
    POSITIVE_WORDS,
)
from .text_processing import strip_accents

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =====================================================================
# NOMBRES DE FEATURES
# =====================================================================
FEATURE_NAMES: List[str] = [
    'pos_lexicon',
    'neg_lexicon',
    'polarity',
    'subjectivity',
    'negation_count',
    'negated_sentiment',
    'intensifier_count',
    'attenuator_count',
    'emoji_pos_ratio',
    'emoji_neg_ratio',
    'laugh_ratio',
    'pos_bigram_score',
    'neg_bigram_score',
    'has_url',
    'hashtag_ratio',
    'is_question',
    'first_person_ratio',
    'all_caps_ratio',
    'token_count_norm',
    'last_sentiment_pos',
    'has_suspension',
    'has_intensif_marker',
    'negation_position',
    'sentence_end_pos',
    'cap_exclamation',
    'code_switch_ratio',
]

assert len(FEATURE_NAMES) == 26, f"Se esperaban 26 features, hay {len(FEATURE_NAMES)}"

# Conjuntos auxiliares
_FIRST_PERSON: frozenset = frozenset({'yo', 'me', 'mi', 'mis', 'nos', 'nosotros', 'nosotras'})
_CODE_SWITCH: frozenset = frozenset({
    'love', 'hate', 'amazing', 'awesome', 'great', 'cool', 'nice', 'cute',
    'perfect', 'beautiful', 'wonderful', 'fantastic', 'incredible', 'happy',
    'sad', 'bad', 'worst', 'awful', 'terrible', 'horrible', 'boring',
    'annoying', 'stupid', 'idiot', 'fun', 'funny', 'ugly',
})
_RE_EXCLAMATION = re.compile(r'[!¡]+')

# Tokens semánticos de emoji (V4 — 9 categorías via emoji library)
_EMOJI_POS_TOKENS: frozenset = frozenset({
    'emoji_pos',      # legacy
    'emoji_amor',     # 😍❤️💕
    'emoji_risa',     # 😂😆🤣
    'emoji_celebra',  # 👍🎉🏆
    'emoji_intenso',  # 🔥💥⚡
})
_EMOJI_NEG_TOKENS: frozenset = frozenset({
    'emoji_neg',      # legacy
    'emoji_tristeza', # 😢💔😞
    'emoji_rabia',    # 😡😤🤬
    'emoji_asco',     # 🤮💩😒
    'emoji_miedo',    # 😨😱💀
    'emoji_rechaza',  # 👎🚫❌
})

def _normalize_set(words: frozenset) -> frozenset:
    """Agrega versiones sin tilde de todas las palabras del léxico."""
    expanded = set(words)
    for w in words:
        s = strip_accents(w)
        if s != w:
            expanded.add(s)
    return frozenset(expanded)

_ALL_POS = _normalize_set(POSITIVE_WORDS | _EMOJI_POS_TOKENS)
_ALL_NEG = _normalize_set(NEGATIVE_WORDS | _EMOJI_NEG_TOKENS)
_NORM_NEGATORS = _normalize_set(NEGATORS)
_NORM_INTENSIFIERS = _normalize_set(INTENSIFIERS)
_NORM_ATTENUATORS = _normalize_set(ATTENUATORS)
_NORM_POS_BIGRAMS = _normalize_set(POSITIVE_BIGRAMS)
_NORM_NEG_BIGRAMS = _normalize_set(NEGATIVE_BIGRAMS)


# =====================================================================
# Registro robusto de texto crudo (reemplaza _SHARED_RAW_LOOKUP)
# =====================================================================
class RawTextRegistry:
    """Registro robusto de mapeo texto_procesado → texto_original.

    Módulo-level singleton. Resiste sklearn.clone() y cross_validate
    porque vive a nivel de módulo, no como atributo de instancia.
    Maneja colisiones (dos textos crudos → mismo texto limpio) guardando
    el último registrado y contando colisiones para diagnóstico.
    """

    def __init__(self):
        self._map: dict = {}
        self._collisions: int = 0

    def register_batch(self, clean_texts, raw_texts):
        """Registra batch de pares (texto_limpio, texto_crudo)."""
        for clean, raw in zip(clean_texts, raw_texts):
            if clean in self._map and self._map[clean] != raw:
                self._collisions += 1
            self._map[clean] = raw

    def get(self, clean_text: str):
        return self._map.get(clean_text)

    def clear(self):
        self._map.clear()
        self._collisions = 0

    @property
    def collisions(self):
        return self._collisions

    def __len__(self):
        return len(self._map)

    def __contains__(self, key):
        return key in self._map


_raw_registry = RawTextRegistry()

# Backward-compatible alias (legacy notebooks)
_SHARED_RAW_LOOKUP = _raw_registry._map


# =====================================================================
# SentimentFeaturesTransformer
# =====================================================================

class SentimentFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Extrae 26 features numéricas de sentimiento de textos preprocesados.

    Espera texto YA procesado por ``TextProcessing.transformer()``
    (minúsculas, sin acentos, tokens especiales como emoji_pos/emoji_neg/risa).

    Para calcular features que dependen del texto original (is_question,
    all_caps_ratio, cap_exclamation), usa el dict compartido
    ``_SHARED_RAW_LOOKUP`` a nivel de módulo. Este se puebla desde el
    notebook con ``feature_extraction._SHARED_RAW_LOOKUP = cleaner.raw_text_lookup_``
    después de ``TextProcessing.transform()``.

    Parámetros
    ----------
    original_texts : list or None
        Alternativa manual: lista de textos originales para el cálculo
        posicional (no funciona con cross_validate, solo con predict).
    """

    def __init__(self, original_texts: Optional[list] = None):
        self.original_texts = original_texts

    def fit(self, X: Any = None, y: Any = None) -> 'SentimentFeaturesTransformer':
        """No-op. Requerido por sklearn API."""
        return self

    def transform(
        self,
        X: Union[list, tuple],
        y: Any = None,
    ) -> np.ndarray:
        """Transforma cada texto en un vector de 26 features.

        Parámetros
        ----------
        X : array-like de str
            Textos preprocesados.

        Retorna
        -------
        np.ndarray de shape (n_samples, 26)
        """
        orig = self.original_texts
        result = []
        for i, text in enumerate(X):
            # Buscar texto crudo: 1) registry módulo, 2) original_texts
            raw = _raw_registry.get(text)
            if raw is None and orig is not None and i < len(orig):
                raw = orig[i]
            result.append(self._extract(text, raw_text=raw))
        return np.array(result, dtype=np.float64)

    def _extract(self, text: str, raw_text: Optional[str] = None) -> List[float]:
        """Extrae los 26 features para un único texto.

        Parámetros
        ----------
        text : str
            Texto preprocesado.
        raw_text : str o None
            Texto original (para all_caps_ratio, is_question exacto).
        """
        if not text or not isinstance(text, str):
            return [0.0] * 26

        tokens = text.split()
        n = max(len(tokens), 1)

        # ── Sets de tokens para búsqueda O(1) ────────────────────
        tok_set = set(tokens)

        # ── 0. pos_lexicon ────────────────────────────────────────
        pos_hits = sum(1 for t in tokens if t in _ALL_POS)
        pos_ratio = pos_hits / n

        # ── 1. neg_lexicon ────────────────────────────────────────
        neg_hits = sum(1 for t in tokens if t in _ALL_NEG)
        neg_ratio = neg_hits / n

        # ── 2. polarity ───────────────────────────────────────────
        polarity = pos_ratio - neg_ratio

        # ── 3. subjectivity ───────────────────────────────────────
        # ~0 indica tweets sin carga afectiva (NONE/NEU)
        subjectivity = (pos_hits + neg_hits) / n

        # ── 4. negation_count ─────────────────────────────────────
        neg_count = sum(1 for t in tokens if t in _NORM_NEGATORS) / n

        # ── 5. negated_sentiment ──────────────────────────────────
        # Tokens con prefijo neg_ que a su vez son términos de sentimiento
        negated = [
            t[4:] for t in tokens
            if t.startswith('neg_') and t[4:] in (_ALL_POS | _ALL_NEG)
        ]
        negated_score = len(negated) / n

        # ── 6. intensifier_count ──────────────────────────────────
        intens = sum(1 for t in tokens if t in _NORM_INTENSIFIERS) / n

        # ── 7. attenuator_count ───────────────────────────────────
        attenu = sum(1 for t in tokens if t in _NORM_ATTENUATORS) / n

        # ── 8. emoji_pos_ratio ────────────────────────────────────
        # Cuenta TODOS los tokens semánticos positivos (9 categorías V4)
        emoji_pos = sum(tokens.count(t) for t in _EMOJI_POS_TOKENS) / n

        # ── 9. emoji_neg_ratio ────────────────────────────────────
        # Cuenta TODOS los tokens semánticos negativos (9 categorías V4)
        emoji_neg = sum(tokens.count(t) for t in _EMOJI_NEG_TOKENS) / n

        # ── 10. laugh_ratio ───────────────────────────────────────
        laugh = tokens.count('risa') / n

        # ── 11. pos_bigram_score ──────────────────────────────────
        bigrams = ['_'.join(tokens[i:i + 2]) for i in range(n - 1)]
        trigrams = ['_'.join(tokens[i:i + 3]) for i in range(max(n - 2, 0))]
        all_ngrams = bigrams + trigrams
        _n_ng = max(len(all_ngrams), 1)
        pos_big = sum(1 for b in all_ngrams if b in _NORM_POS_BIGRAMS) / _n_ng

        # ── 12. neg_bigram_score ──────────────────────────────────────
        neg_big = sum(1 for b in all_ngrams if b in _NORM_NEG_BIGRAMS) / _n_ng

        # ── 13. has_url ───────────────────────────────────────────
        has_url = float('url' in tok_set)

        # ── 14. hashtag_ratio ─────────────────────────────────────
        hashtag_r = tokens.count('hashtag') / n

        # ── 15. is_question ───────────────────────────────────────
        # Usa texto original si disponible
        ref = raw_text if raw_text is not None else text
        is_q = float('?' in ref or '¿' in ref)

        # ── 16. first_person_ratio ────────────────────────────────
        first_p = sum(1 for t in tokens if t in _FIRST_PERSON) / n

        # ── 17. all_caps_ratio ────────────────────────────────────
        if raw_text:
            raw_words = raw_text.split()
            caps_ratio = (
                sum(1 for w in raw_words if w.isupper() and len(w) > 1)
                / max(len(raw_words), 1)
            )
        else:
            # Si no hay texto original, no podemos calcularlo
            caps_ratio = 0.0

        # ── 18. token_count_norm ──────────────────────────────────
        # Normalizado a 50 tokens (tweet típico)
        tok_norm = len(tokens) / 50.0

        # ── 19. last_sentiment_pos ───────────────────────────────
        # Posición [0,1] del último token de sentimiento
        # >0.5 → el sentimiento está cerca del final (más informativo)
        last_sent = 0.0
        for idx, t in enumerate(reversed(tokens)):
            if t in _ALL_POS or t in _ALL_NEG:
                last_sent = (n - 1 - idx) / (n - 1) if n > 1 else 0.5
                break

        # ── 20. has_suspension ───────────────────────────────────
        # Token 'suspensivo' generado por TextProcessing a partir de '...'
        # Señal de ironía, duda o suspense
        has_susp = float('suspensivo' in tok_set)

        # ── 21. has_intensif_marker ───────────────────────────────
        has_intens_marker = float('marcador_intensidad' in tok_set)

        # ── 22. negation_position ────────────────────────────────
        neg_pos = 0.0
        for idx, t in enumerate(tokens):
            if t in _NORM_NEGATORS or t.startswith('no_'):
                neg_pos = idx / (n - 1) if n > 1 else 0.0
                break

        # ── 23. sentence_end_pos ─────────────────────────────────
        # 1 = último token positivo, -1 = último token negativo, 0 = neutro
        last_tok = tokens[-1] if tokens else ''
        if last_tok in _ALL_POS:
            end_pos = 1.0
        elif last_tok in _ALL_NEG:
            end_pos = -1.0
        else:
            end_pos = 0.0

        # ── 24. cap_exclamation ──────────────────────────────────
        # Combinación caps + exclamaciones
        excl_count = (
            len(_RE_EXCLAMATION.findall(raw_text))
            if raw_text else
            len(_RE_EXCLAMATION.findall(text))
        )
        cap_excl = min((caps_ratio + excl_count / max(n, 1)) / 2.0, 1.0)

        # ── 25. code_switch_ratio ────────────────────────────────
        cs_ratio = sum(1 for t in tokens if t in _CODE_SWITCH) / n

        return [
            pos_ratio,        # 0
            neg_ratio,        # 1
            polarity,         # 2
            subjectivity,     # 3
            neg_count,        # 4
            negated_score,    # 5
            intens,           # 6
            attenu,           # 7
            emoji_pos,        # 8
            emoji_neg,        # 9
            laugh,            # 10
            pos_big,          # 11
            neg_big,          # 12
            has_url,          # 13
            hashtag_r,        # 14
            is_q,             # 15
            first_p,          # 16
            caps_ratio,       # 17
            tok_norm,         # 18
            last_sent,        # 19
            has_susp,         # 20
            has_intens_marker,# 21
            neg_pos,          # 22
            end_pos,          # 23
            cap_excl,         # 24
            cs_ratio,         # 25
        ]

    def get_feature_names_out(self) -> List[str]:
        """Retorna los nombres de los 26 features (sklearn API)."""
        return FEATURE_NAMES

    def get_feature_names(self) -> List[str]:
        """Alias retrocompatible."""
        return FEATURE_NAMES


# =====================================================================
# FastTextFeaturesTransformer (carga lazy)
# =====================================================================

class FastTextFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Embeddings fastText (cc.es.300.bin) — media por texto.

    Genera 300 features por texto usando los vectores pre-entrenados
    de fastText en español (Common Crawl, 300 dimensiones).

    El modelo se descarga automáticamente si no existe en ``model_path``
    via ``fasttext.util.download_model('es')``.

    Parámetros
    ----------
    model_path : str o None
        Ruta al fichero .bin de fastText. Si None, busca en:
        ~/.cache/fasttext/cc.es.300.bin

    Ejemplo
    -------
    ::

        from sklearn.pipeline import FeatureUnion
        union = FeatureUnion([
            ('manual', SentimentFeaturesTransformer()),
            ('ft',     FastTextFeaturesTransformer()),
        ])
    """

    _DEFAULT_PATH = Path.home() / '.cache' / 'fasttext' / 'cc.es.300.bin'
    N_FEATURES = 300

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else self._DEFAULT_PATH
        self._model = None   # Carga lazy

    def _load_model(self):
        """Carga el modelo fastText (solo la primera vez)."""
        if self._model is not None:
            return
        try:
            import fasttext  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Instala fastText: pip install fasttext-wheel"
            ) from exc

        if not self.model_path.exists():
            logger.info(
                'Modelo fastText no encontrado en %s. '
                'Descargando via fasttext.util…',
                self.model_path
            )
            print(
                f'[FastTextFeaturesTransformer] Descargando cc.es.300.bin '
                f'(~4.2 GB) en {self.model_path.parent}…'
            )
            import fasttext.util  # noqa: PLC0415
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            cwd = os.getcwd()
            os.chdir(str(self.model_path.parent))
            fasttext.util.download_model('es', if_exists='ignore')
            os.chdir(cwd)

        print(f'[FastTextFeaturesTransformer] Cargando modelo de {self.model_path}…')
        self._model = fasttext.load_model(str(self.model_path))
        print('[FastTextFeaturesTransformer] ✓ Modelo cargado.')

    def fit(self, X: Any = None, y: Any = None) -> 'FastTextFeaturesTransformer':
        """Carga el modelo durante fit() si aún no se ha hecho."""
        self._load_model()
        return self

    def transform(
        self,
        X: Union[list, tuple],
        y: Any = None,
    ) -> np.ndarray:
        """Transforma textos en vectores de 300 dimensiones.

        Cada texto se representa como la media de los vectores de sus tokens.
        Los tokens OOV obtienen su vector fastText (subword-based, nunca es cero).

        Parámetros
        ----------
        X : array-like de str
            Textos preprocesados o crudos.

        Retorna
        -------
        np.ndarray de shape (n_samples, 300)
        """
        if self._model is None:
            self._load_model()

        vectors = []
        for text in X:
            if not text or not isinstance(text, str):
                vectors.append(np.zeros(self.N_FEATURES))
            else:
                # get_sentence_vector calcula la media de los tokens del texto
                try:
                    vec = self._model.get_sentence_vector(text.strip())
                    vectors.append(vec)
                except Exception as exc:
                    logger.warning('FastTextFeaturesTransformer.transform: %s', exc)
                    vectors.append(np.zeros(self.N_FEATURES))
        return np.array(vectors, dtype=np.float32)

    def get_feature_names_out(self) -> List[str]:
        """Retorna nombres para los 300 features."""
        return [f'ft_{i}' for i in range(self.N_FEATURES)]

    def get_feature_names(self) -> List[str]:
        """Alias retrocompatible."""
        return self.get_feature_names_out()


# =====================================================================
# SpaCyVectorTransformer — Embeddings semánticos via spaCy (ligero)
# =====================================================================

class SpaCyVectorTransformer(BaseEstimator, TransformerMixin):
    """Embeddings semánticos pre-entrenados via spaCy (es_core_news_md).

    Genera 300 features densos por texto usando vectores de palabras
    pre-entrenados (word2vec 300-dim, 20K vocabulario).

    **Arquitectura:** Usa el texto ORIGINAL (via ``_raw_registry``) para
    calcular vectores de palabras reales, no de tokens procesados como
    ``neg_gustar`` o ``emoji_pos`` que no existen en el vocabulario
    de word2vec.

    Esto captura **similitud semántica** que TF-IDF no puede:
    - "genial" ≈ "fantástico" ≈ "increíble" (sinónimos)
    - "odio" ≈ "desprecio" ≈ "asco" (campo semántico)
    - Relaciones de contexto aprendidas de un corpus grande

    Parámetros
    ----------
    model_name : str, default='es_core_news_md'
        Modelo spaCy con vectores estáticos (md o lg).
    use_raw_text : bool, default=True
        Si True, busca el texto original en ``_raw_registry`` para
        obtener vectores de palabras reales.

    Ejemplo
    -------
    ::

        from sklearn.pipeline import FeatureUnion
        union = FeatureUnion([
            ('manual', SentimentFeaturesTransformer()),
            ('semantic', SpaCyVectorTransformer()),
        ])
    """

    N_FEATURES = 300  # Dimensión de los vectores de es_core_news_md

    # Cache a nivel de clase: se comparte entre instancias y sobrevive a clone()
    _model_cache: dict = {}

    def __init__(
        self,
        model_name: str = 'es_core_news_md',
        use_raw_text: bool = True,
    ):
        self.model_name = model_name
        self.use_raw_text = use_raw_text

    @property
    def _nlp(self):
        """Acceso al modelo cacheado a nivel de clase."""
        return SpaCyVectorTransformer._model_cache.get(self.model_name)

    def _ensure_loaded(self):
        """Carga lazy del modelo spaCy con vectores (cacheado a nivel de clase)."""
        if self._nlp is not None:
            return
        try:
            import spacy as _spacy
            nlp = _spacy.load(
                self.model_name,
                disable=['ner', 'parser', 'attribute_ruler', 'lemmatizer'],
            )
            SpaCyVectorTransformer._model_cache[self.model_name] = nlp
            vlen = self._nlp.vocab.vectors_length
            if vlen == 0:
                logger.warning(
                    'Modelo %s no tiene vectores estáticos. '
                    'Instala es_core_news_md o es_core_news_lg.',
                    self.model_name,
                )
            else:
                logger.info(
                    'SpaCyVectorTransformer: modelo %s cargado (%d vectores, %d dims)',
                    self.model_name,
                    self._nlp.vocab.vectors.shape[0],
                    vlen,
                )
        except OSError as exc:
            raise RuntimeError(
                f'Modelo spaCy "{self.model_name}" no instalado. '
                f'Ejecuta: python -m spacy download {self.model_name}'
            ) from exc

    def fit(self, X: Any = None, y: Any = None) -> 'SpaCyVectorTransformer':
        """Carga el modelo durante fit() (lazy)."""
        self._ensure_loaded()
        return self

    def transform(
        self,
        X: Union[list, tuple],
        y: Any = None,
    ) -> np.ndarray:
        """Transforma textos en vectores semánticos de 300 dimensiones.

        Para cada texto:
        1. Busca el texto original en ``_raw_registry`` (si ``use_raw_text=True``)
        2. Procesa con spaCy para obtener vectores de token
        3. Calcula la media de los vectores → vector del documento

        Los vectores capturan semántica de palabras reales (no tokens procesados),
        lo que permite al clasificador generalizar a sinónimos y contextos similares.

        Parámetros
        ----------
        X : array-like de str
            Textos (procesados o crudos).

        Retorna
        -------
        np.ndarray de shape (n_samples, 300)
        """
        if self._nlp is None:
            self._ensure_loaded()

        vectors = []
        for text in X:
            # Usar texto original para mejores vectores semánticos
            raw = _raw_registry.get(text) if self.use_raw_text else None
            source = raw if raw else text

            if not source or not isinstance(source, str):
                vectors.append(np.zeros(self.N_FEATURES, dtype=np.float32))
                continue

            # Truncar a 512 chars para eficiencia (tweets son cortos)
            doc = self._nlp(source[:512])
            vec = doc.vector

            if np.any(vec):
                # ── L2-normalización: lleva el vector a norma unitaria.
                # Garantiza que todos los tweets tienen la misma escala
                # independientemente de su longitud o vocabulario.
                # Sin esto, tweets con palabras de alta norma dominan
                # el espacio de features y degradan la precisión.
                norm = np.linalg.norm(vec)
                vec_norm = (vec / norm).astype(np.float32) if norm > 1e-8 else np.zeros(
                    self.N_FEATURES, dtype=np.float32
                )
                vectors.append(vec_norm)
            else:
                vectors.append(np.zeros(self.N_FEATURES, dtype=np.float32))

        return np.array(vectors, dtype=np.float32)

    def get_feature_names_out(self) -> List[str]:
        """Nombres para los 300 features semánticos."""
        return [f'spacy_v{i}' for i in range(self.N_FEATURES)]

    def get_feature_names(self) -> List[str]:
        """Alias retrocompatible."""
        return self.get_feature_names_out()
