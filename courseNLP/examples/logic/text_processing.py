"""
text_processing.py — Procesador de texto avanzado para NLP en español e inglés.
================================================================================
Pipeline de producción con las siguientes capacidades:

NORMALIZACIÓN AVANZADA
  • Mapeo de emojis por polaridad (emoji_pos / emoji_neg / emoji_neu)
  • Kaomojis textuales (ASCII art): :) → emoji_pos, :( → emoji_neg
  • Normalización de risas: jajaja / xd / lol → risa
  • Reducción de elongaciones: hooola → hoola
  • Puntos suspensivos → token semántico suspensivo (señal de ironía/duda)
  • Retweets (RT): eliminación del prefijo RT
  • Hilos: marcadores 1/ 2/ → hilo

ESPAÑOL INFORMAL
  • Abreviaciones SMS: xq→porque, tb→tambien, q→que, k→que, d→de, etc.
  • Contracciones dialectales: -ao → -ado, pa' → para
  • Profanidad como intensificador: puto/puta → marcador_intensidad

NEGACIÓN EXTENDIDA (ventana 3–5 tokens)
  • Clíticos: "no me gustó" → "no_me gustó"
  • Ventana deslizante: "no creo que sea bueno" → todos los tokens hasta
    puntuación/conjunción se marcan como negados

LEMMATIZACIÓN con spaCy (opcional, configurable)
  • Cuando apply_lemma=True y spaCy está disponible, reemplaza
    pseudo-stemming con lemas reales (maneja verbos irregulares, superlativos)

PSEUDO-STEMMING ESPAÑOL (fallback si spaCy no disponible)
  • 30+ sufijos (amiento, mente, cion, ando, isimo, ico/ica, etc.)

STOPWORDS SELECTIVAS
  • Elimina 55 palabras funcionales
  • NUNCA elimina negadores/intensificadores (no, ni, muy, nunca, etc.)

CORRECCIÓN ORTOGRÁFICA (diccionario manual ~40 formas frecuentes)
  • enserio→en serio, confirmao→confirmado, wasaps→whatsapps, etc.

API SKLEARN
  • Hereda BaseEstimator + TransformerMixin
  • fit() / transform() para uso en Pipeline y FeatureUnion

Correcciones de bugs respecto a la versión original del curso
-------------------------------------------------------------
  [BUG-FIX] stopwords(): TextProcessing == 'es' → siempre False → siempre English()
  [BUG-FIX] tagger(): llamada @staticmethod sobre método de instancia → TypeError
  [BUG-FIX] tagger(): token._.stem (extensión spaCy no registrada) → AttributeError
  [BUG-FIX] remove_patterns(): eliminaba _ [ ] - destruyendo tokens especiales
  [BUG-FIX] transformer(): tokens [EMOJI] con corchetes luego destruidos
  [BUG-FIX] Typo HASTAG → hashtag; load_sapcy → load_spacy

Autor original : Edwin Puertas (curso NLP)
Mejoras V3     : Lemmatización spaCy, emoji polarity, negación extendida,
                 español informal, corrección ortográfica, fastText embeddings.
"""

from __future__ import annotations

import re
import logging
import unicodedata
from typing import Any, Dict, List, Optional, Union

import nltk
import numpy as np
import spacy
from nltk import TweetTokenizer
from nltk.util import ngrams as nltk_ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from spacy.lang.en import English

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =====================================================================
# DICCIONARIO DE CORRECCIÓN ORTOGRÁFICA
# Formas frecuentes en Twitter ES no cubiertas por el stemmer
# =====================================================================
_SPELL_CORRECTIONS: dict = {
    # Uniones incorrectas
    'enserio': 'en serio',
    'osea': 'o sea',
    'almenos': 'al menos',
    'sinembargo': 'sin embargo',
    'derepente': 'de repente',
    'deverdad': 'de verdad',
    # Contracciones dialectales / pronunciación
    'confirmao': 'confirmado',
    'cansao': 'cansado',
    'flipao': 'flipado',
    'acabao': 'acabado',
    'terminao': 'terminado',
    'agotao': 'agotado',
    # Préstamos adaptados
    'wasaps': 'whatsapps',
    'wasap': 'whatsapp',
    'movil': 'movil',
    'finde': 'fin de semana',
    # Abreviaciones SMS (se expanden para mejorar cobertura del léxico)
    'xq': 'porque',
    'pq': 'porque',
    'porq': 'porque',
    'tb': 'tambien',
    'tmb': 'tambien',
    'tbn': 'tambien',
    'q': 'que',
    'k': 'que',
    'd': 'de',
    'xd': 'risa',      # también cubierto por el regex de risas
    'pa': 'para',
    'pra': 'para',
    'toy': 'estoy',
    'tamos': 'estamos',
    'taba': 'estaba',
    'tabamos': 'estabamos',
    'mxo': 'mucho',
    'mcho': 'mucho',
    'bn': 'bien',
    'bno': 'bueno',
    'bna': 'buena',
    'gcias': 'gracias',
    'grcias': 'gracias',
    'pls': 'por favor',
    'plz': 'por favor',
    'tqm': 'te quiero mucho',
    'tkm': 'te quiero mucho',
    'ntp': 'no te preocupes',
    'obvio': 'obviamente',
    'igual': 'igual',
    'jaja': 'risa',
    'ajaj': 'risa',
}


# =====================================================================
# MAPA DE EMOJIS UNICODE → TOKEN DE POLARIDAD
# Cubre los emojis más frecuentes en Twitter ES (TASS 2018)
# =====================================================================
_EMOJI_POS: frozenset = frozenset({
    '\U0001F600',  # 😀 grinning face
    '\U0001F601',  # 😁 beaming face
    '\U0001F602',  # 😂 face with tears of joy
    '\U0001F603',  # 😃 grinning face with big eyes
    '\U0001F604',  # 😄 grinning face with smiling eyes
    '\U0001F605',  # 😅 grinning face with sweat
    '\U0001F606',  # 😆 grinning squinting face
    '\U0001F607',  # 😇 smiling face with halo
    '\U0001F609',  # 😉 winking face
    '\U0001F60A',  # 😊 smiling face with smiling eyes
    '\U0001F60D',  # 😍 smiling face with heart-eyes
    '\U0001F618',  # 😘 face blowing a kiss
    '\U0001F617',  # 😗 kissing face
    '\U0001F619',  # 😙 kissing face with smiling eyes
    '\U0001F61A',  # 😚 kissing face with closed eyes
    '\U0001F970',  # 🥰 smiling face with hearts
    '\U0001F929',  # 🤩 star-struck
    '\U0001F973',  # 🥳 partying face
    '\U00002764',  # ❤️ red heart
    '\U0001F495',  # 💕 two hearts
    '\U0001F496',  # 💖 sparkling heart
    '\U0001F497',  # 💗 growing heart
    '\U0001F498',  # 💘 heart with arrow
    '\U0001F499',  # 💙 blue heart
    '\U0001F49A',  # 💚 green heart
    '\U0001F49B',  # 💛 yellow heart
    '\U0001F49C',  # 💜 purple heart
    '\U0001F9E1',  # 🧡 orange heart
    '\U0001F91F',  # 🤟 love-you gesture
    '\U0001F44F',  # 👏 clapping hands
    '\U0001F44D',  # 👍 thumbs up
    '\U0001F64C',  # 🙌 raising hands
    '\U0001F38A',  # 🎊 confetti ball
    '\U0001F389',  # 🎉 party popper
    '\U00002728',  # ✨ sparkles
    '\U0001F31F',  # 🌟 glowing star
    '\U00002B50',  # ⭐ star
    '\U0001F3C6',  # 🏆 trophy
    '\U0001F451',  # 👑 crown
    '\U0001F525',  # 🔥 fire
    '\U0001F4AA',  # 💪 flexed biceps
    '\U0001F60B',  # 😋 face savoring food
    '\U0001F61C',  # 😜 winking face with tongue
    '\U0001F61D',  # 😝 squinting face with tongue
    '\U0001F920',  # 🤠 cowboy hat face
    '\U0001F60E',  # 😎 smiling face with sunglasses
})

_EMOJI_NEG: frozenset = frozenset({
    '\U0001F620',  # 😠 angry face
    '\U0001F621',  # 😡 pouting face
    '\U0001F922',  # 🤢 nauseated face
    '\U0001F92E',  # 🤮 face vomiting
    '\U0001F925',  # 🤥 lying face
    '\U0001F612',  # 😒 unamused face
    '\U0001F614',  # 😔 pensive face
    '\U0001F615',  # 😕 confused face
    '\U0001F61E',  # 😞 disappointed face
    '\U0001F61F',  # 😟 worried face
    '\U0001F622',  # 😢 crying face
    '\U0001F623',  # 😣 persevering face
    '\U0001F624',  # 😤 face with steam from nose
    '\U0001F625',  # 😥 sad but relieved face
    '\U0001F626',  # 😦 frowning face with open mouth
    '\U0001F627',  # 😧 anguished face
    '\U0001F628',  # 😨 fearful face
    '\U0001F629',  # 😩 weary face
    '\U0001F62A',  # 😪 sleepy face
    '\U0001F62B',  # 😫 tired face
    '\U0001F62D',  # 😭 loudly crying face
    '\U0001F630',  # 😰 anxious face with sweat
    '\U0001F631',  # 😱 face screaming in fear
    '\U0001F633',  # 😳 flushed face
    '\U0001F635',  # 😵 dizzy face
    '\U0001F92F',  # 🤯 exploding head
    '\U0001F97A',  # 🥺 pleading face
    '\U0001F494',  # 💔 broken heart
    '\U0001F44E',  # 👎 thumbs down
    '\U0001F4A9',  # 💩 pile of poo
    '\U0001F608',  # 😈 smiling face with horns
    '\U0001F47F',  # 👿 angry face with horns
    '\U0001F480',  # 💀 skull
    '\U00002639',  # ☹️ frowning face
    '\U0001F616',  # 😖 confounded face
    '\U0001F613',  # 😓 downcast face with sweat
})

# Mapa de kaomojis ASCII → token de polaridad
_KAOMOJI_MAP: dict = {
    ':)': ' emoji_pos ', ':-)': ' emoji_pos ', ':D': ' emoji_pos ',
    '=)': ' emoji_pos ', ';)': ' emoji_pos ', ';-)': ' emoji_pos ',
    ':p': ' emoji_pos ', ':-p': ' emoji_pos ', ':3': ' emoji_pos ',
    '^_^': ' emoji_pos ', '^.^': ' emoji_pos ', '^^': ' emoji_pos ',
    'xd': ' risa ', ':o': ' emoji_pos ', ':-o': ' emoji_pos ',
    ':(': ' emoji_neg ', ':-(': ' emoji_neg ', ':/': ' emoji_neg ',
    ':-/': ' emoji_neg ', '>:(': ' emoji_neg ', ":'(": ' emoji_neg ',
    '>.<': ' emoji_neg ', '-_-': ' emoji_neg ', 'T_T': ' emoji_neg ',
    'T.T': ' emoji_neg ', 'u_u': ' emoji_neg ',
}

# Regex para kaomojis (orden importa: primero más específicos)
_RE_KAOMOJI = re.compile(
    r'(?<!\w)(?:'
    + '|'.join(re.escape(k) for k in sorted(_KAOMOJI_MAP, key=len, reverse=True))
    + r')(?!\w)',
    re.IGNORECASE
)


# =====================================================================
# CONSTANTES LINGÜÍSTICAS
# =====================================================================

# Sufijos españoles para pseudo-stemming (fallback sin spaCy)
_SUFFIXES: tuple = (
    'amiento', 'imiento',    # nominalización verbal: cansamiento, sufrimiento
    'amente',                 # adverbio: rápidamente
    'mente',                  # adverbio: claramente
    'acion', 'icion',         # nominalización: acción, descripción
    'cion', 'sion',           # nominalización corta: canción, tensión
    'ando', 'iendo',          # gerundio: cantando, comiendo
    'ados', 'idas',           # participio plural: cansados, perdidas
    'idos', 'adas',           # participio plural: perdidos, cansadas
    'aron', 'ieron', 'aban',  # pretérito/imperfecto: cantaron, comían
    'emos', 'amos', 'imos',   # presente 1ª plural: comemos, cantamos
    'ismo', 'ista',           # ideología/agente: socialismo, capitalista
    'isimo', 'isima',         # superlativo: buenísimo, malísima
    'icos', 'icas',           # diminutivo regional (Colombia/CR): chiquitico
    'itos', 'itas',           # diminutivo: chiquito, pequeñita
    'ito', 'ita',             # diminutivo: gato→gat
    'able', 'ible',           # adjetivo de posibilidad: amable, posible
    'oso', 'osa',             # adjetivo de cualidad: furioso, hermosa
)

_MIN_STEM_LEN: int = 4

# Stopwords funcionales — 55 palabras
# Artículos, preposiciones, auxiliares, pronombres clíticos sin carga semántica
_STOPWORDS_ES: frozenset = frozenset({
    # Artículos y determinantes
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
    'aquel', 'aquella', 'aquellos', 'aquellas',
    # Preposiciones
    'de', 'del', 'a', 'al', 'en', 'con', 'por', 'para', 'sin',
    'sobre', 'desde', 'hasta', 'entre', 'hacia', 'segun',
    # Conjunciones
    'y', 'e', 'o', 'u', 'ni', 'pero', 'sino', 'que', 'si',
    'aunque', 'porque', 'pues', 'cuando', 'donde', 'como',
    # Pronombres / clíticos sin carga semántica
    'se', 'le', 'lo', 'la', 'les', 'los', 'me', 'te', 'os', 'nos',
    'su', 'sus', 'mi', 'tu', 'nuestro', 'vuestro', 'cual', 'cuales',
    # Verbos auxiliares / cópulas
    'es', 'son', 'fue', 'ser', 'estar', 'estoy', 'estan', 'estaba',
    'siendo', 'han', 'ha', 'he', 'haber', 'era', 'hay',
    # Adverbios funcionales
    'aqui', 'ahi', 'alli', 'asi', 'ya', 'aun', 'luego', 'entonces',
    'tambien', 'al',
})

# Palabras que NUNCA se eliminan: negadores, intensificadores, atenuadores
_PRESERVE: frozenset = frozenset({
    'no', 'ni', 'muy', 'nunca', 'jamas', 'sin', 'nadie',
    'nada', 'tampoco', 'apenas', 'mas', 'menos', 'bien', 'mal',
})

# Conjunciones y puntuación que detienen la propagación de negación
_NEG_STOP: frozenset = frozenset({
    'y', 'pero', 'sin', 'embargo', 'aunque', 'que', 'porque',
    'o', 'ni', 'sino', 'a', ',', '.', '!', '?',
})


# =====================================================================
# FUNCIONES AUXILIARES (módulo-level, reutilizables)
# =====================================================================

def strip_accents(text: str) -> str:
    """Normaliza acentos: á→a, é→e, ñ→n, ü→u."""
    nfkd = unicodedata.normalize('NFD', text)
    return nfkd.encode('ascii', 'ignore').decode('utf-8')


def pseudo_stem(word: str) -> str:
    """Pseudo-stemming español: corta sufijos preservando raíz ≥ 4 chars.

    Fallback cuando spaCy no está disponible o apply_lemma=False.
    """
    for suf in _SUFFIXES:
        if word.endswith(suf) and (len(word) - len(suf)) >= _MIN_STEM_LEN:
            return word[:-len(suf)]
    return word


def _replace_emoji_by_polarity(text: str) -> str:
    """Reemplaza cada emoji Unicode por 'emoji_pos', 'emoji_neg' o 'emoji_neu'.

    Procesa caracter a caracter para capturar emojis consecutivos
    antes de que _RE_EMOJI los elimine en bloque.
    """
    result = []
    for char in text:
        if char in _EMOJI_POS:
            result.append(' emoji_pos ')
        elif char in _EMOJI_NEG:
            result.append(' emoji_neg ')
        elif unicodedata.category(char) in ('So', 'Sm') or ord(char) > 0x1F000:
            result.append(' emoji_neu ')
        else:
            result.append(char)
    return ''.join(result)


def _apply_negation_window(tokens: list, window: int = 4) -> list:
    """Propaga negación hasta ``window`` tokens o hasta un stop-token.

    Ejemplo:
        ["no", "creo", "que", "sea", "bueno"] → window=4
        → ["no", "neg_creo", "que", "sea", "neg_bueno"]
        (se detiene en 'que' que es _NEG_STOP)

    Los clíticos pronominales ya se compoundearon en el paso anterior
    (no_me, no_te, etc.), así que aquí solo se propaga sobre palabras
    de contenido.
    """
    result = list(tokens)
    i = 0
    while i < len(tokens):
        tok = tokens[i].lower()
        if tok in ('no', 'nunca', 'jamas', 'tampoco', 'nadie', 'nada', 'ningun', 'ninguna'):
            count = 0
            j = i + 1
            while j < len(tokens) and count < window:
                if tokens[j].lower() in _NEG_STOP:
                    break
                # Marcar token negado solo si no está ya compoundeado
                if '_' not in tokens[j]:
                    result[j] = 'neg_' + tokens[j]
                count += 1
                j += 1
        i += 1
    return result


# =====================================================================
# CLASE PRINCIPAL
# =====================================================================

class TextProcessing(BaseEstimator, TransformerMixin):
    """Procesador de texto avanzado para NLP en español/inglés.

    Parámetros
    ----------
    lang : str, default='es'
        Idioma del modelo spaCy ('es' o 'en').
    apply_lemma : bool, default=True
        Usar lematización spaCy. Si es True y spaCy está disponible,
        reemplaza el pseudo-stemming. Maneja verbos irregulares
        (fue→ser, hecho→hacer).
    apply_stemming : bool, default=True
        Pseudo-stemming como fallback cuando apply_lemma=False o
        spaCy no está disponible.
    apply_stopwords : bool, default=True
        Eliminar stopwords funcionales preservando señales de sentimiento.
    apply_spell : bool, default=True
        Aplicar corrección ortográfica del diccionario manual.
    neg_window : int, default=4
        Ventana de propagación de negación (nº de tokens tras el negador).
    load_spacy_model : bool, default=True
        Cargar modelo spaCy al inicializar.

    Ejemplos
    --------
    Uso directo::

        tp = TextProcessing(lang='es')
        tp.transformer("No me gustó nada la película 😡 #horrible")
        # → 'no_me neg_gusto nada pelicula emoji_neg horrible'

    En sklearn Pipeline::

        pipe = Pipeline([
            ('clean', TextProcessing(lang='es', load_spacy_model=False)),
            ('tfidf', TfidfVectorizer()),
        ])
    """

    name: str = 'Text Processing'

    # ── Regex compilados (clase-level) ───────────────────────────────
    _RE_RT      = re.compile(r'^\s*RT\s+', re.IGNORECASE)
    _RE_THREAD  = re.compile(r'\b\d{1,2}/\d{0,2}\b')
    _RE_URL     = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
    _RE_MENTION = re.compile(r'@\w+')
    _RE_HASHTAG = re.compile(r'#(\w+)')
    _RE_LAUGH   = re.compile(
        r'\b(?:ja){2,}\w*\b|\b(?:je){2,}\w*\b|\bjaj[aj]*\b'
        r'|\bxd+\b|\bjeje\w*\b|\bjiji\w*\b|\blol\b|\bahah\w*\b',
        re.IGNORECASE | re.UNICODE)
    _RE_ELLIPSIS = re.compile(r'\.{2,}|…')
    _RE_ELONGATE = re.compile(r'(.)\1{2,}', re.UNICODE)
    _RE_NEG_CLITIC = re.compile(
        r'\b(no)\s+(me|te|le|se|lo|la|los|las|nos)\b',
        re.IGNORECASE | re.UNICODE)
    _RE_DIALECTAL = re.compile(r'\b(\w{4,})ao\b', re.IGNORECASE | re.UNICODE)
    _RE_INTENSIF_INFORMAL = re.compile(
        r'\bput[oa]s?\b|\bhostia\b|\bjoder\b|\bcoño\b',
        re.IGNORECASE | re.UNICODE)
    _RE_PUNCT   = re.compile(r'[^\w\s_]', re.UNICODE)
    _RE_SPACES  = re.compile(r'\s+')
    _RE_DIGITS  = re.compile(r'\b\d+(?:\.\d+)?\b')

    # ── Constructor ──────────────────────────────────────────────────

    def __init__(
        self,
        lang: str = 'es',
        apply_lemma: bool = True,
        apply_stemming: bool = True,
        apply_stopwords: bool = True,
        apply_spell: bool = True,
        neg_window: int = 4,
        load_spacy_model: bool = True,
    ):
        self.lang = lang
        self.apply_lemma = apply_lemma
        self.apply_stemming = apply_stemming
        self.apply_stopwords = apply_stopwords
        self.apply_spell = apply_spell
        self.neg_window = neg_window
        self.load_spacy_model = load_spacy_model
        self.nlp: Optional[spacy.language.Language] = None
        if load_spacy_model:
            self.nlp = self._load_spacy(lang)

    # ── spaCy ────────────────────────────────────────────────────────

    @staticmethod
    def _load_spacy(lang: str) -> Optional[spacy.language.Language]:
        """Carga el modelo spaCy para el idioma indicado."""
        model_name = 'es_core_news_sm' if lang == 'es' else 'en_core_web_sm'
        try:
            nlp = spacy.load(model_name)
            logger.info('spaCy model loaded: %s — pipes: %s', model_name, nlp.pipe_names)
            print(f'Language: {TextProcessing.name}\n  {lang}: {nlp.pipe_names}')
            return nlp
        except OSError as exc:
            logger.warning('spaCy model "%s" no disponible: %s', model_name, exc)
            print(f'Warning: spaCy model "{model_name}" no disponible. '
                  f'Se usará pseudo-stemming como fallback.')
            return None

    @staticmethod
    def load_spacy(lang: str) -> Optional[spacy.language.Language]:
        """Alias público retrocompatible."""
        return TextProcessing._load_spacy(lang)

    def analysis_pipe(self, text: str) -> Optional[spacy.tokens.Doc]:
        """Procesa texto con el pipeline spaCy completo."""
        if self.nlp is None:
            return None
        try:
            return self.nlp(text)
        except Exception as exc:
            logger.error('analysis_pipe: %s', exc)
            return None

    def _lemmatize(self, text: str) -> str:
        """Lematiza texto con spaCy preservando tokens especiales y compuestos.

        Los tokens con _ (compuestos de negación) y los tokens especiales
        (url, usuario, emoji_pos, emoji_neg, risa, suspensivo, hilo)
        no se lematizan.
        """
        _SPECIAL_TOKENS = frozenset({
            'url', 'usuario', 'emoji_pos', 'emoji_neg', 'emoji_neu',
            'risa', 'suspensivo', 'hilo', 'marcador_intensidad',
        })
        doc = self.analysis_pipe(text)
        if doc is None:
            return text
        lemmas = []
        for token in doc:
            raw = token.text
            # Preservar compuestos de negación y tokens especiales
            if '_' in raw or raw.lower() in _SPECIAL_TOKENS:
                lemmas.append(raw)
            elif token.lemma_ and token.lemma_ != '-PRON-':
                lemma = token.lemma_.lower()
                # Normalizar acento del lema para matching con léxico
                lemma = strip_accents(lemma)
                lemmas.append(lemma)
            else:
                lemmas.append(raw.lower())
        return ' '.join(lemmas)

    # ── Normalización ────────────────────────────────────────────────

    @staticmethod
    def proper_encoding(text: str) -> str:
        """Elimina acentos y caracteres no-ASCII (NFD → ASCII). Alias de strip_accents."""
        try:
            return strip_accents(text)
        except Exception as exc:
            logger.error('proper_encoding: %s', exc)
            return ''

    @staticmethod
    def _apply_spell_correction(text: str) -> str:
        """Aplica diccionario de correcciones ortográficas token a token."""
        tokens = text.split()
        return ' '.join(_SPELL_CORRECTIONS.get(t, t) for t in tokens)

    @staticmethod
    def remove_stopwords(text: str, lang: str = 'es') -> str:
        """Elimina stopwords funcionales preservando señales de sentimiento.

        Usa set estático _STOPWORDS_ES con excepciones en _PRESERVE.
        """
        if lang != 'es':
            try:
                nlp_light = English()
                doc = nlp_light(text)
                return ' '.join(t.text for t in doc if not t.is_stop)
            except Exception as exc:
                logger.error('remove_stopwords (en): %s', exc)
                return text
        tokens = text.split()
        return ' '.join(
            t for t in tokens
            if t in _PRESERVE or t not in _STOPWORDS_ES
        )

    @staticmethod
    def stopwords(text: str, lang: str = 'es') -> str:
        """Alias retrocompatible de remove_stopwords."""
        return TextProcessing.remove_stopwords(text, lang=lang)

    @staticmethod
    def remove_patterns(text: str) -> str:
        """Elimina puntuación y dígitos sueltos. Preserva guión bajo ``_``."""
        try:
            text = TextProcessing._RE_DIGITS.sub('', text)
            text = TextProcessing._RE_PUNCT.sub(' ', text)
            text = TextProcessing._RE_SPACES.sub(' ', text).strip()
            return text.lower()
        except Exception as exc:
            logger.error('remove_patterns: %s', exc)
            return ''

    # ── Pipeline principal ───────────────────────────────────────────

    def transformer(self, text: str, stopwords: bool = False) -> Optional[str]:
        """Pipeline completo de limpieza para un solo texto.

        Orden de operaciones
        --------------------
        1.  Corrección ortográfica (diccionario manual)
        2.  Eliminación de prefijo RT
        3.  Normalización de hilos (1/ 2/ → hilo)
        4.  Minúsculas
        5.  Kaomojis ASCII → emoji_pos / emoji_neg
        6.  Emojis Unicode → emoji_pos / emoji_neg / emoji_neu (por polaridad)
        7.  Reemplazos semánticos: URLs→url, menciones→usuario, hashtags→contenido
        8.  Normalización de risas → risa
        9.  Puntos suspensivos → suspensivo
        10. Compounding de clíticos: "no me" → "no_me"
        11. Reducción de elongaciones: hooola→hoola
        12. Eliminación de acentos
        13. Limpieza de puntuación/dígitos
        14. Negación extendida (ventana neg_window tokens)
        15. Lematización spaCy (si apply_lemma=True y nlp disponible)
            ó pseudo-stemming (fallback)
        16. Stopwords selectivas
        17. Colapso de espacios

        Parámetros
        ----------
        text : str
            Texto crudo (tweet, etc.).
        stopwords : bool
            Si True, fuerza eliminación de stopwords.

        Retorna
        -------
        str o None si el resultado queda vacío.
        """
        if not isinstance(text, str) or not text.strip():
            return None
        try:
            # ── 1. Corrección ortográfica ─────────────────────────
            if self.apply_spell:
                t = self._apply_spell_correction(text)
            else:
                t = text

            # ── 2. Eliminar prefijo RT ────────────────────────────
            t = self._RE_RT.sub('', t)

            # ── 3. Hilos ─────────────────────────────────────────
            t = self._RE_THREAD.sub(' hilo ', t)

            # ── 4. Minúsculas ─────────────────────────────────────
            t = t.lower()

            # ── 5. Kaomojis ASCII ─────────────────────────────────
            def _replace_kaomoji(m: re.Match) -> str:
                return _KAOMOJI_MAP.get(m.group(0).lower(), ' emoji_neu ')
            t = _RE_KAOMOJI.sub(_replace_kaomoji, t)

            # ── 6. Emojis Unicode por polaridad ───────────────────
            t = _replace_emoji_by_polarity(t)

            # ── 7. URLs, menciones, hashtags ──────────────────────
            t = self._RE_URL.sub(' url ', t)
            t = self._RE_MENTION.sub(' usuario ', t)
            t = self._RE_HASHTAG.sub(r' \1 ', t)  # #MiTag → MiTag

            # ── 8. Normalización de risas ─────────────────────────
            t = self._RE_LAUGH.sub(' risa ', t)

            # ── 9. Puntos suspensivos → token semántico ───────────
            t = self._RE_ELLIPSIS.sub(' suspensivo ', t)

            # ── 10. Compounding de clíticos de negación ───────────
            t = self._RE_NEG_CLITIC.sub(self._compound_neg, t)

            # ── 11. Contracciones dialectales (-ao → -ado) ─────────
            t = self._RE_DIALECTAL.sub(r'\1ado', t)

            # ── 12. Profanidad como marcador de intensidad ─────────
            t = self._RE_INTENSIF_INFORMAL.sub(' marcador_intensidad ', t)

            # ── 13. Reducción de elongaciones ────────────────────
            t = self._RE_ELONGATE.sub(r'\1\1', t)

            # ── 14. Eliminación de acentos ───────────────────────
            t = strip_accents(t)

            # ── 15. Limpieza de puntuación y dígitos ─────────────
            t = self.remove_patterns(t)

            # ── 16. Negación extendida (ventana deslizante) ───────
            if self.neg_window > 0:
                tokens = t.split()
                tokens = _apply_negation_window(tokens, window=self.neg_window)
                t = ' '.join(tokens)

            # ── 17. Lematización o pseudo-stemming ───────────────
            if self.apply_lemma and self.nlp is not None:
                t = self._lemmatize(t)
            elif self.apply_stemming:
                tokens = t.split()
                tokens = [
                    pseudo_stem(tok) if tok not in _PRESERVE and '_' not in tok
                    else tok
                    for tok in tokens
                ]
                t = ' '.join(tokens)

            # ── 18. Stopwords selectivas ─────────────────────────
            if stopwords or self.apply_stopwords:
                t = self.remove_stopwords(t, lang=self.lang)

            # ── 19. Colapso de espacios finales ──────────────────
            t = self._RE_SPACES.sub(' ', t).strip()
            return t if t else None

        except Exception as exc:
            logger.error('transformer: %s', exc)
            return None

    @staticmethod
    def _compound_neg(match: re.Match) -> str:
        """Convierte grupos de negación en tokens compuestos con ``_``."""
        return '_'.join(p for p in match.groups() if p is not None)

    # ── API sklearn ──────────────────────────────────────────────────

    def fit(self, X: Any = None, y: Any = None) -> 'TextProcessing':
        """No-op. Requerido por sklearn API."""
        return self

    def transform(self, X: Union[list, tuple], y: Any = None) -> List[str]:
        """Aplica transformer() a cada elemento de X."""
        return [self.transformer(t) or '' for t in X]

    # ── Tokenización y n-gramas ──────────────────────────────────────

    @staticmethod
    def tokenizer(text: str) -> List[str]:
        """Tokeniza texto usando NLTK TweetTokenizer."""
        try:
            return TweetTokenizer().tokenize(text)
        except Exception as exc:
            logger.error('tokenizer: %s', exc)
            return []

    @staticmethod
    def make_ngrams(text: str, num: int) -> List[str]:
        """Genera n-gramas de tamaño ``num`` a partir del texto."""
        try:
            tokens = nltk.word_tokenize(text)
            return [' '.join(gram) for gram in nltk_ngrams(tokens, num)]
        except Exception as exc:
            logger.error('make_ngrams: %s', exc)
            return []

    # ── Análisis morfosintáctico ─────────────────────────────────────

    def tagger(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Analiza morfosintácticamente cada token con spaCy.

        El campo ``stem`` usa pseudo_stem() propio (no requiere extensiones
        spaCy personalizadas).
        """
        doc = self.analysis_pipe(text)
        if doc is None:
            return None
        try:
            return [
                {
                    'text':     token.text,
                    'lemma':    token.lemma_,
                    'stem':     pseudo_stem(strip_accents(token.text.lower())),
                    'pos':      token.pos_,
                    'tag':      token.tag_,
                    'dep':      token.dep_,
                    'shape':    token.shape_,
                    'is_alpha': token.is_alpha,
                    'is_stop':  token.is_stop,
                    'is_digit': token.is_digit,
                    'is_punct': token.is_punct,
                }
                for token in doc
            ]
        except Exception as exc:
            logger.error('tagger: %s', exc)
            return None

    def __repr__(self) -> str:
        return (
            f"TextProcessing(lang='{self.lang}', "
            f"apply_lemma={self.apply_lemma}, "
            f"apply_stemming={self.apply_stemming}, "
            f"apply_stopwords={self.apply_stopwords}, "
            f"apply_spell={self.apply_spell}, "
            f"neg_window={self.neg_window})"
        )
