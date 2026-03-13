"""Test rápido de los 3 fixes del pipeline V4."""
import sys, os
# Asegurar que logic/ sea importable
_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'courseNLP', 'Actividades')
sys.path.insert(0, _base)
os.chdir(_base)

# Test 1: Importación limpia
from logic.text_processing import TextProcessing, strip_accents
from logic.feature_extraction import SentimentFeaturesTransformer, FEATURE_NAMES, _SHARED_RAW_LOOKUP
from logic.lexicon_es import POSITIVE_WORDS, NEGATIVE_WORDS, POSITIVE_BIGRAMS, NEGATIVE_BIGRAMS, NEGATORS

print('=' * 70)
print('TEST 1: Importaciones')
print('=' * 70)
print(f'  POSITIVE_WORDS: {len(POSITIVE_WORDS)} (incluye adorar: {"adorar" in POSITIVE_WORDS})')
print(f'  NEGATIVE_WORDS: {len(NEGATIVE_WORDS)} (incluye doler: {"doler" in NEGATIVE_WORDS})')
print(f'  adorer en POSITIVE: {"adorer" in POSITIVE_WORDS} (debe ser False)')

# Test 2: Pipeline V4 — strip_accents después de spaCy
print('\n' + '=' * 70)
print('TEST 2: Pipeline V4')
print('=' * 70)
tp = TextProcessing(lang='es', apply_lemma=True, apply_stemming=True,
                     apply_stopwords=True, apply_spell=True, neg_window=4)

cases = [
    ('Me encantó la película', 'lema encantar debe aparecer'),
    ('Es horrible y me molesta', 'me/es sobreviven stopwords'),
    ('No me gusta nada esto', 'no_me dispara negación'),
    ('Qué asco de servicio', 'que sobrevive stopwords'),
    ('Esto es una pregunta?', 'preserva ? para is_question'),
]

for text, desc in cases:
    cleaned = tp.transformer(text)
    print(f'\n  [{desc}]')
    print(f'  INPUT : {text}')
    print(f'  OUTPUT: {cleaned}')

# Test 3: Features con raw lookup
print('\n' + '=' * 70)
print('TEST 3: Features con raw_text_lookup_')
print('=' * 70)
raw_texts = [c[0] for c in cases]
cleaned_texts = tp.transform(raw_texts)

import logic.feature_extraction as fe_mod
fe_mod._SHARED_RAW_LOOKUP.update(tp.raw_text_lookup_)

ft = SentimentFeaturesTransformer()
features = ft.transform(cleaned_texts)
print(f'  Features shape: {features.shape}')

for i, (raw, clean) in enumerate(zip(raw_texts, cleaned_texts)):
    nonzero = [(n, round(float(v), 3)) for n, v in zip(FEATURE_NAMES, features[i]) if abs(v) > 0.001]
    print(f'\n  "{raw[:50]}"')
    print(f'  → {nonzero}')

# Test 4: Verificar que bigramas funcionan
print('\n' + '=' * 70)
print('TEST 4: Bigramas léxicos')
print('=' * 70)
test_bigrams = [
    'Me encanta este lugar',
    'Que asco de comida',
    'Te quiero mucho',
    'Es horrible todo',
    'Estoy harto de esto',
]
for text in test_bigrams:
    cleaned = tp.transformer(text)
    tokens = cleaned.split() if cleaned else []
    bigrams = ['_'.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    pos_hits = [b for b in bigrams if b in POSITIVE_BIGRAMS]
    neg_hits = [b for b in bigrams if b in NEGATIVE_BIGRAMS]
    print(f'  "{text}" → tokens: {tokens}')
    print(f'    bigrams: {bigrams}')
    if pos_hits: print(f'    ✅ POS bigrams: {pos_hits}')
    if neg_hits: print(f'    ✅ NEG bigrams: {neg_hits}')
    if not pos_hits and not neg_hits: print(f'    ⚠️  No bigram hits')

print('\n✅ Todos los tests completados')
