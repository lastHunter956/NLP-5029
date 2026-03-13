"""
lexicon_es.py — Léxico de sentimientos para español informal (Twitter/TASS).
=============================================================================
Contiene 7 conjuntos léxicos ampliados y un diccionario de corrección
ortográfica para cubrir el registro informal de Twitter en español.

Cobertura aproximada
--------------------
  POSITIVE_WORDS  : ~360 palabras (estándar + coloquial ES/LATAM)
  NEGATIVE_WORDS  : ~380 palabras (estándar + coloquial ES/LATAM)
  NEGATORS        :  16 formas
  INTENSIFIERS    :  30 formas (incluye coloquialismo: tope, mogollon, puto)
  ATTENUATORS     :  14 formas
  POSITIVE_BIGRAMS:  28 bigramas frecuentes
  NEGATIVE_BIGRAMS:  30 bigramas frecuentes

Nota sobre POSITIVE_EMOJIS / NEGATIVE_EMOJIS
---------------------------------------------
Los tokens de emoji son generados por text_processing.py usando la librería
`emoji` (5225 emojis) con un sistema de 9 categorías semánticas:
  Positivos: emoji_amor, emoji_risa, emoji_celebra, emoji_intenso
  Negativos: emoji_tristeza, emoji_rabia, emoji_asco, emoji_miedo, emoji_rechaza
  Legado:    emoji_pos, emoji_neg (compatibilidad con versiones anteriores)
Los conjuntos POSITIVE_EMOJIS / NEGATIVE_EMOJIS contienen TODOS los tokens
de cada polaridad para que feature_extraction.py los cuente correctamente.

Nota sobre SPELL_CORRECTIONS
-----------------------------
Diccionario manual de ~42 formas frecuentes en Twitter ES.
Se aplica como primera operación en TextProcessing.transformer().
"""

from __future__ import annotations

# =====================================================================
# CORRECCIONES ORTOGRÁFICAS
# Reexportado aquí para que __init__.py pueda importarlo en un lugar
# central. text_processing.py tiene su propia copia para evitar
# dependencia circular en el arranque.
# =====================================================================
SPELL_CORRECTIONS: dict = {
    'enserio':    'en serio',
    'osea':       'o sea',
    'almenos':    'al menos',
    'sinembargo': 'sin embargo',
    'derepente':  'de repente',
    'deverdad':   'de verdad',
    'confirmao':  'confirmado',
    'cansao':     'cansado',
    'flipao':     'flipado',
    'acabao':     'acabado',
    'terminao':   'terminado',
    'agotao':     'agotado',
    'wasaps':     'whatsapps',
    'wasap':      'whatsapp',
    'finde':      'fin de semana',
    'xq':         'porque',
    'pq':         'porque',
    'porq':       'porque',
    'tb':         'tambien',
    'tmb':        'tambien',
    'tbn':        'tambien',
    'q':          'que',
    'k':          'que',
    'd':          'de',
    'pa':         'para',
    'pra':        'para',
    'toy':        'estoy',
    'tamos':      'estamos',
    'taba':       'estaba',
    'tabamos':    'estabamos',
    'mxo':        'mucho',
    'mcho':       'mucho',
    'bn':         'bien',
    'bno':        'bueno',
    'bna':        'buena',
    'gcias':      'gracias',
    'grcias':     'gracias',
    'pls':        'por favor',
    'plz':        'por favor',
    'tqm':        'te quiero mucho',
    'tkm':        'te quiero mucho',
    'ntp':        'no te preocupes',
}


# =====================================================================
# PALABRAS POSITIVAS (estándar + coloquial ES/LATAM)
# =====================================================================
POSITIVE_WORDS: frozenset = frozenset({
    # ── Básico universal ──────────────────────────────────────────
    'bueno', 'buena', 'buenos', 'buenas',
    'bien', 'mejor', 'optimo', 'optima',
    'excelente', 'excelentes',
    'perfecto', 'perfecta', 'perfectos', 'perfectas',
    'maravilloso', 'maravillosa', 'maravillosos', 'maravillosas',
    'magnifico', 'magnifica', 'magnificos', 'magnificas',
    'fantastico', 'fantastica', 'fantasticos', 'fantasticas',
    'increible', 'increibles',
    'espectacular', 'espectaculares',
    'extraordinario', 'extraordinaria', 'extraordinarios', 'extraordinarias',
    'fenomenal', 'fenomenales',
    'genial', 'geniales',
    'sublime', 'sublimes',
    'sobresaliente', 'sobresalientes',
    # ── Emociones positivas ───────────────────────────────────────
    'feliz', 'felices',
    'alegre', 'alegres',
    'contento', 'contenta', 'contentos', 'contentas',
    'emocionado', 'emocionada', 'emocionados', 'emocionadas',
    'encantado', 'encantada', 'encantados', 'encantadas',
    'satisfecho', 'satisfecha', 'satisfechos', 'satisfechas',
    'orgulloso', 'orgullosa', 'orgullosos', 'orgullosas',
    'agradecido', 'agradecida', 'agradecidos', 'agradecidas',
    'enamorado', 'enamorada', 'enamorados', 'enamoradas',
    'divertido', 'divertida', 'divertidos', 'divertidas',
    'entusiasmado', 'entusiasmada', 'entusiasmados', 'entusiasmadas',
    'ilusionado', 'ilusionada', 'ilusionados', 'ilusionadas',
    'tranquilo', 'tranquila', 'tranquilos', 'tranquilas',
    'animado', 'animada', 'animados', 'animadas',
    # ── Verbos positivos ──────────────────────────────────────────
    # Infinitivos (lemas spaCy): gustar, encantar, adorar, etc.
    'amar', 'querer', 'adorar', 'gustar', 'encantar',
    'disfrutar', 'apreciar', 'valorar', 'celebrar',
    'ganar', 'triunfar', 'lograr', 'conseguir',
    'mejorar', 'crecer', 'avanzar', 'progresar',
    'ayudar', 'colaborar', 'compartir',
    # Infinitivos adicionales que spaCy produce como lema
    'alegrar', 'ilusionar', 'animar', 'emocionar',
    'divertir', 'satisfacer', 'fascinar', 'maravillar',
    'brillar', 'inspirar', 'motivar', 'sorprender',
    # Formas conjugadas frecuentes (para textos sin lematizar)
    'encanta', 'encanto', 'encantaste', 'gusto', 'gusta',
    'amo', 'amas', 'ama', 'quiero', 'quieres', 'adoro', 'adora',
    # ── Cualidades positivas ──────────────────────────────────────
    'hermoso', 'hermosa', 'hermosos', 'hermosas',
    'lindo', 'linda', 'lindos', 'lindas',
    'bonito', 'bonita', 'bonitos', 'bonitas',
    'precioso', 'preciosa', 'preciosos', 'preciosas',
    'bello', 'bella', 'bellos', 'bellas',
    'elegante', 'elegantes',
    'gracioso', 'graciosa', 'graciosos', 'graciosas',
    'amable', 'amables',
    'simpático', 'simpatico', 'simpatica', 'simpaticos', 'simpaticas',
    'inteligente', 'inteligentes',
    'talentoso', 'talentosa', 'talentosos', 'talentosas',
    'creativo', 'creativa', 'creativos', 'creativas',
    'solidario', 'solidaria', 'solidarios', 'solidarias',
    'honesto', 'honesta', 'honestos', 'honestas',
    'fiel', 'fieles',
    'valiente', 'valientes',
    'diligente', 'diligentes',
    # ── Coloquial España (Twitter ES) ─────────────────────────────
    'mola', 'molar', 'molando', 'molon', 'molona',
    'guay', 'guais',
    'chulo', 'chula', 'chulos', 'chulas',
    'flipante', 'flipantes', 'flipar', 'flipado', 'flipada',
    'alucinante', 'alucinantes', 'alucinar',
    'brutal', 'brutales',
    'bestial', 'bestiales',
    'cañero', 'cañera', 'cañeros', 'cañeras',
    'pasada', 'pasadas',
    'crack', 'cracks',
    'geniazo', 'geniazo',
    'padrisimo', 'padrisima',
    'chevere', 'cheverisimo',
    'bacano', 'bacana', 'bacán',
    'buenazo', 'buenaza', 'buenazos', 'buenazas',
    'buenisimo', 'buenisima',
    'riquísimo', 'riquísima', 'riquísimos', 'riquísimas',
    'riquisimo', 'riquisima',
    'estuvo', 'estupendo', 'estupenda',
    # ── Coloquial LATAM (Twitter LATAM) ───────────────────────────
    'chido', 'chida', 'chidos', 'chidas',
    'padre', 'padres',
    'chingon', 'chingona', 'chingones', 'chingonas',
    'genio', 'genios',
    'pila', 'pilas',    # rápido/hábil en COL
    # ── Code-switching (frecuente en Twitter ES) ──────────────────
    'amazing', 'awesome', 'great', 'love', 'cool', 'nice', 'cute',
    'perfect', 'beautiful', 'wonderful', 'fantastic', 'incredible',
    # ── Otros ─────────────────────────────────────────────────────
    'gracias', 'felicidades', 'enhorabuena', 'bienvenido', 'bienvenida',
    'exito', 'exitos',
    'victoria', 'victorias',
    'campeón', 'campeon', 'campeona',
    'logro', 'logros',
    'premio', 'premios',
    'regalo', 'regalos',
    'sorpresa', 'sorpresas',
    'oportunidad', 'oportunidades',
    'esperanza', 'esperanzas',
    'amor', 'amores',
    'amistad', 'amistades',
    'familia', 'familias',
    'paz', 'armonia',
    'libertad',
    'vida',
    'salud',
    'fortuna',
    'sano', 'sana',
    'saludable', 'saludables',
    'dicha', 'dichas',
    'placer', 'placeres',
    'risas',
})


# =====================================================================
# PALABRAS NEGATIVAS (estándar + coloquial ES/LATAM)
# =====================================================================
NEGATIVE_WORDS: frozenset = frozenset({
    # ── Básico universal ──────────────────────────────────────────
    'malo', 'mala', 'malos', 'malas',
    'mal', 'peor', 'pésimo', 'pesimo', 'pesima', 'pesimos', 'pesimas',
    'terrible', 'terribles',
    'horrible', 'horribles',
    'espantoso', 'espantosa', 'espantosos', 'espantosas',
    'atroz', 'atroces',
    'nefasto', 'nefasta', 'nefastos', 'nefastas',
    'deplorable', 'deplorables',
    'lamentable', 'lamentables',
    'desastroso', 'desastrosa', 'desastrosos', 'desastrosas',
    'catastrófico', 'catastrofico', 'catastrófica', 'catastrofica',
    'fatal', 'fatales',
    'inaceptable', 'inaceptables',
    'insoportable', 'insoportables',
    'intolerable', 'intolerables',
    # ── Emociones negativas ───────────────────────────────────────
    'triste', 'tristes',
    'tristeza', 'tristezas',
    'deprimido', 'deprimida', 'deprimidos', 'deprimidas',
    'angustiado', 'angustiada', 'angustiados', 'angustiadas',
    'desesperado', 'desesperada', 'desesperados', 'desesperadas',
    'aburrido', 'aburrida', 'aburridos', 'aburridas',
    'agotado', 'agotada', 'agotados', 'agotadas',
    'cansado', 'cansada', 'cansados', 'cansadas',
    'enfadado', 'enfadada', 'enfadados', 'enfadadas',
    'enojado', 'enojada', 'enojados', 'enojadas',
    'frustrado', 'frustrada', 'frustrados', 'frustradas',
    'decepcionado', 'decepcionada', 'decepcionados', 'decepcionadas',
    'preocupado', 'preocupada', 'preocupados', 'preocupadas',
    'asustado', 'asustada', 'asustados', 'asustadas',
    'avergonzado', 'avergonzada', 'avergonzados', 'avergonzadas',
    'molesto', 'molesta', 'molestos', 'molestas',
    'harto', 'harta', 'hartos', 'hartas',
    'enfermo', 'enferma', 'enfermos', 'enfermas',
    # ── Verbos negativos ──────────────────────────────────────────
    # Infinitivos (lemas spaCy): odiar, doler, aburrir, etc.
    'odiar', 'detestar', 'aborrecer', 'despreciar',
    'molestar', 'fastidiar', 'irritar', 'enfadar', 'enojar',
    'sufrir', 'llorar', 'quejarse',
    'fracasar', 'perder', 'fallar', 'arruinar',
    # Infinitivos adicionales que spaCy produce como lema
    'doler', 'aburrir', 'agotar', 'cansar', 'frustrar',
    'decepcionar', 'preocupar', 'asustar', 'deprimir',
    'avergonzar', 'hartar', 'apenar', 'entristecer',
    'destruir', 'estropear', 'empeorar', 'hundir',
    # Formas conjugadas frecuentes (para textos sin lematizar)
    'odio', 'odias', 'odia', 'odian',
    'detesto', 'detestas', 'detesta', 'detestan',
    'duele', 'duelen', 'duelo',
    'extraño', 'extranas', 'extraña',
    # ── Sustantivos de concepto negativo ──────────────────────────
    'problema', 'problemas',
    'fallo', 'fallos', 'falla', 'fallas',
    'error', 'errores',
    'fracaso', 'fracasos',
    'pena', 'penas',
    'dolor', 'dolores',
    'sufrimiento', 'sufrimientos',
    'miedo', 'miedos',
    'miedo', 'miedos',
    'crisis', 'crisi',
    'desastre', 'desastres',
    'caos',
    'tragedia', 'tragedias',
    'accidente', 'accidentes',
    'muerte', 'muertes',
    'enfermedad', 'enfermedades',
    'daño', 'daños', 'daño', 'daños',
    'pérdida', 'perdida', 'pérdidas', 'perdidas',
    'mentira', 'mentiras',
    'traición', 'traicion', 'traiciones',
    'abuso', 'abusos',
    'violencia', 'violencias',
    'injusticia', 'injusticias',
    'corrupcion', 'corrupciones',
    'pobreza',
    'hambre',
    'soledad',
    # ── Coloquial España (Twitter ES) ─────────────────────────────
    'cutre', 'cutres',
    'cursi', 'cursis',
    'petardo', 'petarda', 'petardos', 'petardas',
    'peñazo', 'peñazos',
    'coñazo', 'coñazos',
    'tostón', 'toston', 'tostonazo',
    'pringado', 'pringada', 'pringados', 'pringadas',
    'paleto', 'paleta', 'paletos', 'paletas',
    'hortera', 'horteras',
    'guarrada', 'guarradas',
    'porqueria', 'porquerias',
    'asqueroso', 'asquerosa', 'asquerosos', 'asquerosas',
    'nauseabundo', 'nauseabunda',
    'malísimo', 'malisimo', 'malísima', 'malisima',
    'pesadez', 'pesadeces',
    'jodido', 'jodida', 'jodidos', 'jodidas',
    'cagada', 'cagadas',
    'cachivache', 'cachivaches',
    # ── Coloquial LATAM (Twitter LATAM) ───────────────────────────
    'culero', 'culera', 'culeros', 'culeras',
    'buey', 'güey',
    'cabrón', 'cabron', 'cabrona',
    'pendejo', 'pendeja', 'pendejos', 'pendejas',
    'idiota', 'idiotas',
    'tonto', 'tonta', 'tontos', 'tontas',
    'estupido', 'estupida', 'estupidos', 'estupidas',
    'necio', 'necia', 'necios', 'necias',
    # ── Profanidades (excluidas puto/puta que son intensificadores) ──
    'mierda',
    'asco', 'ascos',
    'basura', 'basuras',
    'coño',
    'cabron', 'cabrona', 'cabrones',
    'hostia',
    # ── Code-switching negativo ───────────────────────────────────
    'hate', 'worst', 'awful', 'terrible', 'horrible', 'bad',
    'ugly', 'boring', 'annoying', 'stupid', 'idiot',
    # ── Otros ─────────────────────────────────────────────────────
    'vergonzoso', 'vergonzosa', 'vergonzosos', 'vergonzosas',
    'inutil', 'inutiles',
    'ineficiente', 'ineficientes',
    'absurdo', 'absurda', 'absurdos', 'absurdas',
    'ridiculo', 'ridicula', 'ridiculos', 'ridiculas',
    'infame', 'infames',
    'repugnante', 'repugnantes',
    'desagradable', 'desagradables',
    'deprimente', 'deprimentes',
    'desesperante', 'desesperantes',
    'agotador', 'agotadora', 'agotadores', 'agotadoras',
    'insufrible', 'insufribles',
})


# =====================================================================
# NEGADORES (incluye formas enclíticas y compuestos)
# =====================================================================
NEGATORS: frozenset = frozenset({
    'no', 'ni', 'sin',
    'nunca', 'jamas', 'jamás',
    'nadie', 'nada', 'ninguno', 'ninguna', 'ningun',
    'tampoco',
    'apenas',
    'no_me', 'no_te', 'no_le', 'no_se', 'no_lo', 'no_la',
    'no_nos', 'no_les',
})


# =====================================================================
# INTENSIFICADORES (≥ 30 formas)
# =====================================================================
INTENSIFIERS: frozenset = frozenset({
    # Cuantitativos / grado
    'muy', 'mucho', 'mucha', 'muchos', 'muchas',
    'bastante', 'bastantes',
    'demasiado', 'demasiada', 'demasiados', 'demasiadas',
    'extremadamente', 'sumamente',
    'enormemente', 'terriblemente', 'completamente', 'totalmente',
    'absolutamente', 'increíblemente', 'increiblemente',
    # Grado superlativo / prefijos
    'super', 'mega', 'ultra', 'extra',
    'hiper', 'sobre', 'archi',
    'requete', 'reconttra',
    # Coloquial España
    'tope',         # "tope bueno"
    'mogollon',     # "mogollón de bien"
    'cantidad',     # "cantidad de cosas buenas"
    'la_hostia',    # como intensificador en ES
    'marcador_intensidad',   # token generado por TextProcessing para puto/puta/hostia/joder
    # Adverbios de grado
    'tan', 'tanto', 'tanta', 'tantos', 'tantas',
    'tal', 'tales',
    'cada',         # "cada vez mejor"
    'ya',           # "ya me llegó" (énfasis)
    'incluso', 'hasta',
})


# =====================================================================
# ATENUADORES (≥ 14 formas)
# =====================================================================
ATTENUATORS: frozenset = frozenset({
    'poco', 'poca', 'pocos', 'pocas',
    'algo', 'medio',
    'casi', 'apenas', 'ligeramente',
    'un_poco', 'un_tanto',
    'tipo',         # "tipo que está bien"
    'como_que',     # "como que no me convenció"
    'mas_o_menos',
    'regular',
})


# =====================================================================
# EMOJIS (tokens generados por TextProcessing, no caracteres Unicode)
# Los tokens 'emoji_pos' y 'emoji_neg' son generados por _replace_emoji_by_polarity()
# =====================================================================
# Tokens semánticos V4 (generados por emoji library en text_processing.py)
# Cada categoría cubre ~5225 emojis via emoji.demojize() + keyword matching
POSITIVE_EMOJIS: frozenset = frozenset({
    'emoji_pos',      # legacy — compatibilidad con versiones anteriores
    'emoji_amor',     # 😍 ❤️ 💕 😘 → amor, cariño, corazones
    'emoji_risa',     # 😂 😆 🤣 😄 → alegría, risa, humor
    'emoji_celebra',  # 👍 🎉 🏆 ✨ 👏 → celebración, aprobación, logros
    'emoji_intenso',  # 🔥 💥 ⚡ 👀 → intensidad, impacto, énfasis positivo
})
NEGATIVE_EMOJIS: frozenset = frozenset({
    'emoji_neg',      # legacy — compatibilidad con versiones anteriores
    'emoji_tristeza', # 😢 💔 😞 😔 → tristeza, decepción, desamor
    'emoji_rabia',    # 😡 😤 🤬 😠 → ira, frustración, enojo
    'emoji_asco',     # 🤮 💩 😒 🤢 → asco, desprecio, desagrado
    'emoji_miedo',    # 😨 😱 💀 👻 → miedo, horror, shock negativo
    'emoji_rechaza',  # 👎 🚫 ❌ 🙅 → rechazo, prohibición, negación
})


# =====================================================================
# BIGRAMAS POSITIVOS (≥ 28 pares frecuentes en Twitter ES)
# Tokens ya normalizados (minúsculas, sin tilde, espacios→guión_bajo)
# =====================================================================
POSITIVE_BIGRAMS: frozenset = frozenset({
    # Expresiones de bienestar (formas superficie)
    'muy_bien', 'muy_bueno', 'muy_buena',
    'que_bien', 'que_guay',
    'muy_chulo', 'muy_chula',
    'buen_rollo',
    # Afecto (formas superficie)
    'te_quiero', 'te_amo', 'me_encanta', 'me_encanto',
    'que_bonito', 'que_bonita',
    'muy_rico', 'super_bien',
    # Logro (formas superficie)
    'bien_hecho', 'muy_bien_hecho',
    'muy_divertido', 'muy_divertida',
    'muy_gracioso', 'muy_graciosa',
    # Satisfacción (formas superficie)
    'me_alegra', 'me_alegro', 'me_alegra_que',
    'esta_genial', 'es_genial',
    'buen_trabajo', 'excelente_trabajo',
    'muy_feliz', 'super_feliz',
    'gran_trabajo', 'gran_idea',
    # ── Formas LEMATIZADAS (spaCy: me→yo, te→tu, es→ser, etc.) ──
    'yo_encantar', 'tu_querer', 'tu_amar',
    'yo_alegrar',
    'ser_genial', 'estar_genial',
    'muy_divertir', 'muy_graciar',
    'ser_bueno', 'ser_buena', 'estar_bien',
    'muy_lindo', 'muy_linda',
    'muy_bonito', 'muy_bonita',
})


# =====================================================================
# BIGRAMAS NEGATIVOS (≥ 30 pares frecuentes en Twitter ES)
# =====================================================================
NEGATIVE_BIGRAMS: frozenset = frozenset({
    # Rechazo y displicencia (formas superficie)
    'no_me_gusta', 'no_me_gusto', 'no_me_parece',
    'me_parece_mal', 'me_parece_fatal',
    'que_pena', 'que_verguenza', 'que_horror', 'que_asco',
    'que_rabia', 'que_palo',
    # Disgusto (formas superficie)
    'da_asco', 'da_pena', 'da_miedo', 'da_igual',
    'es_horrible', 'es_terrible', 'es_pesimo', 'es_fatal',
    # Calificación negativa (formas superficie)
    'muy_malo', 'muy_mala',
    'una_basura', 'un_asco',
    'de_mierda',
    'mal_rollo',
    'no_sirve',
    # Rechazo emocional (formas superficie)
    'lo_odio', 'la_odio', 'los_odio',
    'me_parece_fatal',
    # Fatiga/hastío (formas superficie)
    'estoy_harto', 'estoy_harta',
    'ya_basta', 'hasta_aqui',
    'no_aguanto', 'no_soporto', 'no_tolero',
    # ── Formas LEMATIZADAS (spaCy: es→ser, da→dar, estoy→estar, etc.) ──
    'ser_horrible', 'ser_terrible', 'ser_pesimo', 'ser_fatal',
    'ser_malo', 'ser_mala',
    'dar_asco', 'dar_pena', 'dar_miedo', 'dar_igual',
    'estar_harto', 'estar_harta',
    'yo_odiar',
    'muy_malo', 'muy_mala',
})


# =====================================================================
# ALIASES (retrocompatibilidad con código existente)
# =====================================================================
LEXICON_POS = POSITIVE_WORDS
LEXICON_NEG = NEGATIVE_WORDS
