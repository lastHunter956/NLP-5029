"""
lexicon_es.py — Léxico de sentimiento en español para tweets (TASS 2018)
------------------------------------------------------------------------
Incluye palabras positivas, negativas, negadores, intensificadores,
atenuadores, emojis con polaridad y bigramas frecuentes del corpus.

Todas las palabras están en minúsculas y SIN acentos para maximizar
el match después de limpieza de texto (normalización Unicode).

Uso:
    from logic.lexicon_es import (
        POSITIVE_WORDS, NEGATIVE_WORDS,
        NEGATORS, INTENSIFIERS, ATTENUATORS,
        POSITIVE_EMOJIS, NEGATIVE_EMOJIS,
        POSITIVE_BIGRAMS, NEGATIVE_BIGRAMS,
    )
"""

# ══════════════════════════════════════════════════════════════
# PALABRAS POSITIVAS (~320 entradas)
# ══════════════════════════════════════════════════════════════
POSITIVE_WORDS = {
    # ── Valoración general
    'bueno', 'buena', 'buenos', 'buenas',
    'excelente', 'excelentes',
    'genial', 'geniales',
    'increible', 'increibles',
    'maravilloso', 'maravillosa', 'maravillosos', 'maravillosas',
    'fantastico', 'fantastica', 'fantasticos', 'fantasticas',
    'perfecto', 'perfecta', 'perfectos', 'perfectas',
    'magnifico', 'magnifica',
    'estupendo', 'estupenda',
    'espectacular', 'espectaculares',
    'impresionante', 'impresionantes',
    'extraordinario', 'extraordinaria',
    'fabuloso', 'fabulosa',
    'brillante', 'brillantes',
    'esplendido', 'esplendida',
    'notable', 'notables',
    'optimo', 'optima',
    'positivo', 'positiva',
    'correcto', 'correcta',
    'acertado', 'acertada',
    'adecuado', 'adecuada',
    'apropiado', 'apropiada',

    # ── Emociones positivas
    'feliz', 'felices',
    'alegre', 'alegres',
    'contento', 'contenta', 'contentos', 'contentas',
    'encantado', 'encantada', 'encantados', 'encantadas',
    'encanto', 'encanta', 'encantan',
    'emocionado', 'emocionada',
    'entusiasmado', 'entusiasmada',
    'satisfecho', 'satisfecha',
    'orgulloso', 'orgullosa',
    'enamorado', 'enamorada',
    'apasionado', 'apasionada',
    'ilusionado', 'ilusionada',
    'esperanzado', 'esperanzada',
    'tranquilo', 'tranquila',
    'sereno', 'serena',
    'animado', 'animada',

    # ── Verbos positivos (formas comunes)
    'amar', 'amor', 'amado', 'amada',
    'querer', 'quiero', 'quiere', 'quieren', 'queria',
    'gustar', 'gusta', 'gustan', 'gusto', 'gustado',
    'adorar', 'adoro', 'adora', 'adoran', 'adorado',
    'disfrutar', 'disfruto', 'disfruta', 'disfrutan', 'disfrutado',
    'celebrar', 'celebro', 'celebra', 'celebran', 'celebrado',
    'ganar', 'gano', 'gana', 'ganamos', 'ganaron',
    'lograr', 'logro', 'logra', 'logrado',
    'conseguir', 'consigo', 'consigue', 'conseguido',
    'triunfar', 'triunfo', 'triunfa', 'triunfado',
    'mejorar', 'mejoro', 'mejora', 'mejorado',
    'superar', 'supero', 'supera', 'superado',
    'recomendar', 'recomiendo', 'recomienda',
    'apoyar', 'apoyo', 'apoya',
    'agradecer', 'agradezco', 'agradece',
    'felicitar', 'felicito', 'felicita',

    # ── Sustantivos positivos
    'alegria', 'felicidad', 'gozo', 'placer', 'dicha',
    'exito', 'triunfo', 'victoria', 'logro',
    'esperanza', 'ilusion', 'entusiasmo',
    'amor', 'carino', 'afecto', 'ternura',
    'paz', 'armonia', 'bienestar',
    'salud', 'vida', 'energia',
    'amistad', 'familia', 'union',
    'gracias', 'agradecimiento',
    'bienvenido', 'bienvenida',
    'oportunidad', 'posibilidad',
    'libertad', 'justicia', 'verdad',

    # ── Adjetivos descriptivos positivos
    'bonito', 'bonita', 'bonitos', 'bonitas',
    'hermoso', 'hermosa', 'hermosos', 'hermosas',
    'precioso', 'preciosa', 'preciosos', 'preciosas',
    'lindo', 'linda', 'lindos', 'lindas',
    'bello', 'bella', 'bellos', 'bellas',
    'guapo', 'guapa',
    'elegante', 'elegantes',
    'rico', 'rica', 'ricos', 'ricas',
    'sabroso', 'sabrosa',
    'delicioso', 'deliciosa',
    'agradable', 'agradables',
    'amable', 'amables',
    'simpatico', 'simpatica',
    'gracioso', 'graciosa',
    'divertido', 'divertida',
    'interesante', 'interesantes',
    'fascinante', 'fascinantes',
    'emocionante', 'emocionantes',
    'apasionante', 'apasionantes',
    'reconfortante', 'reconfortantes',
    'inspirador', 'inspiradora',
    'motivador', 'motivadora',
    'creativo', 'creativa',
    'inteligente', 'inteligentes',
    'sabio', 'sabia',
    'valiente', 'valientes',
    'fuerte', 'fuertes',
    'capaz', 'capaces',
    'eficiente', 'eficientes',
    'innovador', 'innovadora',
    'admirable', 'admirables',
    'util', 'utiles',
    'recomendable', 'recomendables',
    'confiable', 'confiables',
    'seguro', 'segura',
    'transparente', 'transparentes',
    'honesto', 'honesta',
    'leal', 'leales',
    'generoso', 'generosa',
    'solidario', 'solidaria',
    'responsable', 'responsables',

    # ── Expresiones de Twitter ES positivas
    'bravo', 'brava',
    'ole', 'vamos', 'dale',
    'chevere', 'chido', 'chida',
    'bacano', 'bacana',
    'espectacular',
    'crack', 'cracks',
    'campeon', 'campeona',
    'figura',
    'alucinante',
    'increiblemente',
    'brutal', 'brutales',
    'flipa', 'flipante',
}

# ══════════════════════════════════════════════════════════════
# PALABRAS NEGATIVAS (~350 entradas)
# ══════════════════════════════════════════════════════════════
NEGATIVE_WORDS = {
    # ── Valoración negativa
    'malo', 'mala', 'malos', 'malas',
    'terrible', 'terribles',
    'horrible', 'horribles',
    'pesimo', 'pesima', 'pesimos', 'pesimas',
    'fatal', 'fatales',
    'nefasto', 'nefasta',
    'desastroso', 'desastrosa',
    'deplorable', 'deplorables',
    'lamentable', 'lamentables',
    'vergonzoso', 'vergonzosa',
    'inaceptable', 'inaceptables',
    'inadmisible', 'inadmisibles',
    'imperdonable', 'imperdonables',
    'insoportable', 'insoportables',
    'intolerable', 'intolerables',
    'injusto', 'injusta',
    'incorrecto', 'incorrecta',
    'equivocado', 'equivocada',
    'erroneo', 'erronea',
    'deficiente', 'deficientes',
    'insuficiente', 'insuficientes',
    'negativo', 'negativa',

    # ── Emociones negativas
    'triste', 'tristes',
    'tristeza',
    'triston', 'tristona',
    'deprimido', 'deprimida',
    'angustiado', 'angustiada',
    'ansioso', 'ansiosa',
    'asustado', 'asustada',
    'aterrorizado', 'aterrorizada',
    'preocupado', 'preocupada',
    'nervioso', 'nerviosa',
    'estresado', 'estresada',
    'agotado', 'agotada',
    'cansado', 'cansada',
    'harto', 'harta',
    'aburrido', 'aburrida',
    'frustrado', 'frustrada',
    'decepcionado', 'decepcionada',
    'defraudado', 'defraudada',
    'traicionado', 'traicionada',
    'abandonado', 'abandonada',
    'rechazado', 'rechazada',
    'humillado', 'humillada',
    'avergonzado', 'avergonzada',

    # ── Verbos negativos (formas comunes)
    'odiar', 'odio', 'odia', 'odian', 'odiado',
    'detestar', 'detesto', 'detesta',
    'llorar', 'lloro', 'llora', 'lloran', 'llorando',
    'sufrir', 'sufro', 'sufre', 'sufren', 'sufrido',
    'doler', 'duele', 'duelen', 'dolio',
    'perder', 'pierdo', 'pierde', 'perdemos', 'perdieron',
    'fracasar', 'fracaso', 'fracasa', 'fracasado',
    'fallar', 'fallo', 'falla', 'fallado',
    'romper', 'rompo', 'rompe', 'roto',
    'destruir', 'destruyo', 'destruye', 'destruido',
    'arruinar', 'arruino', 'arruina', 'arruinado',
    'matar', 'mata', 'matan',
    'herir', 'hiere', 'hieren', 'herido',
    'atacar', 'ataca', 'atacan',
    'agredir', 'agrede', 'agredido',
    'insultar', 'insulta', 'insultado',
    'mentir', 'miente', 'mienten',
    'engañar', 'engana', 'enganado',

    # ── Sustantivos negativos
    'tristeza', 'pena', 'dolor', 'sufrimiento', 'angustia',
    'miedo', 'terror', 'panico', 'horror',
    'odio', 'rabia', 'ira', 'enojo', 'furia',
    'frustracion', 'decepcion', 'desilusión', 'desilusion',
    'fracaso', 'derrota', 'perdida', 'ruina',
    'problema', 'problemas', 'conflicto', 'conflictos',
    'error', 'errores', 'fallo', 'fallos', 'falla', 'fallas',
    'culpa', 'culpas',
    'verguenza', 'humillacion',
    'mentira', 'mentiras', 'engano', 'trampa', 'fraude',
    'traicion', 'injusticia',
    'violencia', 'agresion', 'abuso',
    'corrupcion', 'corrupto', 'corrupta',
    'desastre', 'catastrofe', 'caos',
    'crisis', 'emergencia', 'peligro',
    'amenaza', 'riesgo',
    'muerte', 'muerto', 'muerta',
    'enfermedad', 'enfermo', 'enferma',
    'pobreza', 'miseria',
    'desigualdad', 'discriminacion',
    'incapacidad', 'incompetencia',

    # ── Adjetivos negativos descriptivos
    'feo', 'fea', 'feos', 'feas',
    'grotesco', 'grotesca',
    'asqueroso', 'asquerosa',
    'sucio', 'sucia',
    'estupido', 'estupida', 'estupidos', 'estupidas',
    'idiota', 'idiotas',
    'imbecil', 'imbeciles',
    'inutil', 'inutiles',
    'incompetente', 'incompetentes',
    'irresponsable', 'irresponsables',
    'cobarde', 'cobardes',
    'egoista', 'egoistas',
    'arrogante', 'arrogantes',
    'cruel', 'crueles',
    'violento', 'violenta',
    'agresivo', 'agresiva',
    'peligroso', 'peligrosa',
    'grave', 'graves',
    'preocupante', 'preocupantes',
    'alarmante', 'alarmantes',
    'deprimente', 'deprimentes',
    'angustioso', 'angustiosa',
    'desesperante', 'desesperantes',
    'indigno', 'indigna',
    'lastimoso', 'lastimosa',
    'penoso', 'penosa',
    'ridiculo', 'ridicula',
    'absurdo', 'absurda',
    'ilogico', 'ilogica',

    # ── Expresiones de Twitter ES negativas
    'asco', 'asqueroso',
    'vergüenza', 'verguenza',
    'pff', 'pfff',
    'ugh', 'argh',
    'maldito', 'maldita',
    'malditos', 'malditas',
    'ojalá', 'ojala',
    'desgraciado', 'desgraciada',
    'infeliz', 'infelices',
    'patético', 'patetico',
    'nefasto',
    'penoso',
    'lamentable',
}

# ══════════════════════════════════════════════════════════════
# NEGADORES — invierten la polaridad del contexto
# ══════════════════════════════════════════════════════════════
NEGATORS = {
    'no', 'ni', 'nunca', 'jamas', 'tampoco',
    'nadie', 'nada', 'ninguno', 'ninguna', 'ningun',
    'sin', 'apenas', 'escasamente',
    'imposible', 'incapaz',
}

# ══════════════════════════════════════════════════════════════
# INTENSIFICADORES — amplifican la polaridad
# ══════════════════════════════════════════════════════════════
INTENSIFIERS = {
    'muy', 'mucho', 'mucha', 'muchos', 'muchas',
    'demasiado', 'demasiada',
    'bastante', 'bastantes',
    'super', 'ultra', 'hiper', 'mega',
    'extremadamente', 'totalmente', 'completamente',
    'absolutamente', 'realmente', 'verdaderamente',
    'tremendamente', 'increiblemente', 'sumamente',
    'tan', 'tanto', 'tanta',
    'mas', 'maximo', 'maxima',
}

# ══════════════════════════════════════════════════════════════
# ATENUADORES — reducen la polaridad
# ══════════════════════════════════════════════════════════════
ATTENUATORS = {
    'poco', 'poca', 'pocos', 'pocas',
    'algo', 'un_poco', 'algo_de',
    'casi', 'apenas', 'ligeramente',
    'relativamente', 'medianamente',
    'no_muy', 'no_tan',
}

# ══════════════════════════════════════════════════════════════
# EMOJIS POSITIVOS (texto semántico post-cleaner)
# ══════════════════════════════════════════════════════════════
POSITIVE_EMOJIS = {
    '_emojipositivo_', '_emoji_pos_',
    # Equivalencias de texto para emojis muy frecuentes en TASS
    'emoji',   # placeholder genérico positivo cuando acompaña contexto positivo
}

# ══════════════════════════════════════════════════════════════
# EMOJIS NEGATIVOS
# ══════════════════════════════════════════════════════════════
NEGATIVE_EMOJIS = {
    '_emojinegativo_', '_emoji_neg_',
}

# ══════════════════════════════════════════════════════════════
# BIGRAMAS POSITIVOS frecuentes en TASS 2018
# ══════════════════════════════════════════════════════════════
POSITIVE_BIGRAMS = {
    'muy_bueno', 'muy_buena',
    'muy_bien',
    'muy_bonito', 'muy_bonita',
    'muy_feliz',
    'me_encanta', 'me_gusta',
    'lo_mejor',
    'que_bueno', 'que_rico',
    'buen_trabajo', 'buena_suerte',
    'muchas_gracias',
    'buen_dia', 'buen_dia',
    'feliz_cumpleanos',
}

# ══════════════════════════════════════════════════════════════
# BIGRAMAS NEGATIVOS frecuentes en TASS 2018
# ══════════════════════════════════════════════════════════════
NEGATIVE_BIGRAMS = {
    'muy_malo', 'muy_mala',
    'muy_mal',
    'muy_feo', 'muy_fea',
    'me_duele', 'me_molesta',
    'lo_peor',
    'que_malo', 'que_asco',
    'mala_suerte', 'mala_idea',
    'no_sirve', 'no_funciona',
    'no_me_gusta',
}

# ══════════════════════════════════════════════════════════════
# Alias convenientes para importación directa
# ══════════════════════════════════════════════════════════════
LEXICON_POS = POSITIVE_WORDS
LEXICON_NEG = NEGATIVE_WORDS

if __name__ == '__main__':
    print(f"POSITIVE_WORDS : {len(POSITIVE_WORDS):>4} entradas")
    print(f"NEGATIVE_WORDS : {len(NEGATIVE_WORDS):>4} entradas")
    print(f"NEGATORS       : {len(NEGATORS):>4} entradas")
    print(f"INTENSIFIERS   : {len(INTENSIFIERS):>4} entradas")
    print(f"ATTENUATORS    : {len(ATTENUATORS):>4} entradas")
    print(f"POS_BIGRAMS    : {len(POSITIVE_BIGRAMS):>4} entradas")
    print(f"NEG_BIGRAMS    : {len(NEGATIVE_BIGRAMS):>4} entradas")
    print(f"TOTAL          : {len(POSITIVE_WORDS)+len(NEGATIVE_WORDS):>4} palabras")
