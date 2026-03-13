import emoji

# Explorar API
print("=== emoji_list ===")
tests = [
    "Me encantó 😍😍 qué bueno",
    "Qué asco 😡 horrible todo",
    "jajaja 😂😂 me muero",
    "Te quiero 💕❤️",
    "Todo va mal 😢💔",
    "Felicidades 🎉🎊👏",
    "👍 me parece bien",
    "Texto sin emojis",
]
for t in tests:
    found = emoji.emoji_list(t)
    names = [emoji.demojize(e['emoji']) for e in found]
    print(f"  {t!r}")
    print(f"  → {names}")
    print()

# Probar replace_emoji con función personalizada
print("=== replace_emoji ===")
def replace_fn(e, data):
    name = emoji.demojize(e).strip(':').replace('_', ' ')
    return f" {name} "

result = emoji.replace_emoji("Me encantó 😍 pero odio 😡 esto", replace_fn)
print(f"  replace_emoji: {result!r}")

# Clasificación semántica por keywords
POSITIVE_KW = {'heart','smile','laugh','joy','love','happy','party','celebrate',
               'star','fire','clap','ok','thumbs_up','flower','sun','rainbow',
               'face_with_tears_of_joy','grinning','beaming','wink','kiss',
               'sparkling','blush','slightly_smiling','relieved','hugging',
               'smiling_face_with_heart','100','trophy','medal','crown'}
NEGATIVE_KW = {'angry','enraged','rage','sad','cry','broken','skull','poop',
               'disappointed','worried','tired','weary','scream','fear','nauseated',
               'sneezing','sick','sweat','cold','hot_face','lying','zany',
               'confounded','anguished','grimacing','frowning','cursing','bomb'}

def classify_emoji(e):
    name = emoji.demojize(e).strip(':').lower()
    parts = set(name.replace('-','_').split('_'))
    if parts & POSITIVE_KW:
        return 'pos'
    if parts & NEGATIVE_KW:
        return 'neg'
    return 'neu'

print("\n=== Clasificación semántica ===")
test_emojis = ['😍','😡','😂','💔','👍','🎉','😢','🔥','👏','💩','❤️','😤','🥰','😭','✨','💀']
for e in test_emojis:
    name = emoji.demojize(e)
    cls = classify_emoji(e)
    print(f"  {e} {name:40s} → {cls}")
