# ================================================================
# PC-Transformer Interpretability
# Unified Experiment Script (Experiments 1–5)
# ================================================================
# Usage:
#   1. Set RUN_MODELS in Section 1
#   2. Run all cells
#   3. Results saved to JSON; figures generated automatically
#
# Requirements: transformer_lens, torch, numpy, scipy, matplotlib, pandas
# Tested on: Google Colab T4 GPU
# ================================================================

# ── Section 0: Install & Import ──────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("transformer_lens")

import os, json, gc, warnings
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Section 1: Model Selection ───────────────────────────────────
RUN_MODELS = [
    "gpt2",                       # 117M, 12 layers
    "gpt2-medium",                # 345M, 24 layers
    "gpt2-large",                 # 774M, 36 layers
    "EleutherAI/gpt-neo-125M",   # 125M, 12 layers
    "EleutherAI/pythia-160m",    # 160M, 12 layers
    "EleutherAI/pythia-410m",    # 410M, 24 layers
]

OUTPUT_JSON = "multi_model_results_v2.json"
LANGS = ["en", "es", "fr", "ko", "ru", "ja", "ar"]

# ── Section 2: Prompts & Pairs ───────────────────────────────────

# Experiment 1 & 3: 140 multilingual prompts
PROMPTS_A = [
    # English (30)
    ("en", "The sun rises in the east and sets in the west every day."),
    ("en", "If all birds have wings and penguins are birds then penguins have wings."),
    ("en", "She felt an overwhelming sense of joy when she saw her family again."),
    ("en", "The human brain contains approximately eighty six billion neurons."),
    ("en", "The French Revolution began in seventeen eighty nine and transformed Europe."),
    ("en", "The dog barked loudly in the garden every single morning."),
    ("en", "When temperature drops below zero water turns into ice."),
    ("en", "Courage is not the absence of fear but the decision to act despite it."),
    ("en", "Scientists have discovered a new species of fish in the deep ocean."),
    ("en", "The capital of France is Paris and it is a beautiful city."),
    ("en", "Democracy is a system of government where citizens elect their representatives."),
    ("en", "The immune system protects the body from pathogens and foreign substances."),
    ("en", "Neural networks learn by adjusting weights through a process called backpropagation."),
    ("en", "Climate change is caused primarily by the accumulation of greenhouse gases."),
    ("en", "The speed of light in a vacuum is approximately three hundred thousand kilometers per second."),
    ("en", "Quantum mechanics describes the behavior of particles at the atomic scale."),
    ("en", "The printing press revolutionized the spread of knowledge across Europe."),
    ("en", "Ancient Rome built an empire that stretched across three continents."),
    ("en", "The relationship between supply and demand determines prices in a market economy."),
    ("en", "Photosynthesis converts sunlight into chemical energy stored in glucose molecules."),
    ("en", "The loss of his closest friend left him feeling completely empty inside."),
    ("en", "Fear and excitement mixed together as she stepped onto the stage."),
    ("en", "He struggled with loneliness during the long winter months alone."),
    ("en", "Hope is the quiet voice that whispers try again after every failure."),
    ("en", "The painting evoked a deep feeling of nostalgia for a simpler time."),
    ("en", "All mammals breathe air and whales are mammals so whales breathe air."),
    ("en", "If the server crashes all unsaved data will be permanently lost forever."),
    ("en", "Because she studied hard every day she passed the exam with ease."),
    ("en", "The old bridge was rebuilt after the flood damaged it severely last year."),
    ("en", "They decided to plant a garden in the backyard this coming spring."),
    # Spanish (20)
    ("es", "El sol sale por el este y se pone por el oeste cada dia."),
    ("es", "Si todos los humanos son mortales entonces nadie vive para siempre."),
    ("es", "El amor y la amistad son fundamentales para la felicidad humana."),
    ("es", "La ciencia nos permite entender el mundo que nos rodea cada dia."),
    ("es", "El cerebro humano contiene aproximadamente ochenta y seis mil millones de neuronas."),
    ("es", "La democracia es un sistema de gobierno donde los ciudadanos eligen a sus representantes."),
    ("es", "El cambio climatico es causado principalmente por la acumulacion de gases de efecto invernadero."),
    ("es", "La revolucion francesa comenzo en mil setecientos ochenta y nueve."),
    ("es", "Cuando la temperatura baja de cero el agua se convierte en hielo."),
    ("es", "Ella sintio una abrumadora sensacion de alegria cuando vio a su familia."),
    ("es", "La luna orbita alrededor de la tierra cada veintiocho dias aproximadamente."),
    ("es", "Los cientificos han descubierto una nueva especie de pez en el oceano profundo."),
    ("es", "La inteligencia artificial esta transformando la forma en que vivimos y trabajamos."),
    ("es", "Si la demanda supera a la oferta los precios tienden a subir en el mercado."),
    ("es", "El miedo y la emocion se mezclaron cuando ella subio al escenario."),
    ("es", "La fotosintesis convierte la luz solar en energia quimica almacenada en glucosa."),
    ("es", "Todos los mamiferos respiran aire y las ballenas son mamiferos luego respiran aire."),
    ("es", "El sistema inmunologico protege al cuerpo de patogenos y sustancias extranas."),
    ("es", "La imprenta revoluciono la difusion del conocimiento en toda Europa."),
    ("es", "El perro ladraba fuerte en el jardin cada manana sin falta."),
    # French (20)
    ("fr", "Le soleil se leve a l est et se couche a l ouest chaque jour."),
    ("fr", "Si tous les humains sont mortels alors personne ne vit eternellement."),
    ("fr", "L amour et l amitie sont essentiels au bonheur humain absolument."),
    ("fr", "La science nous permet de comprendre le monde qui nous entoure."),
    ("fr", "Le cerveau humain contient environ quatre vingt six milliards de neurones."),
    ("fr", "La democratie est un systeme de gouvernement ou les citoyens elisent leurs representants."),
    ("fr", "Le changement climatique est cause par l accumulation de gaz a effet de serre."),
    ("fr", "La revolution francaise a commence en mille sept cent quatre vingt neuf."),
    ("fr", "Quand la temperature descend en dessous de zero l eau se transforme en glace."),
    ("fr", "Elle a ressenti une joie immense quand elle a revu sa famille."),
    ("fr", "La lune orbite autour de la terre tous les vingt huit jours environ."),
    ("fr", "Les scientifiques ont decouvert une nouvelle espece de poisson dans l ocean profond."),
    ("fr", "L intelligence artificielle transforme notre facon de vivre et de travailler."),
    ("fr", "Si la demande depasse l offre les prix ont tendance a augmenter sur le marche."),
    ("fr", "La peur et l excitation se melaient quand elle montait sur scene."),
    ("fr", "La photosynthese convertit la lumiere solaire en energie chimique dans la glucose."),
    ("fr", "Tous les mammiferes respirent de l air et les baleines sont des mammiferes."),
    ("fr", "Le systeme immunitaire protege le corps contre les agents pathogenes etrangers."),
    ("fr", "L imprimerie a revolutionne la diffusion du savoir a travers toute l Europe."),
    ("fr", "Le chien aboyait fort dans le jardin chaque matin sans exception."),
    # Korean (20)
    ("ko", "서울은 대한민국의 수도이며 가장 큰 도시이다."),
    ("ko", "아이들은 해가 질 때까지 공원에서 놀았다."),
    ("ko", "과학은 우리가 세상을 이해할 수 있게 해준다."),
    ("ko", "모든 인간이 죽는다면 아무도 영원히 살지 못한다."),
    ("ko", "사랑과 우정은 인간의 행복에 필수적이다."),
    ("ko", "인간의 뇌는 약 팔백육십억 개의 뉴런을 포함하고 있다."),
    ("ko", "민주주의는 시민이 대표를 선출하는 정치 체제이다."),
    ("ko", "기후 변화는 주로 온실 가스 축적으로 인해 발생한다."),
    ("ko", "프랑스 혁명은 천칠백팔십구년에 시작되었다."),
    ("ko", "온도가 영하로 내려가면 물은 얼음으로 변한다."),
    ("ko", "그녀는 가족을 다시 만났을 때 엄청난 기쁨을 느꼈다."),
    ("ko", "과학자들은 심해에서 새로운 어종을 발견했다."),
    ("ko", "달은 약 이십팔일마다 지구 주위를 돈다."),
    ("ko", "인공지능은 우리가 살고 일하는 방식을 변화시키고 있다."),
    ("ko", "수요가 공급을 초과하면 시장에서 가격이 오르는 경향이 있다."),
    ("ko", "광합성은 태양 에너지를 포도당에 저장된 화학 에너지로 변환한다."),
    ("ko", "모든 포유류는 공기를 호흡하고 고래는 포유류이므로 공기를 호흡한다."),
    ("ko", "면역 체계는 병원체와 외래 물질로부터 신체를 보호한다."),
    ("ko", "인쇄기는 유럽 전역에 걸쳐 지식의 보급을 혁명적으로 변화시켰다."),
    ("ko", "개는 매일 아침 정원에서 크게 짖었다."),
    # Russian (20)
    ("ru", "Солнце встаёт на востоке и садится на западе каждый день."),
    ("ru", "Дети играли в парке до захода солнца каждый день."),
    ("ru", "Наука позволяет нам понять окружающий мир вокруг нас."),
    ("ru", "Если все люди смертны то никто не живёт вечно на земле."),
    ("ru", "Любовь и дружба необходимы для счастья человека всегда."),
    ("ru", "Мозг человека содержит около восьмидесяти шести миллиардов нейронов."),
    ("ru", "Демократия это система правления при которой граждане избирают своих представителей."),
    ("ru", "Изменение климата вызвано главным образом накоплением парниковых газов в атмосфере."),
    ("ru", "Французская революция началась в тысяча семьсот восемьдесят девятом году."),
    ("ru", "Когда температура опускается ниже нуля вода превращается в лёд."),
    ("ru", "Она почувствовала огромную радость когда снова увидела свою семью."),
    ("ru", "Учёные обнаружили новый вид рыб в глубинах океана совсем недавно."),
    ("ru", "Луна вращается вокруг Земли примерно каждые двадцать восемь дней."),
    ("ru", "Искусственный интеллект меняет то как мы живём и работаем."),
    ("ru", "Если спрос превышает предложение цены на рынке имеют тенденцию расти."),
    ("ru", "Фотосинтез превращает солнечный свет в химическую энергию хранящуюся в глюкозе."),
    ("ru", "Все млекопитающие дышат воздухом а киты являются млекопитающими значит дышат воздухом."),
    ("ru", "Иммунная система защищает организм от патогенов и чужеродных веществ ежедневно."),
    ("ru", "Книгопечатание произвело революцию в распространении знаний по всей Европе."),
    ("ru", "Собака громко лаяла в саду каждое утро без исключения."),
    # Japanese (20)
    ("ja", "太陽は東から昇り西に沈む。"),
    ("ja", "子供たちは日が沈むまで公園で遊んだ。"),
    ("ja", "科学は私たちが世界を理解するのを助けてくれる。"),
    ("ja", "すべての人間が死ぬなら誰も永遠には生きられない。"),
    ("ja", "愛と友情は人間の幸福に不可欠だ。"),
    ("ja", "人間の脳には約八百六十億個のニューロンが含まれている。"),
    ("ja", "民主主義は市民が代表者を選ぶ政治体制だ。"),
    ("ja", "気候変動は主に温室効果ガスの蓄積によって引き起こされる。"),
    ("ja", "フランス革命は千七百八十九年に始まった。"),
    ("ja", "気温が零度以下に下がると水は氷になる。"),
    ("ja", "彼女は家族に再会したとき圧倒的な喜びを感じた。"),
    ("ja", "科学者たちは深海で新しい魚の種を発見した。"),
    ("ja", "月は約二十八日ごとに地球の周りを回っている。"),
    ("ja", "人工知能は私たちの生活と仕事の方法を変えている。"),
    ("ja", "需要が供給を超えると市場では価格が上昇する傾向がある。"),
    ("ja", "光合成は太陽光をグルコースに蓄えられた化学エネルギーに変換する。"),
    ("ja", "すべての哺乳類は空気を呼吸しクジラは哺乳類なので空気を呼吸する。"),
    ("ja", "免疫システムは病原体や異物から体を保護する。"),
    ("ja", "印刷機はヨーロッパ全土での知識の普及に革命をもたらした。"),
    ("ja", "犬は毎朝庭で大きく吠えた。"),
    # Arabic (10)
    ("ar", "تشرق الشمس من الشرق وتغرب من الغرب كل يوم."),
    ("ar", "لعب الأطفال في الحديقة حتى غروب الشمس."),
    ("ar", "تتيح لنا العلوم فهم العالم من حولنا."),
    ("ar", "إذا كان كل البشر فانين فلا أحد يعيش إلى الأبد."),
    ("ar", "الحب والصداقة ضروريان لسعادة الإنسان."),
    ("ar", "يحتوي الدماغ البشري على حوالي ستة وثمانين مليار خلية عصبية."),
    ("ar", "الديمقراطية نظام حكم يختار فيه المواطنون ممثليهم."),
    ("ar", "يُعزى تغير المناخ أساسًا إلى تراكم غازات الاحتباس الحراري."),
    ("ar", "بدأت الثورة الفرنسية عام ألف وسبعمائة وتسعة وثمانين."),
    ("ar", "عندما تنخفض درجة الحرارة إلى ما دون الصفر يتحول الماء إلى جليد."),
]

# English prompts only (for Experiments 3 & 5)
PROMPTS_EN = [p for lang, p in PROMPTS_A if lang == "en"]

# Experiment 2: Semantic pairs for activation patching
PAIRS_B = [
    {"clean": "The capital of France is Paris",
     "corrupt": "The capital of Germany is Berlin",
     "clean_token": " Paris", "corrupt_token": " Berlin",
     "name": "cap_france_germany"},
    {"clean": "The capital of Japan is Tokyo",
     "corrupt": "The capital of China is Beijing",
     "clean_token": " Tokyo", "corrupt_token": " Beijing",
     "name": "cap_japan_china"},
    {"clean": "The capital of Italy is Rome",
     "corrupt": "The capital of Spain is Madrid",
     "clean_token": " Rome", "corrupt_token": " Madrid",
     "name": "cap_italy_spain"},
    {"clean": "The capital of Greece is Athens",
     "corrupt": "The capital of Turkey is Ankara",
     "clean_token": " Athens", "corrupt_token": " Ankara",
     "name": "cap_greece_turkey"},
    {"clean": "The language of France is French",
     "corrupt": "The language of Germany is German",
     "clean_token": " French", "corrupt_token": " German",
     "name": "lang_france_germany"},
]

# Experiment 4: Extended semantic pairs for zero-ablation
PAIRS_EXTENDED = [
    # Capitals (10)
    ("The capital of France is", " Paris",
     "The capital of Spain is",  " Madrid",   "capital"),
    ("The capital of Japan is",  " Tokyo",
     "The capital of China is",  " Beijing",  "capital"),
    ("The capital of Italy is",  " Rome",
     "The capital of Greece is", " Athens",   "capital"),
    ("The capital of Germany is"," Berlin",
     "The capital of France is", " Paris",    "capital"),
    ("The capital of Russia is", " Moscow",
     "The capital of Germany is"," Berlin",   "capital"),
    ("The capital of Egypt is",  " Cairo",
     "The capital of Russia is", " Moscow",   "capital"),
    ("The capital of Greece is", " Athens",
     "The capital of Italy is",  " Rome",     "capital"),
    ("The capital of Spain is",  " Madrid",
     "The capital of Japan is",  " Tokyo",    "capital"),
    ("The capital of China is",  " Beijing",
     "The capital of Egypt is",  " Cairo",    "capital"),
    ("The capital of India is",  " Delhi",
     "The capital of China is",  " Beijing",  "capital"),
    # Currencies (8)
    ("The currency of Japan is the",  " yen",
     "The currency of China is the",  " yuan",   "currency"),
    ("The currency of UK is the",     " pound",
     "The currency of Japan is the",  " yen",    "currency"),
    ("The currency of China is the",  " yuan",
     "The currency of UK is the",     " pound",  "currency"),
    ("The currency of India is the",  " rup",
     "The currency of UK is the",     " pound",  "currency"),
    ("The currency of Russia is the", " ruble",
     "The currency of India is the",  " rup",    "currency"),
    ("The currency of Mexico is the", " peso",
     "The currency of Russia is the", " ruble",  "currency"),
    ("The currency of USA is the",    " dollar",
     "The currency of Mexico is the", " peso",   "currency"),
    ("The currency of Korea is the",  " won",
     "The currency of USA is the",    " dollar", "currency"),
    # Language relations (8)
    ("People in France speak",   " French",
     "People in Germany speak",  " German",    "language"),
    ("People in Germany speak",  " German",
     "People in France speak",   " French",    "language"),
    ("People in Japan speak",    " Japanese",
     "People in China speak",    " Chinese",   "language"),
    ("People in China speak",    " Chinese",
     "People in Japan speak",    " Japanese",  "language"),
    ("People in Italy speak",    " Italian",
     "People in France speak",   " French",    "language"),
    ("People in Russia speak",   " Russian",
     "People in Italy speak",    " Italian",   "language"),
    ("People in Spain speak",    " Spanish",
     "People in Russia speak",   " Russian",   "language"),
    ("People in Greece speak",   " Greek",
     "People in Spain speak",    " Spanish",   "language"),
    # Colors (6)
    ("The color of grass is",    " green",
     "The color of the sky is",  " blue",      "color"),
    ("The color of the sky is",  " blue",
     "The color of grass is",    " green",     "color"),
    ("The color of snow is",     " white",
     "The color of coal is",     " black",     "color"),
    ("The color of coal is",     " black",
     "The color of snow is",     " white",     "color"),
    ("The color of blood is",    " red",
     "The color of snow is",     " white",     "color"),
    ("The color of the sun is",  " yellow",
     "The color of blood is",    " red",       "color"),
    # Miscellaneous (5)
    ("The sound a dog makes is", " bark",
     "The sound a cat makes is", " me",        "animal"),
    ("The largest ocean is the", " Pac",
     "The smallest ocean is the"," Arc",       "geography"),
    ("Water freezes at",         " zero",
     "Water boils at",           " one",       "science"),
    ("The opposite of hot is",   " cold",
     "The opposite of big is",   " small",     "antonym"),
    ("The opposite of day is",   " night",
     "The opposite of hot is",   " cold",      "antonym"),
]

# ── Section 3: Utility Functions ─────────────────────────────────

def is_single_token(model, word):
    try:
        model.to_single_token(word)
        return True
    except:
        return False

def get_zone_layers(n_layers, zone):
    ranges = {
        "early_spike": (0.10, 0.20),
        "convergence": (0.25, 0.75),
        "late_spike":  (0.80, 0.95),
    }
    lo, hi = ranges[zone]
    return [l for l in range(n_layers) if lo <= l / n_layers <= hi]

# ── Section 4: Experiment Functions ──────────────────────────────

def run_exp1(model, model_name, n_layers):
    """Residual stream convergence across 7 languages."""
    print(f"  [Exp 1] {model_name}")
    results = []
    for idx, (lang, prompt) in enumerate(PROMPTS_A):
        try:
            tokens = model.to_tokens(prompt)[:, 1:].to(device)
            _, cache = model.run_with_cache(tokens)

            # Cosine similarities between consecutive layers
            sims = []
            for l in range(1, n_layers):
                prev = cache[f"blocks.{l-1}.hook_resid_post"][0].cpu()
                curr = cache[f"blocks.{l}.hook_resid_post"][0].cpu()
                sims.append(
                    torch.nn.functional.cosine_similarity(prev, curr, dim=-1).numpy()
                )
            sims = np.array(sims)

            # Delta magnitudes
            deltas = []
            for l in range(n_layers):
                pre  = cache[f"blocks.{l}.hook_resid_pre"][0].cpu()
                post = cache[f"blocks.{l}.hook_resid_post"][0].cpu()
                deltas.append(torch.norm(post - pre, p=2, dim=-1).numpy())
            deltas = np.array(deltas)

            early_d = deltas[:max(1, int(n_layers*0.4))].flatten()
            late_d  = deltas[int(n_layers*0.6):].flatten()
            _, delta_p = stats.ttest_ind(early_d, late_d)

            mid_s   = sims[int(n_layers*0.25):int(n_layers*0.75)].flatten()
            early_s = sims[:max(1, int(n_layers*0.25))].flatten()

            results.append({
                "lang":            lang,
                "middle_sim_mean": float(mid_s.mean()),
                "delta_p":         float(delta_p),
            })

        except Exception as e:
            pass

        if (idx + 1) % 20 == 0:
            print(f"    {idx+1}/{len(PROMPTS_A)} done")

    df = pd.DataFrame(results)
    summary = df.groupby("lang")[["middle_sim_mean", "delta_p"]].mean()
    return summary.to_dict(), results


def run_exp2(model, model_name, n_layers):
    """Activation patching across semantic pairs."""
    print(f"  [Exp 2] {model_name}")
    b_results = {}

    for pair in PAIRS_B:
        if not (is_single_token(model, pair["clean_token"]) and
                is_single_token(model, pair["corrupt_token"])):
            continue
        try:
            clean_toks   = model.to_tokens(pair["clean"])[:, 1:].to(device)
            corrupt_toks = model.to_tokens(pair["corrupt"])[:, 1:].to(device)
            seq_len      = min(clean_toks.shape[1], corrupt_toks.shape[1])
            clean_toks   = clean_toks[:, :seq_len]
            corrupt_toks = corrupt_toks[:, :seq_len]

            _, clean_cache = model.run_with_cache(clean_toks)
            ct  = model.to_single_token(pair["clean_token"])
            cot = model.to_single_token(pair["corrupt_token"])
            n_pos   = corrupt_toks.shape[1]
            effects = torch.zeros(n_layers, n_pos)

            for layer in range(n_layers):
                hook_name = f"blocks.{layer}.hook_resid_post"
                for pos in range(n_pos):
                    cv = clean_cache[hook_name][0, pos, :].detach().clone()
                    def make_hook(c, p):
                        def fn(value, hook):
                            value = value.clone()
                            value[0, p, :] = c
                            return value
                        return fn
                    patched = model.run_with_hooks(
                        corrupt_toks,
                        fwd_hooks=[(hook_name, make_hook(cv, pos))]
                    )
                    effects[layer, pos] = (patched[0,-1,ct] - patched[0,-1,cot]).item()

            b_results[pair["name"]] = effects.numpy().tolist()
            print(f"    done: {pair['name']}")

        except Exception as e:
            print(f"    skip {pair['name']}: {e}")

    return b_results


def run_exp3(model, model_name, n_layers):
    """MLP transform magnitude profiling."""
    print(f"  [Exp 3] {model_name}")
    all_norms = []
    for prompt in PROMPTS_EN:
        try:
            tokens = model.to_tokens(prompt)[:, 1:].to(device)
            _, cache = model.run_with_cache(tokens)
            norms = []
            for l in range(n_layers):
                try:
                    mlp_in  = cache[f"blocks.{l}.ln2.hook_normalized"][0].cpu()
                    mlp_out = cache[f"blocks.{l}.hook_mlp_out"][0].cpu()
                    norms.append(torch.norm(mlp_out - mlp_in, dim=-1).mean().item())
                except:
                    norms.append(0.0)
            all_norms.append(norms)
        except:
            pass
    return np.mean(all_norms, axis=0).tolist() if all_norms else [0.0] * n_layers


def run_exp4(model, model_name, n_layers):
    """Zero-ablation of MLP zones."""
    print(f"  [Exp 4] {model_name}")
    zones     = ["early_spike", "convergence", "late_spike"]
    zone_data = {z: [] for z in zones}

    valid_pairs = [p for p in PAIRS_EXTENDED
                   if is_single_token(model, p[1]) and is_single_token(model, p[3])]
    print(f"    valid pairs: {len(valid_pairs)}")

    def zero_hook(value, hook):
        return torch.zeros_like(value)

    skipped = 0
    for item in valid_pairs:
        cp, ct_str, xp, xt_str, cat = item
        try:
            ct   = model.to_single_token(ct_str)
            xt   = model.to_single_token(xt_str)
            toks = model.to_tokens(cp)[:, 1:].to(device)

            with torch.no_grad():
                bl = model(toks)
            baseline = (bl[0,-1,ct] - bl[0,-1,xt]).item()
            if baseline < 1.0:
                skipped += 1
                continue

            for zone in zones:
                layers = get_zone_layers(n_layers, zone)
                hooks  = [(f"blocks.{l}.hook_mlp_out", zero_hook) for l in layers]
                with torch.no_grad():
                    ab = model.run_with_hooks(toks, fwd_hooks=hooks)
                ablated = (ab[0,-1,ct] - ab[0,-1,xt]).item()
                drop    = (baseline - ablated) / abs(baseline)
                zone_data[zone].append({"drop": drop, "category": cat})

        except:
            skipped += 1

    print(f"    skipped (baseline<1.0 or error): {skipped}")
    result = {}
    for zone in zones:
        drops = [d["drop"] for d in zone_data[zone]]
        if drops:
            result[zone] = {
                "mean_drop": float(np.mean(drops)),
                "std_drop":  float(np.std(drops)),
                "n_pairs":   len(drops),
                "raw":       drops,
            }
            print(f"    {zone}: {result[zone]['mean_drop']:+.3f} ± "
                  f"{result[zone]['std_drop']:.3f} (n={len(drops)})")
    return result


def run_exp5(model, model_name, n_layers):
    """Logit lens: track target token rank across layers."""
    print(f"  [Exp 5] {model_name}")
    W_U = model.W_U.detach()
    b_U = model.b_U.detach()
    layer_ranks = [[] for _ in range(n_layers)]

    for prompt in PROMPTS_EN[:15]:
        try:
            tokens     = model.to_tokens(prompt).to(device)
            if tokens.shape[1] < 4:
                continue
            target_tok = tokens[0, -1].item()
            input_toks = tokens[:, :-1]

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    input_toks,
                    names_filter=lambda n: "hook_resid_post" in n
                )

            for l in range(n_layers):
                key = f"blocks.{l}.hook_resid_post"
                if key not in cache:
                    continue
                resid      = cache[key][0, -1, :].float()
                resid_norm = (resid - resid.mean()) / (resid.std() + 1e-8)
                logits_l   = resid_norm @ W_U.float() + b_U.float()
                sorted_ids = logits_l.argsort(descending=True)
                matches    = (sorted_ids == target_tok).nonzero(as_tuple=True)
                if len(matches[0]) > 0:
                    layer_ranks[l].append(matches[0][0].item())

            del cache
            torch.cuda.empty_cache()
        except:
            continue

    avg_ranks = [float(np.mean(r)) if r else None for r in layer_ranks]
    valid = [r for r in avg_ranks if r is not None]
    if valid:
        print(f"    L0≈{valid[0]:.0f} → Lfinal≈{valid[-1]:.2f}")
    return avg_ranks

# ── Section 5: Main Loop ──────────────────────────────────────────

if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON) as f:
        all_results = json.load(f)
    print(f"Loaded existing results: {list(all_results.keys())}")
else:
    all_results = {}

for model_name in RUN_MODELS:
    short = model_name.split("/")[-1]
    if short in all_results:
        print(f"\n[{short}] already done, skipping")
        continue

    print(f"\n{'='*55}\nModel: {model_name}\n{'='*55}")

    try:
        model = HookedTransformer.from_pretrained(model_name, device=device)
        model.eval()
        n_layers = model.cfg.n_layers
        print(f"Layers: {n_layers}")

        summary_A, results_A = run_exp1(model, short, n_layers)
        results_B            = run_exp2(model, short, n_layers)
        results_C            = run_exp3(model, short, n_layers)
        results_D            = run_exp4(model, short, n_layers)
        results_E            = run_exp5(model, short, n_layers)

        all_results[short] = {
            "model_full_name": model_name,
            "n_layers":        n_layers,
            "n_params":        sum(p.numel() for p in model.parameters()),
            "A_summary":       summary_A,
            "A_results":       results_A,
            "B_results":       results_B,
            "C_mlp_norms":     results_C,
            "D_zero_ablation": results_D,
            "E_logit_lens":    results_E,
        }

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[{short}] saved.")

    except Exception as e:
        print(f"[{model_name}] FAILED: {e}")
    finally:
        try: del model
        except: pass
        torch.cuda.empty_cache()
        gc.collect()

print(f"\nAll done: {list(all_results.keys())}")

# ── Section 6: Generate All Figures ──────────────────────────────

models = list(all_results.keys())
MODEL_LABELS = {
    "gpt2":         "GPT-2 (117M)",
    "gpt2-medium":  "GPT-2 Med (345M)",
    "gpt2-large":   "GPT-2 Large (774M)",
    "gpt-neo-125M": "GPT-Neo (125M)",
    "pythia-160m":  "Pythia (160M)",
    "pythia-410m":  "Pythia (410M)",
}
LANG_COLORS = {
    "en": "#1f77b4", "es": "#ff7f0e", "fr": "#2ca02c",
    "ko": "#d62728", "ru": "#9467bd", "ja": "#8c564b", "ar": "#e377c2"
}
MODEL_COLORS = {
    "gpt2":         "#1f77b4",
    "gpt2-medium":  "#aec7e8",
    "gpt2-large":   "#0a4a7a",
    "gpt-neo-125M": "#d62728",
    "pythia-160m":  "#9467bd",
    "pythia-410m":  "#8c564b",
}
ZONE_COLORS = {
    "early_spike": "#e74c3c",
    "convergence": "#3498db",
    "late_spike":  "#e67e22",
}

# Figure 1: Cosine Similarity
print("\n[Figure 1] Cosine Similarity...")
n = len(models)
fig, axes = plt.subplots(1, n, figsize=(4.5*n, 5.5), sharey=False)
if n == 1: axes = [axes]
for ax, m in zip(axes, models):
    summary = all_results[m]["A_summary"]["middle_sim_mean"]
    vals    = [summary.get(lang, 0) for lang in LANGS]
    bars    = ax.bar(range(7), vals,
                     color=[LANG_COLORS[l] for l in LANGS],
                     alpha=0.88, width=0.7, edgecolor="white")
    bars[0].set_edgecolor("black"); bars[0].set_linewidth(2.2)
    v_min, v_max = min(v for v in vals if v > 0), max(vals)
    margin = (v_max - v_min) * 0.3
    ax.set_ylim(max(0, v_min - margin), min(1.0, v_max + margin*0.5))
    ax.set_xticks(range(7)); ax.set_xticklabels(LANGS, fontsize=10)
    en_rank = sorted(vals, reverse=True).index(summary.get("en", 0)) + 1
    ax.set_title(f"{MODEL_LABELS.get(m, m)}\n(en #{en_rank}/7)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Language", fontsize=10)
    ax.set_ylabel("Mid-Layer Cosine Similarity", fontsize=9)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    en_val = summary.get("en", 0)
    ax.annotate(f"{en_val:.3f}", xy=(0, en_val), xytext=(0, 5),
                textcoords="offset points", ha="center",
                fontsize=8, fontweight="bold")
handles = [plt.Rectangle((0,0),1,1, color=LANG_COLORS[l]) for l in LANGS]
fig.legend(handles, LANGS, loc="upper right", bbox_to_anchor=(1.0, 1.02),
           ncol=1, fontsize=10)
fig.suptitle("Figure 1: Mid-Layer Cosine Similarity by Language\n"
             "(Independent y-axes | en #rank = English rank among 7 languages)",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("figure1.png", dpi=150, bbox_inches="tight", facecolor="white")
print("  figure1.png saved")

# Figure 2: MLP Magnitude (by family)
print("[Figure 2] MLP Magnitude...")
families = {
    "GPT-2 Family":   [m for m in models if "gpt2" in m],
    "GPT-Neo Family": [m for m in models if "neo" in m],
    "Pythia Family":  [m for m in models if "pythia" in m],
}
families = {k: v for k, v in families.items() if v}
fig, axes = plt.subplots(1, len(families), figsize=(7*len(families), 5.5))
if len(families) == 1: axes = [axes]
for ax, (title, group) in zip(axes, families.items()):
    for m in group:
        norms = all_results[m]["C_mlp_norms"]
        x = [i/len(norms) for i in range(len(norms))]
        ax.plot(x, norms, marker="o", markersize=3, linewidth=2,
                color=MODEL_COLORS.get(m, "#333"),
                label=MODEL_LABELS.get(m, m), alpha=0.9)
    ax.axvspan(0.10, 0.20, alpha=0.08, color="red")
    ax.axvspan(0.80, 0.95, alpha=0.08, color="blue")
    ax.set_xlabel("Normalized Layer Position", fontsize=11)
    ax.set_ylabel("MLP Transform Magnitude", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.suptitle("Figure 2: MLP Transform Magnitude (Normalized Layer Position)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figure2.png", dpi=150, bbox_inches="tight", facecolor="white")
print("  figure2.png saved")

# Figure 3: Activation Patching
print("[Figure 3] Activation Patching...")
def get_avg_patch(b_res):
    mats = []
    for _, mat in b_res.items():
        m = np.array(mat)
        mn, mx = m.min(), m.max()
        mats.append((m - mn)/(mx - mn) if mx - mn > 1e-6 else np.zeros_like(m))
    if not mats: return None
    mp = max(m.shape[1] for m in mats)
    nl = mats[0].shape[0]
    p  = np.zeros((len(mats), nl, mp))
    for i, m in enumerate(mats): p[i, :, mp-m.shape[1]:] = m
    return np.mean(p, axis=0)

mb = [m for m in models if all_results[m].get("B_results")]
if mb:
    fig, axes = plt.subplots(1, len(mb), figsize=(4.8*len(mb), 5.5))
    if len(mb) == 1: axes = [axes]
    last_im = None
    for ax, m in zip(axes, mb):
        mat = get_avg_patch(all_results[m]["B_results"])
        if mat is None: continue
        nl = mat.shape[0]
        last_im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                            vmin=0, vmax=1, origin="lower",
                            interpolation="nearest")
        ax.axhline(y=nl*0.25-0.5, color="yellow", lw=1.8, ls="--", alpha=0.95)
        ax.axhline(y=nl*0.33+0.5, color="yellow", lw=1.8, ls="--", alpha=0.95)
        ax.set_title(MODEL_LABELS.get(m, m), fontsize=11, fontweight="bold")
        ax.set_xlabel("Token Position", fontsize=10)
        if ax == axes[0]: ax.set_ylabel("Layer", fontsize=10)
        step = max(1, nl//6)
        ax.set_yticks(range(0, nl, step))
        ax.set_yticklabels([f"L{i}" for i in range(0, nl, step)], fontsize=8)
    if last_im:
        plt.colorbar(last_im, ax=axes[-1], fraction=0.046, pad=0.04,
                     label="Normalized Patching Effect")
    fig.suptitle("Figure 3: Activation Patching — Residual Stream\n"
                 "Yellow dashed = normalized layer pos 0.25–0.33",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figure3.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("  figure3.png saved")

# Figure 4: delta_p heatmap (Appendix)
print("[Figure 4] delta_p heatmap...")
delta_mat, valid_mods = [], []
for m in models:
    row = [all_results[m]["A_summary"].get("delta_p", {}).get(l, 1.0)
           for l in LANGS]
    delta_mat.append(row)
    valid_mods.append(MODEL_LABELS.get(m, m))
fig, ax = plt.subplots(figsize=(10, max(4, len(valid_mods)*1.4)))
im = ax.imshow(delta_mat, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="auto")
ax.set_xticks(range(7)); ax.set_xticklabels(LANGS, fontsize=12)
ax.set_yticks(range(len(valid_mods))); ax.set_yticklabels(valid_mods, fontsize=10)
for i, row in enumerate(delta_mat):
    for j, val in enumerate(row):
        txt = f"{val:.3f}" if val >= 0.001 else "<.001"
        col = "white" if val < 0.02 or val > 0.08 else "black"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=8, color=col, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
             label="delta p-value\n(green=significant, red=not significant)")
ax.set_title("Figure 4: delta_p by Language and Model\n"
             "(Supplement — Experiment 1)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figure4_delta_p.png", dpi=150, bbox_inches="tight", facecolor="white")
print("  figure4_delta_p.png saved")

# Figure 5: Zero-Ablation
print("[Figure 5] Zero-Ablation...")
zones = ["early_spike", "convergence", "late_spike"]
ZONE_LABELS = {
    "early_spike": "Early\nSpike\n(0.10–0.20)",
    "convergence": "Convergence\n(0.25–0.75)",
    "late_spike":  "Late\nSpike\n(0.80–0.95)",
}
md = [m for m in models if all_results[m].get("D_zero_ablation")]
if md:
    fig, axes = plt.subplots(1, len(md), figsize=(4*len(md), 5.5), sharey=True)
    if len(md) == 1: axes = [axes]
    for ax, m in zip(axes, md):
        res   = all_results[m]["D_zero_ablation"]
        means = [res.get(z, {}).get("mean_drop", 0) for z in zones]
        stds  = [res.get(z, {}).get("std_drop",  0) for z in zones]
        ns    = [res.get(z, {}).get("n_pairs",   0) for z in zones]
        bars  = ax.bar(range(3), means,
                       color=[ZONE_COLORS[z] for z in zones],
                       alpha=0.85, width=0.6, yerr=stds, capsize=5,
                       edgecolor="white")
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels([ZONE_LABELS[z] for z in zones], fontsize=9)
        ax.set_title(MODEL_LABELS.get(m, m), fontsize=11, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Performance Drop\n(+= harmful)", fontsize=10)
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for bar, mean, n_val in zip(bars, means, ns):
            ax.text(bar.get_x() + bar.get_width()/2,
                    mean + (0.03 if mean >= 0 else -0.06),
                    f"{mean:+.2f}\n(n={n_val})",
                    ha="center", fontsize=8.5, fontweight="bold")
    fig.suptitle("Figure 5: Zero-Ablation of MLP Zones\n"
                 "Convergence zone shows largest drop despite lowest MLP magnitude",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figure5_ablation_improved.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  figure5_ablation_improved.png saved")

# Figure 6: Logit Lens
print("[Figure 6] Logit Lens...")
me = [m for m in models if all_results[m].get("E_logit_lens")]
if me:
    fig, ax = plt.subplots(figsize=(11, 6))
    for m in me:
        ranks = all_results[m]["E_logit_lens"]
        nl    = len(ranks)
        x     = [i/(nl-1) for i in range(nl) if ranks[i] is not None]
        y     = [r for r in ranks if r is not None]
        ax.plot(x, y, marker="o", markersize=3, linewidth=2,
                color=MODEL_COLORS.get(m, "#333"),
                label=MODEL_LABELS.get(m, m), alpha=0.85)
    ax.axvspan(0.10, 0.20, alpha=0.08, color="red")
    ax.axvspan(0.25, 0.75, alpha=0.08, color="blue")
    ax.axvspan(0.80, 0.95, alpha=0.08, color="orange")
    ax.set_xlabel("Normalized Layer Position", fontsize=12)
    ax.set_ylabel("Avg Rank of Target Token\n(lower = better)", fontsize=11)
    ax.set_title("Figure 6: Logit Lens — Progressive Prediction Refinement\n"
                 "Red=Early spike | Blue=Convergence | Orange=Late spike",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, ls="--")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("figure6_logit_lens.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  figure6_logit_lens.png saved")

# Download
try:
    from google.colab import files
    for fname in [OUTPUT_JSON,
                  "figure1.png", "figure2.png", "figure3.png",
                  "figure4_delta_p.png", "figure5_ablation_improved.png",
                  "figure6_logit_lens.png"]:
        if os.path.exists(fname):
            files.download(fname)
except ImportError:
    pass

print("\n✅ All experiments and figures complete!")
