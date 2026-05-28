# ================================================================
# PC-Transformer Interpretability — Unified Experiment Script
# Memory-optimized: one model per run, aggressive cache clearing
# ================================================================
# Usage:
#   1. Set RUN_MODEL below
#   2. Run all cells
#   3. Repeat with next model (Runtime -> Factory Reset between models)
# ================================================================

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
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── ★ 여기만 바꾸고 실행 ★ ────────────────────────────────────
RUN_MODEL = "gpt2"
# "gpt2" / "gpt2-medium" / "gpt2-large"
# "EleutherAI/gpt-neo-125M"
# "EleutherAI/pythia-160m" / "EleutherAI/pythia-410m"
# ─────────────────────────────────────────────────────────────

OUTPUT_JSON = "multi_model_results_v2.json"
LANGS = ["en", "es", "fr", "ko", "ru", "ja", "ar"]

# ── Prompts ───────────────────────────────────────────────────
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
PROMPTS_EN = [p for _, p in PROMPTS_A if _ == "en"]

PAIRS_B = [
    {"clean": "The capital of France is Paris",
     "corrupt": "The capital of Germany is Berlin",
     "clean_token": " Paris", "corrupt_token": " Berlin", "name": "cap_france"},
    {"clean": "The capital of Japan is Tokyo",
     "corrupt": "The capital of China is Beijing",
     "clean_token": " Tokyo", "corrupt_token": " Beijing", "name": "cap_japan"},
    {"clean": "The capital of Italy is Rome",
     "corrupt": "The capital of Spain is Madrid",
     "clean_token": " Rome", "corrupt_token": " Madrid", "name": "cap_italy"},
    {"clean": "The capital of Greece is Athens",
     "corrupt": "The capital of Turkey is Ankara",
     "clean_token": " Athens", "corrupt_token": " Ankara", "name": "cap_greece"},
    {"clean": "The language of France is French",
     "corrupt": "The language of Germany is German",
     "clean_token": " French", "corrupt_token": " German", "name": "lang_france"},
]

PAIRS_EXTENDED = [
    ("The capital of France is"," Paris","The capital of Spain is"," Madrid","capital"),
    ("The capital of Japan is"," Tokyo","The capital of China is"," Beijing","capital"),
    ("The capital of Italy is"," Rome","The capital of Greece is"," Athens","capital"),
    ("The capital of Germany is"," Berlin","The capital of France is"," Paris","capital"),
    ("The capital of Russia is"," Moscow","The capital of Germany is"," Berlin","capital"),
    ("The capital of Egypt is"," Cairo","The capital of Russia is"," Moscow","capital"),
    ("The capital of Greece is"," Athens","The capital of Italy is"," Rome","capital"),
    ("The capital of Spain is"," Madrid","The capital of Japan is"," Tokyo","capital"),
    ("The capital of China is"," Beijing","The capital of Egypt is"," Cairo","capital"),
    ("The capital of India is"," Delhi","The capital of China is"," Beijing","capital"),
    ("The currency of Japan is the"," yen","The currency of China is the"," yuan","currency"),
    ("The currency of UK is the"," pound","The currency of Japan is the"," yen","currency"),
    ("The currency of China is the"," yuan","The currency of UK is the"," pound","currency"),
    ("The currency of Russia is the"," ruble","The currency of India is the"," rup","currency"),
    ("The currency of Mexico is the"," peso","The currency of Russia is the"," ruble","currency"),
    ("The currency of USA is the"," dollar","The currency of Mexico is the"," peso","currency"),
    ("The currency of Korea is the"," won","The currency of USA is the"," dollar","currency"),
    ("People in France speak"," French","People in Germany speak"," German","language"),
    ("People in Germany speak"," German","People in France speak"," French","language"),
    ("People in Japan speak"," Japanese","People in China speak"," Chinese","language"),
    ("People in China speak"," Chinese","People in Japan speak"," Japanese","language"),
    ("People in Italy speak"," Italian","People in France speak"," French","language"),
    ("People in Russia speak"," Russian","People in Italy speak"," Italian","language"),
    ("People in Spain speak"," Spanish","People in Russia speak"," Russian","language"),
    ("People in Greece speak"," Greek","People in Spain speak"," Spanish","language"),
    ("The color of grass is"," green","The color of the sky is"," blue","color"),
    ("The color of the sky is"," blue","The color of grass is"," green","color"),
    ("The color of snow is"," white","The color of coal is"," black","color"),
    ("The color of coal is"," black","The color of snow is"," white","color"),
    ("The color of blood is"," red","The color of snow is"," white","color"),
    ("The color of the sun is"," yellow","The color of blood is"," red","color"),
    ("Water freezes at"," zero","Water boils at"," one","science"),
    ("The opposite of hot is"," cold","The opposite of big is"," small","antonym"),
    ("The opposite of day is"," night","The opposite of hot is"," cold","antonym"),
]

# ── Utilities ─────────────────────────────────────────────────

def is_single_token(model, word):
    try:
        model.to_single_token(word)
        return True
    except:
        return False

def get_zone_layers(n, zone):
    lo, hi = {"early_spike":(0.10,0.20),
               "convergence":(0.25,0.75),
               "late_spike": (0.80,0.95)}[zone]
    return [l for l in range(n) if lo <= l/n <= hi]

def clear():
    torch.cuda.empty_cache()
    gc.collect()

# ── Experiment 1: Residual Stream Convergence ─────────────────

def run_exp1(model, n_layers):
    print("  [Exp 1] Residual stream convergence...")
    results = []
    # names_filter: only cache what we need
    needed = (
        [f"blocks.{l}.hook_resid_pre"  for l in range(n_layers)] +
        [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    )
    for idx, (lang, prompt) in enumerate(PROMPTS_A):
        try:
            tokens = model.to_tokens(prompt)[:, 1:].to(device)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=lambda n: n in needed)

            sims = []
            for l in range(1, n_layers):
                prev = cache[f"blocks.{l-1}.hook_resid_post"][0].cpu()
                curr = cache[f"blocks.{l}.hook_resid_post"][0].cpu()
                sims.append(
                    torch.nn.functional.cosine_similarity(
                        prev, curr, dim=-1).numpy())
            sims = np.array(sims)

            deltas = []
            for l in range(n_layers):
                pre  = cache[f"blocks.{l}.hook_resid_pre"][0].cpu()
                post = cache[f"blocks.{l}.hook_resid_post"][0].cpu()
                deltas.append(torch.norm(post-pre, p=2, dim=-1).numpy())
            deltas = np.array(deltas)

            del cache; clear()

            early_d = deltas[:max(1,int(n_layers*.4))].flatten()
            late_d  = deltas[int(n_layers*.6):].flatten()
            _, dp   = stats.ttest_ind(early_d, late_d)
            mid_s   = sims[int(n_layers*.25):int(n_layers*.75)].flatten()
            results.append({"lang":lang,
                            "middle_sim_mean":float(mid_s.mean()),
                            "delta_p":float(dp)})
        except Exception as e:
            pass
        if (idx+1) % 20 == 0:
            print(f"    {idx+1}/{len(PROMPTS_A)}")

    df = pd.DataFrame(results)
    summary = df.groupby("lang")[["middle_sim_mean","delta_p"]].mean()
    print(summary.to_string()); return summary.to_dict(), results

# ── Experiment 2: Activation Patching ────────────────────────

def run_exp2(model, n_layers):
    print("  [Exp 2] Activation patching...")
    b_results = {}
    for pair in PAIRS_B:
        if not (is_single_token(model, pair["clean_token"]) and
                is_single_token(model, pair["corrupt_token"])):
            continue
        try:
            ct  = model.to_single_token(pair["clean_token"])
            cot = model.to_single_token(pair["corrupt_token"])
            ctoks = model.to_tokens(pair["clean"])[:,1:].to(device)
            xtoks = model.to_tokens(pair["corrupt"])[:,1:].to(device)
            sl    = min(ctoks.shape[1], xtoks.shape[1])
            ctoks, xtoks = ctoks[:,:sl], xtoks[:,:sl]

            hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
            with torch.no_grad():
                _, clean_cache = model.run_with_cache(
                    ctoks, names_filter=lambda n: n in hook_names)

            n_pos   = xtoks.shape[1]
            effects = torch.zeros(n_layers, n_pos)

            for l in range(n_layers):
                hn = f"blocks.{l}.hook_resid_post"
                for pos in range(n_pos):
                    cv = clean_cache[hn][0, pos, :].detach().clone()
                    def make_hook(c, p):
                        def fn(v, hook):
                            v = v.clone(); v[0,p,:] = c; return v
                        return fn
                    with torch.no_grad():
                        out = model.run_with_hooks(
                            xtoks, fwd_hooks=[(hn, make_hook(cv, pos))])
                    effects[l, pos] = (out[0,-1,ct] - out[0,-1,cot]).item()
                    del out; clear()

            del clean_cache; clear()
            b_results[pair["name"]] = effects.numpy().tolist()
            print(f"    done: {pair['name']}")
        except Exception as e:
            print(f"    skip {pair['name']}: {e}")
    return b_results

# ── Experiment 3: MLP Magnitude ──────────────────────────────

def run_exp3(model, n_layers):
    print("  [Exp 3] MLP magnitude...")
    needed = (
        [f"blocks.{l}.ln2.hook_normalized" for l in range(n_layers)] +
        [f"blocks.{l}.hook_mlp_out"         for l in range(n_layers)]
    )
    all_norms = []
    for prompt in PROMPTS_EN:
        try:
            tokens = model.to_tokens(prompt)[:,1:].to(device)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=lambda n: n in needed)
            norms = []
            for l in range(n_layers):
                try:
                    mi = cache[f"blocks.{l}.ln2.hook_normalized"][0].cpu()
                    mo = cache[f"blocks.{l}.hook_mlp_out"][0].cpu()
                    norms.append(torch.norm(mo-mi, dim=-1).mean().item())
                except:
                    norms.append(0.0)
            all_norms.append(norms)
            del cache; clear()
        except:
            pass
    return np.mean(all_norms, axis=0).tolist() if all_norms else [0.]*n_layers

# ── Experiment 4: Zero-Ablation ──────────────────────────────

def run_exp4(model, n_layers):
    print("  [Exp 4] Zero-ablation...")
    zones     = ["early_spike","convergence","late_spike"]
    zone_data = {z:[] for z in zones}
    valid     = [p for p in PAIRS_EXTENDED
                 if is_single_token(model,p[1]) and is_single_token(model,p[3])]
    print(f"    valid pairs: {len(valid)}")

    def zero_hook(v, hook): return torch.zeros_like(v)

    skipped = 0
    for cp, ct_s, xp, xt_s, cat in valid:
        try:
            ct   = model.to_single_token(ct_s)
            xt   = model.to_single_token(xt_s)
            toks = model.to_tokens(cp)[:,1:].to(device)
            with torch.no_grad():
                bl  = model(toks)
            base = (bl[0,-1,ct] - bl[0,-1,xt]).item()
            del bl; clear()
            if base < 1.0:
                skipped += 1; continue
            for zone in zones:
                layers = get_zone_layers(n_layers, zone)
                hooks  = [(f"blocks.{l}.hook_mlp_out", zero_hook) for l in layers]
                with torch.no_grad():
                    ab = model.run_with_hooks(toks, fwd_hooks=hooks)
                drop = (base - (ab[0,-1,ct]-ab[0,-1,xt]).item()) / abs(base)
                zone_data[zone].append({"drop":drop,"category":cat})
                del ab; clear()
        except:
            skipped += 1
    print(f"    skipped: {skipped}")
    result = {}
    for zone in zones:
        drops = [d["drop"] for d in zone_data[zone]]
        if drops:
            result[zone] = {"mean_drop":float(np.mean(drops)),
                            "std_drop": float(np.std(drops)),
                            "n_pairs":  len(drops),
                            "raw":      drops}
            print(f"    {zone}: {result[zone]['mean_drop']:+.3f}"
                  f" ± {result[zone]['std_drop']:.3f} (n={len(drops)})")
    return result

# ── Experiment 5: Logit Lens ──────────────────────────────────

def run_exp5(model, n_layers):
    print("  [Exp 5] Logit lens...")
    W_U = model.W_U.detach()
    b_U = model.b_U.detach()
    layer_ranks = [[] for _ in range(n_layers)]
    needed = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]

    for prompt in PROMPTS_EN[:15]:
        try:
            tokens     = model.to_tokens(prompt).to(device)
            if tokens.shape[1] < 4: continue
            target_tok = tokens[0,-1].item()
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens[:,:-1],
                    names_filter=lambda n: n in needed)
            for l in range(n_layers):
                key = f"blocks.{l}.hook_resid_post"
                if key not in cache: continue
                r   = cache[key][0,-1,:].float()
                r   = (r - r.mean()) / (r.std() + 1e-8)
                lg  = r @ W_U.float() + b_U.float()
                rank = (lg.argsort(descending=True) == target_tok
                        ).nonzero(as_tuple=True)[0]
                if len(rank) > 0:
                    layer_ranks[l].append(rank[0].item())
            del cache; clear()
        except:
            continue
    avg = [float(np.mean(r)) if r else None for r in layer_ranks]
    valid = [r for r in avg if r is not None]
    if valid: print(f"    L0≈{valid[0]:.0f} → Lfinal≈{valid[-1]:.2f}")
    return avg

# ── Main ──────────────────────────────────────────────────────

short = RUN_MODEL.split("/")[-1]

# Load existing
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON) as f:
        all_results = json.load(f)
else:
    all_results = {}

if short in all_results:
    print(f"[{short}] already in JSON. Delete the key to re-run.")
else:
    print(f"\n{'='*55}\nModel: {RUN_MODEL}\n{'='*55}")
    model = HookedTransformer.from_pretrained(RUN_MODEL, device=device)
    model.eval()
    n_layers = model.cfg.n_layers
    print(f"Layers: {n_layers} | "
          f"Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    sumA, resA = run_exp1(model, n_layers)
    resB       = run_exp2(model, n_layers)
    resC       = run_exp3(model, n_layers)
    resD       = run_exp4(model, n_layers)
    resE       = run_exp5(model, n_layers)

    all_results[short] = {
        "model_full_name": RUN_MODEL,
        "n_layers":        n_layers,
        "A_summary":       sumA,
        "A_results":       resA,
        "B_results":       resB,
        "C_mlp_norms":     resC,
        "D_zero_ablation": resD,
        "E_logit_lens":    resE,
    }

    del model; clear()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ [{short}] saved to {OUTPUT_JSON}")

print(f"\nCompleted models so far: {list(all_results.keys())}")

# ── Figures (runs after each model, updates automatically) ────

data   = all_results
models = list(data.keys())
if not models:
    print("No data to plot yet.")
else:
    ML = {"gpt2":"GPT-2 (117M)","gpt2-medium":"GPT-2 Med (345M)",
          "gpt2-large":"GPT-2 Large (774M)","gpt-neo-125M":"GPT-Neo (125M)",
          "pythia-160m":"Pythia (160M)","pythia-410m":"Pythia (410M)"}
    LC = {"en":"#1f77b4","es":"#ff7f0e","fr":"#2ca02c",
          "ko":"#d62728","ru":"#9467bd","ja":"#8c564b","ar":"#e377c2"}
    MC = {"gpt2":"#1f77b4","gpt2-medium":"#aec7e8","gpt2-large":"#0a4a7a",
          "gpt-neo-125M":"#d62728","pythia-160m":"#9467bd","pythia-410m":"#8c564b"}
    ZC = {"early_spike":"#e74c3c","convergence":"#3498db","late_spike":"#e67e22"}

    # Fig 1
    n = len(models)
    fig,axes = plt.subplots(1,n,figsize=(4.5*n,5.5),sharey=False)
    if n==1: axes=[axes]
    for ax,m in zip(axes,models):
        s   = data[m]["A_summary"]["middle_sim_mean"]
        vs  = [s.get(l,0) for l in LANGS]
        bars= ax.bar(range(7),vs,color=[LC[l] for l in LANGS],
                     alpha=.88,width=.7,edgecolor="white")
        bars[0].set_edgecolor("black"); bars[0].set_linewidth(2)
        mn,mx = min(v for v in vs if v>0),max(vs)
        mg = (mx-mn)*.3
        ax.set_ylim(max(0,mn-mg),min(1,mx+mg*.5))
        ax.set_xticks(range(7)); ax.set_xticklabels(LANGS,fontsize=9)
        enr = sorted(vs,reverse=True).index(s.get("en",0))+1
        ax.set_title(f"{ML.get(m,m)}\n(en #{enr}/7)",fontsize=10,fontweight="bold")
        ax.set_xlabel("Language",fontsize=9)
        ax.set_ylabel("Mid-Layer Cosine Sim",fontsize=8)
        ax.grid(axis="y",alpha=.3,ls="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ev = s.get("en",0)
        ax.annotate(f"{ev:.3f}",xy=(0,ev),xytext=(0,4),
                    textcoords="offset points",ha="center",fontsize=7,fontweight="bold")
    hs = [plt.Rectangle((0,0),1,1,color=LC[l]) for l in LANGS]
    fig.legend(hs,LANGS,loc="upper right",bbox_to_anchor=(1,.02),ncol=1,fontsize=9)
    fig.suptitle("Figure 1: Mid-Layer Cosine Similarity\n(independent y-axes)",
                 fontsize=12,fontweight="bold",y=1.03)
    plt.tight_layout()
    plt.savefig("figure1.png",dpi=150,bbox_inches="tight",facecolor="white")
    print("figure1.png saved")

    # Fig 2
    fams = {"GPT-2 Family":[m for m in models if "gpt2" in m],
            "GPT-Neo Family":[m for m in models if "neo" in m],
            "Pythia Family":[m for m in models if "pythia" in m]}
    fams = {k:v for k,v in fams.items() if v}
    if fams:
        fig,axes = plt.subplots(1,len(fams),figsize=(7*len(fams),5.5))
        if len(fams)==1: axes=[axes]
        for ax,(title,grp) in zip(axes,fams.items()):
            for m in grp:
                ns = data[m]["C_mlp_norms"]
                x  = [i/len(ns) for i in range(len(ns))]
                ax.plot(x,ns,marker="o",markersize=3,linewidth=2,
                        color=MC.get(m,"#333"),label=ML.get(m,m),alpha=.9)
            ax.axvspan(.10,.20,alpha=.08,color="red")
            ax.axvspan(.80,.95,alpha=.08,color="blue")
            ax.set_xlabel("Normalized Layer Position",fontsize=11)
            ax.set_ylabel("MLP Transform Magnitude",fontsize=11)
            ax.set_title(title,fontsize=12,fontweight="bold")
            ax.legend(fontsize=9); ax.grid(alpha=.3,ls="--")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.suptitle("Figure 2: MLP Transform Magnitude",fontsize=13,fontweight="bold")
        plt.tight_layout()
        plt.savefig("figure2.png",dpi=150,bbox_inches="tight",facecolor="white")
        print("figure2.png saved")

    # Fig 3
    def avg_patch(br):
        ms=[]
        for _,mat in br.items():
            m=np.array(mat); mn,mx=m.min(),m.max()
            ms.append((m-mn)/(mx-mn) if mx-mn>1e-6 else np.zeros_like(m))
        if not ms: return None
        mp=max(m.shape[1] for m in ms); nl=ms[0].shape[0]
        p=np.zeros((len(ms),nl,mp))
        for i,m in enumerate(ms): p[i,:,mp-m.shape[1]:]=m
        return np.mean(p,axis=0)
    mb=[m for m in models if data[m].get("B_results")]
    if mb:
        fig,axes=plt.subplots(1,len(mb),figsize=(4.8*len(mb),5.5))
        if len(mb)==1: axes=[axes]
        lim=None
        for ax,m in zip(axes,mb):
            mat=avg_patch(data[m]["B_results"])
            if mat is None: continue
            nl=mat.shape[0]
            lim=ax.imshow(mat,aspect="auto",cmap="RdBu_r",vmin=0,vmax=1,
                          origin="lower",interpolation="nearest")
            ax.axhline(y=nl*.25-.5,color="yellow",lw=1.8,ls="--",alpha=.95)
            ax.axhline(y=nl*.33+.5,color="yellow",lw=1.8,ls="--",alpha=.95)
            ax.set_title(ML.get(m,m),fontsize=11,fontweight="bold")
            ax.set_xlabel("Token Position",fontsize=10)
            if ax==axes[0]: ax.set_ylabel("Layer",fontsize=10)
            st=max(1,nl//6)
            ax.set_yticks(range(0,nl,st))
            ax.set_yticklabels([f"L{i}" for i in range(0,nl,st)],fontsize=8)
        if lim: plt.colorbar(lim,ax=axes[-1],fraction=.046,pad=.04,
                              label="Normalized Patching Effect")
        fig.suptitle("Figure 3: Activation Patching\nYellow = norm. pos 0.25–0.33",
                     fontsize=13,fontweight="bold")
        plt.tight_layout()
        plt.savefig("figure3.png",dpi=150,bbox_inches="tight",facecolor="white")
        print("figure3.png saved")

    # Fig 4
    dm,vm=[],[]
    for m in models:
        dm.append([data[m]["A_summary"].get("delta_p",{}).get(l,1.) for l in LANGS])
        vm.append(ML.get(m,m))
    fig,ax=plt.subplots(figsize=(10,max(4,len(vm)*1.4)))
    im=ax.imshow(dm,cmap="RdYlGn_r",vmin=0,vmax=.1,aspect="auto")
    ax.set_xticks(range(7)); ax.set_xticklabels(LANGS,fontsize=12)
    ax.set_yticks(range(len(vm))); ax.set_yticklabels(vm,fontsize=10)
    for i,row in enumerate(dm):
        for j,v in enumerate(row):
            t=f"{v:.3f}" if v>=.001 else "<.001"
            c="white" if v<.02 or v>.08 else "black"
            ax.text(j,i,t,ha="center",va="center",fontsize=8,color=c,fontweight="bold")
    plt.colorbar(im,ax=ax,fraction=.03,pad=.02,
                 label="delta p-value\n(green=sig, red=not sig)")
    ax.set_title("Figure 4: delta_p by Language and Model (Appendix)",
                 fontsize=13,fontweight="bold")
    plt.tight_layout()
    plt.savefig("figure4_delta_p.png",dpi=150,bbox_inches="tight",facecolor="white")
    print("figure4_delta_p.png saved")

    # Fig 5 (y-axis clipped, asterisk for outliers)
    zones=["early_spike","convergence","late_spike"]
    ZL={"early_spike":"Early\nSpike\n(0.10–0.20)",
        "convergence":"Convergence\n(0.25–0.75)",
        "late_spike": "Late\nSpike\n(0.80–0.95)"}
    MODEL_ORDER=["gpt2","gpt2-medium","gpt2-large",
                 "gpt-neo-125M","pythia-160m","pythia-410m"]
    md=[m for m in MODEL_ORDER if m in data and data[m].get("D_zero_ablation")]
    if md:
        fig,axes=plt.subplots(1,len(md),figsize=(4*len(md),5.5),sharey=True)
        if len(md)==1: axes=[axes]
        for ax,m in zip(axes,md):
            r=data[m]["D_zero_ablation"]
            ms=[r.get(z,{}).get("mean_drop",0) for z in zones]
            ss=[r.get(z,{}).get("std_drop",0)  for z in zones]
            ns=[r.get(z,{}).get("n_pairs",0)   for z in zones]
            clipped=[max(-0.5,min(1.5,v)) for v in ms]
            clipped_s=[min(s,0.5) for s in ss]
            bars=ax.bar(range(3),clipped,color=[ZC[z] for z in zones],
                        alpha=.85,width=.6,yerr=clipped_s,capsize=5,edgecolor="white")
            ax.axhline(0,color="black",lw=1,ls="--",alpha=.6)
            ax.set_ylim(-0.5,1.5)
            ax.set_xticks(range(3)); ax.set_xticklabels([ZL[z] for z in zones],fontsize=9)
            ax.set_title(ML.get(m,m),fontsize=11,fontweight="bold")
            if ax==axes[0]: ax.set_ylabel("Performance Drop\n(+=harmful, -=helpful)",fontsize=10)
            ax.grid(axis="y",alpha=.3,ls="--")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            for bar,clip,orig,nv in zip(bars,clipped,ms,ns):
                label=f"{orig:+.2f}*\n(n={nv})" if abs(orig)>1.5 else f"{orig:+.2f}\n(n={nv})"
                ax.text(bar.get_x()+bar.get_width()/2,
                        max(-0.45,min(1.45,clip))+0.03,
                        label,ha="center",fontsize=8,fontweight="bold")
        fig.suptitle("Figure 5: Zero-Ablation of MLP Zones\n"
                     "Convergence zone shows largest drop despite lowest MLP magnitude "
                     "(*value clipped for display)",
                     fontsize=12,fontweight="bold")
        plt.tight_layout()
        plt.savefig("figure5_ablation_improved.png",dpi=150,
                    bbox_inches="tight",facecolor="white")
        print("figure5_ablation_improved.png saved")

    # Fig 6 (split panels: GPT-2 family / GPT-Neo & Pythia)
    gpt2_m=[m for m in MODEL_ORDER if "gpt2" in m and m in data and data[m].get("E_logit_lens")]
    other_m=[m for m in MODEL_ORDER if "gpt2" not in m and m in data and data[m].get("E_logit_lens")]
    if gpt2_m or other_m:
        fig,(ax_l,ax_r)=plt.subplots(1,2,figsize=(16,6))
        for m in gpt2_m:
            ranks=data[m]["E_logit_lens"]; nl=len(ranks)
            x=[i/(nl-1) for i in range(nl) if ranks[i] is not None]
            y=[r for r in ranks if r is not None]
            ax_l.plot(x,y,marker="o",markersize=4,linewidth=2.2,
                      color=MC.get(m,"#333"),label=ML.get(m,m),alpha=.9)
        for ax in [ax_l]:
            ax.axvspan(.10,.20,alpha=.08,color="red")
            ax.axvspan(.25,.75,alpha=.08,color="blue")
            ax.axvspan(.80,.95,alpha=.08,color="orange")
            ax.set_xlabel("Normalized Layer Position",fontsize=12)
            ax.set_ylabel("Avg Rank of Target Token\n(lower = better)",fontsize=11)
            ax.legend(fontsize=10); ax.grid(alpha=.3,ls="--"); ax.invert_yaxis()
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax_l.set_title("GPT-2 Family",fontsize=13,fontweight="bold")
        for m in other_m:
            ranks=data[m]["E_logit_lens"]; nl=len(ranks)
            x=[i/(nl-1) for i in range(nl) if ranks[i] is not None]
            y=[r for r in ranks if r is not None]
            ax_r.plot(x,y,marker="o",markersize=4,linewidth=2.2,
                      color=MC.get(m,"#333"),label=ML.get(m,m),alpha=.9)
        ax_r.axvspan(.10,.20,alpha=.08,color="red")
        ax_r.axvspan(.25,.75,alpha=.08,color="blue")
        ax_r.axvspan(.80,.95,alpha=.08,color="orange")
        ax_r.set_xlabel("Normalized Layer Position",fontsize=12)
        ax_r.set_ylabel("Avg Rank of Target Token\n(lower = better)",fontsize=11)
        ax_r.set_title("GPT-Neo & Pythia Family",fontsize=13,fontweight="bold")
        ax_r.legend(fontsize=10); ax_r.grid(alpha=.3,ls="--"); ax_r.invert_yaxis()
        ax_r.spines["top"].set_visible(False); ax_r.spines["right"].set_visible(False)
        fig.suptitle("Figure 6: Logit Lens — Progressive Prediction Refinement\n"
                     "Red=Early spike | Blue=Convergence | Orange=Late spike  "
                     "(Left: GPT-2 family shows clear monotonic refinement)",
                     fontsize=13,fontweight="bold")
        plt.tight_layout()
        plt.savefig("figure6_logit_lens.png",dpi=150,
                    bbox_inches="tight",facecolor="white")
        print("figure6_logit_lens.png saved")

# Download
try:
    from google.colab import files
    for fn in [OUTPUT_JSON,"figure1.png","figure2.png","figure3.png",
               "figure4_delta_p.png","figure5_ablation_improved.png",
               "figure6_logit_lens.png"]:
        if os.path.exists(fn): files.download(fn)
except ImportError:
    pass

print("\n✅ Done!")
