import os
import numpy as np
import pandas as pd
import gradio as gr
from surprise import dump
import os, sys


import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

df = pd.read_csv("preproc.csv")
df["user_id"] = df["user_id"].astype(int)
df["item_id"] = df["item_id"].astype(int)
ITEM_TITLES = {}
if "title" in df.columns:
    ITEM_TITLES = (
        df.dropna(subset=["title"])
        .drop_duplicates("item_id")
        .set_index("item_id")["title"]
        .to_dict()
    )

ITEM_POP = (
    df.groupby("item_id")
    .size()
    .sort_values(ascending=False)
)
POPULAR_ITEM_IDS = ITEM_POP.head(50).index.tolist()

def title_of(iid):
    return ITEM_TITLES.get(iid, f"Item {iid}")

_, algo = dump.load("svdpp_model.pkl")
trainset = algo.trainset
#print(trainset._raw2inner_id_users)

top_users = (
    df.groupby("user_id")["like"].sum()
      .sort_values(ascending=False)
      .head(50)
      .index
      .astype(int)
      .tolist()
)
#print(top_users)
SAMPLE_USER_IDS = list(trainset._raw2inner_id_users.keys())
#SAMPLE_USER_IDS = [str(uid) for uid in SAMPLE_USER_IDS]






RATING_MIN, RATING_MAX = 1, 5
GLOBAL_MEAN = trainset.global_mean

ALL_ITEM_INNER = list(trainset.all_items())
ALL_ITEM_RAW = [trainset.to_raw_iid(ii) for ii in ALL_ITEM_INNER]










USER_SEEN = {}
for u_inner in trainset.all_users():
    uid_raw = trainset.to_raw_uid(u_inner)
    USER_SEEN[uid_raw] = set(trainset.to_raw_iid(ii) for ii, _ in trainset.ur[u_inner])

def to_prob(est, rmin=RATING_MIN, rmax=RATING_MAX):
    est = max(rmin, min(rmax, est))
    return (est - rmin) / (rmax - rmin + 1e-12)

_UID_SAMPLE = next(iter(trainset._raw2inner_id_users.keys()))
_IID_SAMPLE = next(iter(trainset._raw2inner_id_items.keys()))
_UID_IS_STR = isinstance(_UID_SAMPLE, str)
_IID_IS_STR = isinstance(_IID_SAMPLE, str)

def canon_uid(x):
    if _UID_IS_STR:
        s = str(x).strip()
        if s.endswith('.0'):
            s = s[:-2]
        return s
    else:
        return int(float(x))

def canon_iid(x):
    if _IID_IS_STR:
        s = str(x).strip()
        if s.endswith('.0'):
            s = s[:-2]
        return s
    else:
        return int(float(x))

def check_known(uid, iid):
    try:
        _ = trainset.to_inner_uid(uid)
        _ = trainset.to_inner_iid(iid)
        return uid, iid
    except ValueError:
        return None

def resolve_uid(example_uid, manual_uid):
    if manual_uid is not None and manual_uid != 0:
        return manual_uid
    else:
        return example_uid

def resolve_iid(example_iid, manual_iid):
    if manual_iid is not None and manual_iid != 0:
        return manual_iid
    else:
        return example_iid



spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("ALS_COMPARE")
    .getOrCreate()
)

ALS_PATH = "als_model"
als_model = ALSModel.load(ALS_PATH)

def als_prob(x, t=1.0):
    return 1.0 / (1.0 + np.exp(-x / t))

def svd_top_k(user_id, top_k=5):
    uid = canon_uid(user_id)
    try:
        _ = trainset.to_inner_uid(uid)
    except ValueError:
        tops = list(ITEM_POP.index)[:top_k]
        return pd.DataFrame(
            [{"ID ролика": iid, "Название": title_of(iid), "Вероятность": "user id not in dataset"} for iid in tops]
        )

    seen = USER_SEEN.get(uid, set())
    candidates = [iid for iid in ALL_ITEM_RAW if iid not in seen]

    testset = [(uid, iid, GLOBAL_MEAN) for iid in candidates]
    preds = algo.test(testset)
    preds = [p for p in preds if not p.details.get("was_impossible", False)]

    pop_rank = {iid: r for r, iid in enumerate(ITEM_POP.index)}
    preds.sort(key=lambda p: (p.est, -pop_rank.get(p.iid, -10**9)), reverse=True)
    preds = preds[:top_k]

    rows = []
    for p in preds:
        prob = round(to_prob(p.est) * 100, 1)
        rows.append({"ID ролика": p.iid, "Название": title_of(p.iid), "Вероятность": f"{prob}%"})
    return pd.DataFrame(rows)

def als_top_k(user_id, top_k=5):
    # Для ALS user_id должен быть int, как в обучении
    uid_str = canon_uid(user_id)
    print(uid_str,type(uid_str))

    try:
        uid_int = int(float(uid_str))
    except Exception:
        tops = list(ITEM_POP.index)[:top_k]
        return pd.DataFrame(
            [{"ID ролика": iid, "Название": title_of(iid), "Вероятность": "user id not in dataset"} for iid in tops]
        )

    user_df = spark.createDataFrame([(uid_int,)], ["user_id"])
    recs = als_model.recommendForUserSubset(user_df, top_k)
    recs_pd = (
        recs.selectExpr("explode(recommendations) as rec")
            .selectExpr("rec.item_id as item_id", "rec.rating as score")
            .toPandas()
    )

    if recs_pd.empty:
        tops = list(ITEM_POP.index)[:top_k]
        return pd.DataFrame(
            [{"ID ролика": iid, "Название": title_of(iid), "Вероятность": "user id not in dataset"} for iid in tops]
        )

    recs_pd["ID ролика"] = recs_pd["item_id"].astype(str)
    recs_pd["Название"] = recs_pd["ID ролика"].map(title_of)
    recs_pd["Вероятность"] = (als_prob(recs_pd["score"]) * 100).round(1).astype(str) + "%"
    return recs_pd[["ID ролика", "Название", "Вероятность"]]

def check_movie(user_id, movie_id, threshold=0.1):
    uid = canon_uid(user_id)
    iid = canon_iid(movie_id)
    print(iid,type(iid))
    if not check_known(uid, iid):
        p = to_prob(GLOBAL_MEAN)
        conf = int(round(p * 100))
        msg = f" Недостаточно данных, baseline; уверенность: {conf}%"
        return f"Ролик подходит пользователю ({msg})" if p > threshold else f" Ролик не подходит ({msg})"

    pred = algo.predict(uid, iid)
    if pred.details.get("was_impossible", False):
        p = to_prob(GLOBAL_MEAN)
    else:
        p = to_prob(pred.est)
    conf = int(round(p * 100))
    return f"Ролик подходит пользователю (уверенность: {conf}%)" if p > threshold else f" Ролик не подходит (уверенность: {conf}%)"

def recommend_movies(user_id, top_k=5):
    return svd_top_k(user_id, top_k=top_k)

def compare_models(user_id, top_k=5, only_intersection=False):
    als_df = als_top_k(user_id, top_k=top_k)
    svd_df = svd_top_k(user_id, top_k=top_k)

    if only_intersection:
        common_ids = set(als_df["ID ролика"]) & set(svd_df["ID ролика"])
        als_df = als_df[als_df["ID ролика"].isin(common_ids)].reset_index(drop=True)
        svd_df = svd_df[svd_df["ID ролика"].isin(common_ids)].reset_index(drop=True)

    return als_df, svd_df


with gr.Blocks(theme=gr.themes.Soft(), title="Рекомендательная система VK") as demo:
    gr.Markdown("# Рекомендательная система VK")

    with gr.Tab(" Проверка ролика"):
        with gr.Row():
            user_id_1_dd = gr.Dropdown(
                label="User ID (из списка)",
                choices=SAMPLE_USER_IDS,
                value=SAMPLE_USER_IDS[0],
                interactive=True,
            )
            user_id_1_manual = gr.Number(
                label="Или введите User ID вручную (0 = из списка)",
                precision=0,
                value=0,
            )

        with gr.Row():
            movie_id_dd = gr.Dropdown(
                label="Video ID (из списка популярных)",
                choices=POPULAR_ITEM_IDS,
                value=POPULAR_ITEM_IDS[0],
                interactive=True,
            )
            movie_id_manual = gr.Number(
                label="Или введите Video ID вручную (0 = из списка)",
                precision=0,
                value=0,
            )

        out1 = gr.Textbox(label="Результат", interactive=False)

        gr.Button("Проверить").click(
            fn=lambda ex_uid, manual_uid, ex_mid, manual_mid: check_movie(
                resolve_uid(ex_uid, manual_uid),
                resolve_iid(ex_mid, manual_mid),
            ),
            inputs=[user_id_1_dd, user_id_1_manual, movie_id_dd, movie_id_manual],
            outputs=out1,
        )

    with gr.Tab("Рекомендации"):
        with gr.Row():
            user_id_2_dd = gr.Dropdown(
                label="User ID (из списка)",
                choices=SAMPLE_USER_IDS,
                value=SAMPLE_USER_IDS[0],
                interactive=True,
            )
            user_id_2_manual = gr.Number(
                label="Или введите User ID вручную",
                precision=0,
            )

        out2 = gr.Dataframe(label="Рекомендованные ролики")

        gr.Button("Получить рекомендации").click(
            fn=lambda ex_uid, manual_uid: recommend_movies(
                resolve_uid(ex_uid, manual_uid), top_k=5
            ),
            inputs=[user_id_2_dd, user_id_2_manual],
            outputs=out2,
        )

    with gr.Tab("Сравнение моделей"):
        with gr.Row():
            user_id_3_dd = gr.Dropdown(
                label="User ID (из списка)",
                choices=SAMPLE_USER_IDS,
                value=SAMPLE_USER_IDS[0],
                interactive=True,
            )
            user_id_3_manual = gr.Number(
                label="Или введите User ID вручную",
                precision=0,
            )

        top_k_3 = gr.Slider(
            minimum=1,
            maximum=500,
            value=5,
            step=1,
            label="Размер выборки (Top-K)",
        )
        only_common_3 = gr.Checkbox(
            label="Показывать только общие ролики (ALS ∩ SVD++)",
            value=False,
        )

        with gr.Row():
            table_als = gr.Dataframe(label="Результаты ALS")
            table_svd = gr.Dataframe(label="Результаты SVD++")

        gr.Button("Сравнить модели").click(
            fn=lambda ex_uid, manual_uid, top_k, only_common: compare_models(
                resolve_uid(ex_uid, manual_uid),
                top_k=top_k,
                only_intersection=only_common,
            ),
            inputs=[user_id_3_dd, user_id_3_manual, top_k_3, only_common_3],
            outputs=[table_als, table_svd],
        )

    with gr.Tab("О нас"):
        gr.Markdown("""
            # О проекте Рекомендательная система VK
        
            Этот проект — интерфейс для тестирования рекомендательных алгоритмов (ALS, SVD++).  
            Он создан для удобной проверки моделей, сравнения их качества и визуализации результатов.
        
        
            ## Цель проекта
            Создать удобную и понятную платформу для тестирования моделей рекомендаций  
            и упростить процесс взаимодействия с ML backend
        
        
            ## Авторы
            - **Коршиков Евгений** — разработка интерфейса  
            - **Нагель Аркадий** — мл инженер  
            - Использованы библиотеки: Surprise, PySpark, Gradio
        
            """)

demo.launch()
