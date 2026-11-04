#!/usr/bin/env python3
"""
Разведочный анализ (EDA) датасета оттока игроков.

Запуск:
    python eda_churn.py путь/к/файлу.csv
    python eda_churn.py путь/к/файлу.csv --out eda_output

Результат (папка --out, по умолчанию eda_output/):
    eda_report.md   - текстовый отчёт с цифрами (можно вставлять в документ)
    *.png           - графики

Ожидаемые столбцы: UserID, ServerTime, Level, MaxLevel, EndType,
RollingLosses, PlayTime, UsedCoins, ExtraMoves, BuyMoreMoves,
WatchedVideo, Platform, Churn. Отсутствующие столбцы пропускаются.
Лишние столбцы тоже учитываются в общих проверках.

Зависимости: pandas, numpy, matplotlib
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NUM_COLS = ["Level", "MaxLevel", "RollingLosses", "PlayTime",
            "UsedCoins", "ExtraMoves", "BuyMoreMoves", "WatchedVideo"]


class Report:
    """Собирает markdown-отчёт и дублирует его в консоль."""

    def __init__(self):
        self.lines = []

    def h(self, text, level=2):
        self.add("\n" + "#" * level + " " + text + "\n")

    def add(self, text=""):
        print(text)
        self.lines.append(str(text))

    def table(self, df):
        self.add(df.to_markdown() if _has_tabulate() else df.to_string())
        self.add()

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def savefig(out, name):
    plt.tight_layout()
    plt.savefig(os.path.join(out, name), dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="путь к CSV-файлу")
    ap.add_argument("--out", default="eda_output", help="папка для результатов")
    ap.add_argument("--sep", default=",", help="разделитель CSV")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    r = Report()

    df = pd.read_csv(args.csv, sep=args.sep, low_memory=False)
    r.h("Разведочный анализ датасета", 1)

    # ------------------------------------------------------------------ 1
    r.h("1. Размер и структура")
    r.add(f"- Строк: **{len(df):,}**")
    r.add(f"- Столбцов: **{df.shape[1]}**")
    if "UserID" in df:
        r.add(f"- Уникальных игроков: **{df['UserID'].nunique():,}**")
        per_user = df.groupby("UserID").size()
        r.add(f"- Строк (уровней) на игрока: медиана {per_user.median():.0f}, "
              f"среднее {per_user.mean():.1f}, максимум {per_user.max()}")
    r.add()
    info = pd.DataFrame({"dtype": df.dtypes.astype(str),
                         "пропусков": df.isna().sum(),
                         "% пропусков": (df.isna().mean() * 100).round(2),
                         "уникальных": df.nunique()})
    r.table(info)

    # ------------------------------------------------------------------ 2
    r.h("2. Качество данных")
    dups = int(df.duplicated().sum())
    r.add(f"- Полных дубликатов строк: **{dups:,}**")
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    r.add(f"- Столбцы с одним значением (неинформативные): "
          f"{const_cols if const_cols else 'нет'}")
    if "ServerTime" in df:
        df["ServerTime"] = pd.to_datetime(df["ServerTime"], errors="coerce")
        bad = int(df["ServerTime"].isna().sum())
        r.add(f"- Период данных: {df['ServerTime'].min()} … {df['ServerTime'].max()}")
        r.add(f"- Нераспознанных дат: {bad:,}")

    # ------------------------------------------------------------------ 3
    r.h("3. Целевая переменная Churn")
    if "Churn" in df:
        vc = df["Churn"].value_counts(dropna=False).sort_index()
        share = (vc / len(df) * 100).round(3)
        r.table(pd.DataFrame({"строк": vc, "% строк": share}))
        base = df["Churn"].value_counts(normalize=True).max()
        r.add(f"- Accuracy фиктивной модели «всегда мажоритарный класс»: "
              f"**{base * 100:.2f}%**. Поэтому accuracy как метрика не годится, "
              f"нужны PR-AUC, recall/precision по классу оттока.")
        if "UserID" in df:
            u = df.groupby("UserID")["Churn"].max()
            r.add(f"- Доля оттока **по игрокам** (хоть раз Churn=1): "
                  f"**{u.mean() * 100:.2f}%** "
                  f"({int(u.sum()):,} из {len(u):,})")
            mix = df.groupby("UserID")["Churn"].nunique()
            r.add(f"- Игроков, у которых Churn меняется внутри истории: "
                  f"{int((mix > 1).sum()):,} "
                  f"(если много, Churn считается по скользящему моменту, "
                  f"а не по игроку целиком; важно для разбиения без утечки)")
        plt.figure(figsize=(4, 4))
        vc.plot(kind="bar", color=["#4c78a8", "#e45756"][:len(vc)])
        plt.title("Churn: число строк")
        savefig(args.out, "churn_balance.png")
    else:
        r.add("Столбец Churn не найден.")

    # ------------------------------------------------------------------ 4
    r.h("4. Числовые признаки")
    num = [c for c in NUM_COLS if c in df]
    if num:
        desc = df[num].describe(percentiles=[.5, .9, .99]).T
        r.table(desc.round(3))
        skew = df[num].skew().round(2)
        r.add("Асимметрия (skew > 2 - сильно скошенное, возможен log-преобразование):")
        r.add(", ".join(f"{k}={v}" for k, v in skew.items()))
        r.add()
        cols = 3
        rows = int(np.ceil(len(num) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        for ax, c in zip(np.ravel(axes), num):
            s = df[c].dropna()
            s = s[s <= s.quantile(0.99)]
            ax.hist(s, bins=40, color="#4c78a8")
            ax.set_title(c)
        for ax in np.ravel(axes)[len(num):]:
            ax.axis("off")
        savefig(args.out, "numeric_distributions.png")

    # ------------------------------------------------------------------ 5
    r.h("5. Категориальные признаки")
    for c in ["EndType", "Platform"]:
        if c in df:
            vc = df[c].value_counts(dropna=False)
            r.add(f"**{c}**")
            r.table(pd.DataFrame({"строк": vc,
                                  "%": (vc / len(df) * 100).round(2)}))

    # ------------------------------------------------------------------ 6
    r.h("6. Связь признаков с Churn")
    if "Churn" in df:
        for c in ["Platform", "EndType"]:
            if c in df:
                g = df.groupby(c)["Churn"].agg(["mean", "sum", "count"])
                g["mean"] = (g["mean"] * 100).round(3)
                g.columns = ["% оттока", "отток (строк)", "всего строк"]
                r.add(f"**Доля оттока по {c}**")
                r.table(g)
        if num:
            m = df.groupby("Churn")[num].mean().T.round(3)
            r.add("**Средние значения признаков по классам Churn**")
            r.table(m)
            corr = df[num + ["Churn"]].corr(numeric_only=True)["Churn"] \
                .drop("Churn").sort_values(key=abs, ascending=False).round(4)
            r.add("**Корреляция с Churn (Пирсон)**")
            r.table(corr.to_frame("corr"))
            plt.figure(figsize=(5, 4))
            corr.plot(kind="barh", color="#e45756")
            plt.title("Корреляция признаков с Churn")
            savefig(args.out, "corr_with_churn.png")

    # ------------------------------------------------------------------ 7
    r.h("7. Мультиколлинеарность")
    if len(num) > 1:
        cm = df[num].corr()
        pairs = (cm.where(np.triu(np.ones(cm.shape), 1).astype(bool))
                 .stack().rename("corr"))
        top = pairs.loc[pairs.abs().sort_values(ascending=False).index[:8]]
        r.add("Пары признаков с наибольшей корреляцией:")
        r.table(top.round(3).to_frame())
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
        plt.xticks(range(len(num)), num, rotation=90)
        plt.yticks(range(len(num)), num)
        plt.colorbar()
        plt.title("Корреляционная матрица")
        savefig(args.out, "corr_matrix.png")

    # ------------------------------------------------------------------ 8
    r.h("8. Динамика во времени")
    if "ServerTime" in df and df["ServerTime"].notna().any():
        day = df.set_index("ServerTime").resample("D").size()
        plt.figure(figsize=(8, 3))
        day.plot(color="#4c78a8")
        plt.title("Сыгранных уровней по дням")
        savefig(args.out, "levels_per_day.png")
        r.add(f"- Дней в данных: {len(day)}, "
              f"медиана уровней в день: {day.median():.0f}")
        if "Churn" in df:
            ch = df.set_index("ServerTime")["Churn"].resample("W").mean() * 100
            plt.figure(figsize=(8, 3))
            ch.plot(color="#e45756")
            plt.title("% оттока по неделям (по строкам)")
            savefig(args.out, "churn_by_week.png")
            r.add("- Доля оттока по неделям сохранена в churn_by_week.png "
                  "(если тренд заметный, нужно разбиение по времени).")

    # ------------------------------------------------------------------ 9
    r.h("9. Выводы для моделирования (проверьте под свои цифры)")
    r.add("- Если доля оттока < 5%: сильный дисбаланс. Нужны веса классов, "
          "стратификация, метрики PR-AUC и recall/precision по классу 1.")
    r.add("- Данные на уровне уровней, а не игроков: нужна агрегация по "
          "UserID и скользящие окна только по прошлому (без утечки).")
    r.add("- Разбиение делить по игрокам и/или по времени, чтобы игрок не "
          "попадал одновременно в train и test.")
    r.add("- Столбцы с одним значением удалить; сильно скошенные признаки "
          "проверить с log-преобразованием (для линейных моделей).")

    r.save(os.path.join(args.out, "eda_report.md"))
    print(f"\nГотово. Отчёт и графики: {args.out}/")


if __name__ == "__main__":
    sys.exit(main())
