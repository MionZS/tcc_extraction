import marimo

__generated_with = "0.11.0"
app = marimo.App(width="wide", app_title="EDA — Dataset AMI Araucária (Copel)")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    from pathlib import Path

    from src.eda import dataset_eda as eda

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
    return Path, eda, mo, np, pl, plt


@app.cell
def _(Path, mo):
    _intro = mo.md(
        """
        # EDA completo do dataset AMI — Araucária / Copel

        Notebook de análise exploratória para apresentação à banca. Cobre: **volume e cobertura temporal**,
        **completude da memória de massa (0–100 %)**, **outliers de consumo por faixa de consumo médio**,
        **heatmap de atraso de comunicação**, **distribuição das features** e **ciclo de vida dos medidores**.
        """
    )
    ROOT = Path(__file__).resolve().parent.parent
    if not (ROOT / "output").exists():
        ROOT = Path.cwd()

    delay_dir_input = mo.ui.text(value=r"D:\Projects\delay_analysis\data", label="Pasta data/ do delay_analysis", full_width=True)
    delay_source = mo.ui.dropdown(
        options=["VOLTAGE", "CURRENT", "CURRENT_INSTANT", "UI_ANGLE", "F_ENERGY_D"], value="VOLTAGE", label="Grandeza ORCA"
    )
    min_cov = mo.ui.slider(0.5, 1.0, step=0.05, value=0.8, label="Cobertura mínima p/ dia confiável")
    n_bands = mo.ui.slider(3, 10, step=1, value=5, label="Nº de faixas de consumo médio")
    z_thr = mo.ui.slider(2.5, 6.0, step=0.5, value=3.5, label="Limiar z modificado (MAD)")
    tukey = mo.ui.slider(1.0, 3.0, step=0.5, value=1.5, label="k de Tukey (IQR)")
    agg_min = mo.ui.dropdown(options=[15, 30, 60, 120, 240], value=60, label="Agregação do heatmap (min)")
    mo.vstack(
        [
            _intro,
            mo.hstack([min_cov, n_bands, z_thr, tukey], justify="start", gap=2),
            mo.hstack([delay_dir_input, delay_source, agg_min], justify="start", gap=2),
        ]
    )
    return ROOT, agg_min, delay_dir_input, delay_source, min_cov, n_bands, tukey, z_thr


@app.cell
def _(ROOT, eda, mo, pl):
    # ── Carga ────────────────────────────────────────────────────────────────
    _files = eda.discover_daily_features(ROOT)
    daily = eda.load_daily_features(_files)

    _cls_parquet = ROOT / "output" / "reports" / "classified_ucs_latest.parquet"
    _train_parquet = ROOT / "output" / "model_input" / "v1" / "training_dataset.parquet"
    _src = _cls_parquet if _cls_parquet.exists() else _train_parquet
    windows = pl.read_parquet(_src) if _src.exists() else pl.DataFrame()

    _ctx = ROOT / "output" / "context"
    uc_ctx = pl.read_csv(_ctx / "uc_context.csv", separator=";", infer_schema_length=5000, ignore_errors=True) if (_ctx / "uc_context.csv").exists() else pl.DataFrame()
    meter_hist = pl.read_csv(_ctx / "meter_installation_history.csv", separator=";", infer_schema_length=5000, ignore_errors=True) if (_ctx / "meter_installation_history.csv").exists() else pl.DataFrame()

    mo.callout(
        mo.md(
            f"**Partições diárias:** {len(_files)} · **linhas UC×dia:** {daily.height:,} · "
            f"**janelas UC×cutoff:** {windows.height:,} · **UCs no cadastro:** {uc_ctx.height:,} · "
            f"**registros de medidores:** {meter_hist.height:,}"
        ),
        kind="success" if daily.height else "warn",
    )
    return daily, meter_hist, uc_ctx, windows


@app.cell
def _(daily, mo, pl):
    # ── 1. Visão geral ──────────────────────────────────────────────────────
    if daily.is_empty():
        _ov = mo.callout("Sem features diárias em `output/features/uc_day_features`. Execute o pipeline de extração.", kind="warn")
    else:
        _days = daily["REPORT_DAY"].n_unique()
        _ov = mo.hstack(
            [
                mo.stat(value=f"{daily['UC'].n_unique():,}", label="UCs com telemetria"),
                mo.stat(value=f"{_days}", label="Dias de telemetria"),
                mo.stat(value=f"{daily['REPORT_DAY'].min()} → {daily['REPORT_DAY'].max()}", label="Período"),
                mo.stat(value=f"{daily['FA_INTERVAL_SUM'].median():.2f} kWh", label="Mediana diária (UC)"),
                mo.stat(value=f"{daily['INTERVAL_COVERAGE'].mean():.1%}", label="Cobertura média diária"),
            ],
            justify="space-between",
        )
    mo.vstack([mo.md("## 1. Visão geral e cobertura temporal"), _ov])
    return


@app.cell
def _(daily, mo, pl, plt):
    # UCs ativas por dia e cobertura média por dia
    if daily.is_empty():
        _v = mo.md("")
    else:
        _per = (
            daily.group_by("REPORT_DAY")
            .agg(pl.col("UC").n_unique().alias("ucs"), pl.col("INTERVAL_COVERAGE").mean().alias("cov"))
            .sort("REPORT_DAY")
        )
        _fig, _ax = plt.subplots(figsize=(11, 3.8))
        _ax.plot(_per["REPORT_DAY"].to_list(), _per["ucs"].to_list(), color="#2563eb")
        _ax.set_ylabel("UCs com dados")
        _ax2 = _ax.twinx()
        _ax2.plot(_per["REPORT_DAY"].to_list(), (_per["cov"] * 100).to_list(), color="#dc2626", lw=1)
        _ax2.set_ylabel("Cobertura média (%)")
        _ax.set_title("UCs com telemetria e cobertura média por dia")
        _fig.tight_layout()
        _v = mo.as_html(_fig)
    _v
    return


@app.cell
def _(daily, eda, mo):
    # ── 2. Completude da memória de massa ───────────────────────────────────
    completeness = eda.meter_completeness(daily)
    comp_hist = eda.completeness_histogram(completeness, step_pct=10)
    comp_fig = eda.plot_completeness(comp_hist) if completeness.height else None

    if completeness.height:
        _c = completeness["completeness_calendar"]
        _stats = mo.hstack(
            [
                mo.stat(value=f"{(_c >= 0.95).mean():.1%}", label="Medidores ≥ 95 % completos"),
                mo.stat(value=f"{(_c >= 0.80).mean():.1%}", label="Medidores ≥ 80 %"),
                mo.stat(value=f"{(_c < 0.50).mean():.1%}", label="Medidores < 50 %"),
                mo.stat(value=f"{_c.median():.1%}", label="Mediana de completude"),
            ],
            justify="space-between",
        )
        _body = mo.vstack([_stats, mo.as_html(comp_fig), mo.ui.table(comp_hist.to_dicts(), page_size=12)])
    else:
        _body = mo.callout("Sem dados para completude.", kind="warn")
    mo.vstack([mo.md("## 2. Completude da memória de massa por medidor (0–100 %)\n\n"
                     "*Completude calendário:* soma da cobertura diária dividida pelos dias desde a 1ª leitura do medidor "
                     "(dias sem dado contam como 0). Medidor recém-instalado não é penalizado pelo passado."), _body])
    return comp_fig, comp_hist, completeness


@app.cell
def _(completeness, mo, plt):
    # Relação entre dias presentes e completude + CDF
    if completeness.is_empty():
        _v = mo.md("")
    else:
        _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
        _c = completeness["completeness_calendar"].sort().to_numpy() * 100
        _surv = [100 * (1 - i / max(1, len(_c) - 1)) for i in range(len(_c))]
        _axes[0].plot(_c, _surv, color="#2563eb")
        _axes[0].set_xlabel("Completude (%)")
        _axes[0].set_ylabel("% de medidores com completude ≥ x")
        _axes[0].set_title("Curva de sobrevivência da completude")
        _axes[0].grid(alpha=0.3)
        _axes[1].scatter(completeness["n_days_calendar"], completeness["completeness_calendar"] * 100, s=2, alpha=0.2)
        _axes[1].set_xlabel("Dias desde a 1ª leitura")
        _axes[1].set_ylabel("Completude (%)")
        _axes[1].set_title("Completude × tempo de observação")
        _fig.tight_layout()
        _v = mo.as_html(_fig)
    _v
    return


@app.cell
def _(daily, eda, min_cov, mo, n_bands):
    # ── 3. Faixas de consumo e outliers ─────────────────────────────────────
    bands = eda.consumption_bands(daily, n_bands=n_bands.value, min_coverage=min_cov.value) if not daily.is_empty() else None
    if bands is not None and bands.height:
        _t = bands.group_by("band", "band_idx").len().sort("band_idx").rename({"len": "medidores"})
        _view = mo.vstack([mo.ui.table(_t.to_dicts(), page_size=10)])
    else:
        _view = mo.callout("Sem dados de consumo.", kind="warn")
    mo.vstack([mo.md("## 3. Outliers de consumo em medidores de mesma faixa média\n\n"
                     "Os medidores são agrupados em **faixas de consumo médio** (quantis da mediana diária, considerando apenas dias confiáveis). "
                     "Dentro de cada faixa contamos medições diárias fora do padrão sob duas definições: "
                     "**(a)** fora das cercas de Tukey dos *pares da faixa* (escala log1p) e **(b)** z modificado (mediana/MAD) vs o *histórico do próprio medidor*."),
               _view])
    return (bands,)


@app.cell
def _(bands, daily, eda, min_cov, mo, pl, tukey, z_thr):
    if bands is None or bands.is_empty():
        outliers = None
        outlier_summary = None
        outlier_fig = None
        _body = mo.callout("Sem outliers calculáveis.", kind="warn")
    else:
        outliers = eda.consumption_outliers(daily, bands, min_coverage=min_cov.value, z_threshold=z_thr.value, tukey_k=tukey.value)
        outlier_summary = eda.outlier_summary_by_band(outliers)
        outlier_fig = eda.plot_outlier_bands(outlier_summary)
        _m = outliers["UC"].n_unique()
        _meter_any = outliers.group_by("UC").agg(pl.col("out_band").any().alias("x"))["x"].mean()
        _body = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(value=f"{outliers['out_band'].mean():.2%}", label="Medições fora do padrão da faixa"),
                        mo.stat(value=f"{outliers['out_self'].mean():.2%}", label="Medições fora do padrão próprio"),
                        mo.stat(value=f"{_meter_any:.1%}", label="Medidores c/ ≥ 1 outlier de faixa"),
                        mo.stat(value=f"{_m:,}", label="Medidores analisados"),
                    ],
                    justify="space-between",
                ),
                mo.as_html(outlier_fig),
                mo.ui.table(outlier_summary.to_dicts(), page_size=12),
            ]
        )
    _body
    return outlier_fig, outlier_summary, outliers


@app.cell
def _(eda, mo, outliers, pl, plt):
    # Distribuição de outliers por medidor
    if outliers is None:
        _v = mo.md("")
    else:
        _dist = eda.outlier_meter_distribution(outliers)
        _fig, _ax = plt.subplots(figsize=(10, 4))
        _ax.hist(_dist["out_self_rate"].to_numpy() * 100, bins=40, alpha=0.7, label="vs histórico próprio", color="#f59e0b")
        _ax.hist(_dist["out_band_rate"].to_numpy() * 100, bins=40, alpha=0.6, label="vs pares da faixa", color="#dc2626")
        _ax.set_yscale("log")
        _ax.set_xlabel("% dos dias do medidor classificados como outlier")
        _ax.set_ylabel("Nº de medidores (log)")
        _ax.legend()
        _ax.set_title("Quantos dias fora do padrão cada medidor possui")
        _fig.tight_layout()
        _v = mo.as_html(_fig)
    _v
    return


@app.cell
def _(eda, mo, np, outliers, pl, plt):
    # Boxplot de consumo por faixa (log1p) — evidencia dispersão intra-faixa
    if outliers is None:
        _v = mo.md("")
    else:
        _groups = outliers.sort("band_idx").partition_by("band", maintain_order=True)
        _fig, _ax = plt.subplots(figsize=(11, 4.5))
        _ax.boxplot([np.log1p(g["FA_INTERVAL_SUM"].to_numpy()) for g in _groups], labels=[g["band"][0].split(":")[0] for g in _groups], showfliers=True, flierprops=dict(markersize=1, alpha=0.3))
        _ax.set_ylabel("log1p(kWh/dia)")
        _ax.set_title("Consumo diário por faixa de consumo médio (dias confiáveis)")
        _fig.tight_layout()
        _v = mo.as_html(_fig)
    _v
    return


@app.cell
def _(agg_min, delay_dir_input, delay_source, eda, mo):
    # ── 4. Heatmap de atraso de comunicação ─────────────────────────────────
    _files = eda.discover_delay_histograms(delay_dir_input.value, delay_source.value)
    delay_hist = eda.load_delay_histograms(_files)
    if delay_hist.is_empty():
        delay_fig = None
        arrival_fig = None
        arrival = None
        delay_daily = None
        _body = mo.callout(
            mo.md(f"Nenhum histograma real encontrado em `{delay_dir_input.value}\\{delay_source.value}`. "
                  "Extraia com `uv run python extrair_orca.py --start AAAA-MM-DD --sources "
                  f"{delay_source.value}` no projeto `delay_analysis`."),
            kind="warn",
        )
    else:
        _z, _w, _b = eda.delay_heatmap_matrix(delay_hist, agg_minutes=agg_min.value)
        delay_fig = eda.plot_delay_heatmap(
            _z, _w, _b,
            title=f"Heatmap de atraso de chegada — {delay_source.value} ({delay_hist['dt_referencia'].min()} → {delay_hist['dt_referencia'].max()})",
        )
        arrival = eda.delay_arrival_table(delay_hist)
        arrival_fig = eda.plot_delay_arrival(arrival)
        delay_daily = eda.daily_delay_summary(delay_hist)
        _body = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(value=f"{delay_hist['dt_referencia'].n_unique()}", label="Dias de histograma"),
                        mo.stat(value=f"{delay_daily['share_le_1h'].mean():.1%}", label="Chegam em ≤ 1 h"),
                        mo.stat(value=f"{delay_daily['share_le_4h'].mean():.1%}", label="Chegam em ≤ 4 h"),
                        mo.stat(value=f"{delay_daily['share_le_24h'].mean():.1%}", label="Chegam em ≤ 24 h"),
                        mo.stat(value=f"{delay_daily['mean_delay_min'].mean():.1f} min", label="Atraso médio"),
                    ],
                    justify="space-between",
                ),
                mo.as_html(delay_fig),
                mo.as_html(arrival_fig),
            ]
        )
    mo.vstack([
        mo.md("## 4. Comunicação: quanto tempo as leituras levam para chegar\n\n"
              "Construído a partir dos histogramas ORCA do projeto `delay_analysis` (atraso = gravação − medição, em minutos). "
              "O eixo X é a janela do dia em que a leitura foi medida; o eixo Y é a faixa de atraso; a cor é a densidade de leituras. "
              "> ⚠️ Os histogramas do ORCA cobrem a rede inteira (retenção de 3 meses); servem como retrato do canal de comunicação da Copel, "
              "do qual Araucária (RF Mesh) é uma amostra."),
        _body,
    ])
    return arrival, arrival_fig, delay_daily, delay_fig, delay_hist


@app.cell
def _(delay_daily, mo, plt):
    if delay_daily is None or delay_daily.is_empty():
        _v = mo.md("")
    else:
        _fig, _ax = plt.subplots(figsize=(11, 3.8))
        _x = delay_daily["dt_referencia"].to_list()
        for _c, _l in [("share_le_1h", "≤ 1 h"), ("share_le_4h", "≤ 4 h"), ("share_le_24h", "≤ 24 h")]:
            _ax.plot(_x, (delay_daily[_c] * 100).to_list(), label=_l)
        _ax.set_ylabel("% das leituras")
        _ax.set_title("Evolução diária do prazo de chegada das leituras")
        _ax.legend()
        _ax.tick_params(axis="x", rotation=60, labelsize=6)
        _fig.tight_layout()
        _v = mo.as_html(_fig)
    _v
    return


@app.cell
def _(mo, np, pl, plt, windows):
    # ── 5. Distribuição das features de janela ──────────────────────────────
    _title = mo.md("## 5. Distribuição das características por janela (UC × cutoff)")
    if windows.is_empty():
        _out = mo.callout("Sem tabela de janelas.", kind="warn")
    else:
        _feats = [c for c in [
            "x__hist_fa_sum_median", "x__hist_fa_sum_mad", "x__fa_cv_window", "x__hist_load_factor_mean",
            "x__hist_voltage_imbalance_max", "x__hist_current_imbalance_max", "x__trend_fa_slope",
            "x__avg_interval_coverage", "x__hist_ra_reversal_ratio_max", "x__meter_age_days",
        ] if c in windows.columns]
        _fig, _axes = plt.subplots(2, 5, figsize=(18, 6))
        for _ax, _c in zip(_axes.ravel(), _feats):
            _v = windows[_c].drop_nulls().to_numpy().astype(float)
            _v = _v[np.isfinite(_v)]
            if len(_v):
                _lo, _hi = np.percentile(_v, [0.5, 99.5])
                _ax.hist(np.clip(_v, _lo, _hi), bins=50, color="#2563eb")
            _ax.set_title(_c.replace("x__", ""), fontsize=8)
        _fig.suptitle("Histogramas (cortados nos percentis 0,5–99,5)")
        _fig.tight_layout()
        _out = mo.as_html(_fig)
    mo.vstack([_title, _out])
    return


@app.cell
def _(mo, np, pl, plt, windows):
    # Estatísticas descritivas + correlação
    if windows.is_empty():
        _v = mo.md("")
    else:
        _cols = [c for c in windows.columns if c.startswith("x__") and windows[c].dtype.is_numeric()]
        _desc = windows.select(_cols).describe()
        _corr_cols = _cols[:20]
        _m = windows.select(_corr_cols).drop_nulls().to_numpy().astype(float)
        _m = _m[np.isfinite(_m).all(axis=1)][:100000]
        _corr = np.corrcoef(_m, rowvar=False) if len(_m) > 2 else np.zeros((len(_corr_cols),) * 2)
        _fig, _ax = plt.subplots(figsize=(9, 8))
        _im = _ax.imshow(_corr, cmap="coolwarm", vmin=-1, vmax=1)
        _ax.set_xticks(range(len(_corr_cols)), [c.replace("x__", "") for c in _corr_cols], rotation=90, fontsize=6)
        _ax.set_yticks(range(len(_corr_cols)), [c.replace("x__", "") for c in _corr_cols], fontsize=6)
        _fig.colorbar(_im, ax=_ax)
        _ax.set_title("Correlação de Pearson entre features")
        _fig.tight_layout()
        _v = mo.vstack([mo.ui.table(_desc.to_dicts(), page_size=12), mo.as_html(_fig)])
    _v
    return


@app.cell
def _(mo, pl, plt, windows):
    # ── 6. Qualidade de telecom e categorias ────────────────────────────────
    _title = mo.md("## 6. Qualidade de comunicação, quarentena e categorias diagnósticas")
    if windows.is_empty():
        _out = mo.callout("Sem janelas.", kind="warn")
    else:
        _figs = []
        _fig, _axes = plt.subplots(1, 2, figsize=(14, 4.5))
        if "meta__telecom_status" in windows.columns:
            _t = windows.group_by("meta__telecom_status").len().sort("len", descending=True)
            _axes[0].barh(_t["meta__telecom_status"].to_list()[::-1], _t["len"].to_list()[::-1], color="#2563eb")
            _axes[0].set_title("Status de telecom das janelas")
        if "CATEGORIA_DIAGNOSTICA" in windows.columns:
            _c = windows.group_by("CATEGORIA_DIAGNOSTICA").len().sort("len", descending=True)
            _axes[1].barh(_c["CATEGORIA_DIAGNOSTICA"].to_list()[::-1], _c["len"].to_list()[::-1], color="#7c3aed")
            _axes[1].set_xscale("log")
            _axes[1].set_title("Categorias diagnósticas (escala log)")
        _fig.tight_layout()
        _out = mo.as_html(_fig)
    mo.vstack([_title, _out])
    return


@app.cell
def _(mo, pl, plt, windows):
    # Cobertura por cutoff e por alimentador
    if windows.is_empty() or "meta__cutoff_date" not in windows.columns:
        _v = mo.md("")
    else:
        _by_cut = windows.group_by("meta__cutoff_date").agg(
            pl.len().alias("n"),
            pl.col("x__avg_interval_coverage").mean().alias("cov"),
            (pl.col("meta__is_clean_for_train").mean() if "meta__is_clean_for_train" in windows.columns else pl.lit(None)).alias("clean"),
        ).sort("meta__cutoff_date")
        _fig, _ax = plt.subplots(figsize=(8, 3.8))
        _ax.bar([str(x) for x in _by_cut["meta__cutoff_date"].to_list()], (_by_cut["clean"] * 100).to_list(), color="#10b981")
        _ax.set_ylabel("% janelas íntegras p/ treino")
        _ax.set_title("Janelas íntegras por cutoff")
        _fig.tight_layout()
        _blocks = [mo.as_html(_fig)]
        if "meta__feeder" in windows.columns:
            _f = windows.group_by("meta__feeder").agg(
                pl.col("id__uc_id").n_unique().alias("ucs"),
                pl.col("x__avg_interval_coverage").mean().alias("cobertura_media"),
            ).sort("ucs", descending=True)
            _blocks.append(mo.md("**UCs e cobertura média por alimentador**"))
            _blocks.append(mo.ui.table(_f.to_dicts(), page_size=15))
        _v = mo.vstack(_blocks)
    _v
    return


@app.cell
def _(meter_hist, mo, pl, plt, uc_ctx):
    # ── 7. Ciclo de vida dos medidores e cadastro ───────────────────────────
    _title = mo.md("## 7. Parque de medidores e cadastro")
    if meter_hist.is_empty():
        _out = mo.callout("Sem histórico de instalação.", kind="warn")
    else:
        _h = meter_hist.with_columns(pl.col("DATA_INSTALACAO").cast(pl.Date, strict=False))
        _m = _h.group_by(pl.col("DATA_INSTALACAO").dt.truncate("1mo").alias("mes")).len().sort("mes")
        _fig, _axes = plt.subplots(1, 3, figsize=(17, 4))
        _axes[0].bar(_m["mes"].to_list(), _m["len"].to_list(), width=25, color="#2563eb")
        _axes[0].set_title(f"Instalações por mês (mín. {_h['DATA_INSTALACAO'].min()})")
        _axes[0].tick_params(axis="x", rotation=60, labelsize=6)
        _per_uc = _h.group_by("UC").agg(pl.len().alias("n_med"))
        _vc = _per_uc.group_by("n_med").agg(pl.len().alias("ucs")).sort("n_med")
        _axes[1].bar([str(x) for x in _vc["n_med"].to_list()], _vc["ucs"].to_list(), color="#f59e0b")
        _axes[1].set_yscale("log")
        _axes[1].set_title("Medidores históricos por UC")
        if "FAMILIA_SMART" in _h.columns and "FASE" in _h.columns:
            _fa = _h.group_by("FASE").len().sort("len", descending=True)
            _axes[2].bar(_fa["FASE"].fill_null("?").to_list(), _fa["len"].to_list(), color="#7c3aed")
            _axes[2].set_title("Fase / tipo de medidor")
        _fig.tight_layout()
        _blocks = [mo.as_html(_fig)]
        if not uc_ctx.is_empty() and "CLASSE" in uc_ctx.columns:
            _cl = uc_ctx.group_by("CLASSE").len().sort("len", descending=True).head(15)
            _blocks.append(mo.md("**Classes de consumo (top 15)**"))
            _blocks.append(mo.ui.table(_cl.to_dicts(), page_size=15))
        _out = mo.vstack(_blocks)
    mo.vstack([_title, _out])
    return


@app.cell
def _(mo):
    export_btn = mo.ui.run_button(label="💾 Exportar figuras para docs/figures/eda")
    return (export_btn,)


@app.cell
def _(ROOT, arrival_fig, comp_fig, delay_fig, export_btn, mo, outlier_fig):
    # ── 8. Exportação de figuras para o artigo ──────────────────────────────
    _msg = mo.md("")
    if export_btn.value:
        _out_dir = ROOT / "docs" / "figures" / "eda"
        _out_dir.mkdir(parents=True, exist_ok=True)
        _saved = []
        for _name, _f in [("completude_memoria_massa", comp_fig), ("outliers_por_faixa", outlier_fig),
                          ("heatmap_atraso", delay_fig), ("chegada_acumulada", arrival_fig)]:
            if _f is not None:
                _f.savefig(_out_dir / f"{_name}.png", dpi=200, bbox_inches="tight")
                _saved.append(_name)
        _msg = mo.callout(f"Salvas em `{_out_dir}`: {', '.join(_saved) or 'nenhuma'}", kind="success")
    mo.vstack([mo.md("## 8. Exportar figuras"), export_btn, _msg])
    return


if __name__ == "__main__":
    app.run()
