"""Gera as figuras e os capítulos do artigo sobre o dataset (EDA) com números calculados dos dados reais.

Uso:
    uv run python scripts/generate_eda_report.py
    uv run python scripts/generate_eda_report.py --delay-dir D:\\Projects\\delay_analysis\\data --source VOLTAGE

Saídas:
    docs/figures/eda/*.png
    docs/artigo_capitulos_dataset.md   (capítulos prontos, com os números preenchidos)
    output/reports/eda_metrics.json    (todas as métricas citadas no texto)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl

from src.eda import dataset_eda as eda


def _pct(x: float | None, nd: int = 1) -> str:
    return "n/d" if x is None else f"{x * 100:.{nd}f}%"


def _n(x: int | float | None) -> str:
    return "n/d" if x is None else f"{int(x):,}".replace(",", ".")


def compute(args) -> tuple[dict, dict]:
    fig_dir = ROOT_DIR / "docs" / "figures" / "eda"
    fig_dir.mkdir(parents=True, exist_ok=True)
    m: dict = {}
    figs: dict[str, str] = {}

    # ── dataset base ─────────────────────────────────────────────────────────
    files = eda.discover_daily_features(ROOT_DIR)
    daily = eda.load_daily_features(files)
    cls_p = ROOT_DIR / "output" / "reports" / "classified_ucs_latest.parquet"
    tr_p = ROOT_DIR / "output" / "model_input" / "v1" / "training_dataset.parquet"
    src = cls_p if cls_p.exists() else tr_p
    windows = pl.read_parquet(src) if src.exists() else pl.DataFrame()
    ctx = ROOT_DIR / "output" / "context"
    meter_hist = (
        pl.read_csv(ctx / "meter_installation_history.csv", separator=";", infer_schema_length=5000, ignore_errors=True)
        if (ctx / "meter_installation_history.csv").exists() else pl.DataFrame()
    )

    m["partitions"] = len(files)
    if not daily.is_empty():
        m.update(
            ucs=daily["UC"].n_unique(), days=daily["REPORT_DAY"].n_unique(),
            first_day=str(daily["REPORT_DAY"].min()), last_day=str(daily["REPORT_DAY"].max()),
            uc_day_rows=daily.height, mean_daily_coverage=float(daily["INTERVAL_COVERAGE"].mean()),
            median_daily_kwh=float(daily["FA_INTERVAL_SUM"].median()),
        )
    if not windows.is_empty():
        m["windows"] = windows.height
        if "meta__cutoff_date" in windows.columns:
            m["cutoffs"] = sorted(str(x) for x in windows["meta__cutoff_date"].unique().to_list())
        if "meta__is_clean_for_train" in windows.columns:
            m["clean_share"] = float(windows["meta__is_clean_for_train"].mean())
        if "meta__telecom_status" in windows.columns:
            m["telecom_status"] = {r[0]: r[1] for r in windows.group_by("meta__telecom_status").len().iter_rows()}
    if not meter_hist.is_empty():
        h = meter_hist.with_columns(pl.col("DATA_INSTALACAO").cast(pl.Date, strict=False))
        m["meter_records"] = h.height
        m["earliest_install"] = str(h["DATA_INSTALACAO"].min())
        smart = h.filter(pl.col("DATA_INSTALACAO") >= pl.date(2023, 1, 1))
        m["earliest_smart_install"] = str(smart["DATA_INSTALACAO"].min()) if smart.height else None

    # ── completude ───────────────────────────────────────────────────────────
    if not daily.is_empty():
        comp = eda.meter_completeness(daily)
        hist = eda.completeness_histogram(comp, step_pct=10)
        c = comp["completeness_calendar"]
        m["completeness"] = {
            "meters": comp.height,
            "ge95": float((c >= 0.95).mean()), "ge80": float((c >= 0.80).mean()),
            "lt50": float((c < 0.50).mean()), "median": float(c.median()),
            "histogram": hist.select(["bin", "meters", "share"]).to_dicts(),
        }
        f = eda.plot_completeness(hist)
        f.savefig(fig_dir / "completude_memoria_massa.png", dpi=200, bbox_inches="tight")
        figs["completude"] = "figures/eda/completude_memoria_massa.png"

        # ── outliers ─────────────────────────────────────────────────────────
        bands = eda.consumption_bands(daily, n_bands=args.bands, min_coverage=args.min_coverage)
        out = eda.consumption_outliers(daily, bands, min_coverage=args.min_coverage,
                                       z_threshold=args.z, tukey_k=args.tukey)
        if not out.is_empty():
            summ = eda.outlier_summary_by_band(out)
            any_band = out.group_by("UC").agg(pl.col("out_band").any().alias("x"))["x"].mean()
            any_self = out.group_by("UC").agg(pl.col("out_self").any().alias("x"))["x"].mean()
            m["outliers"] = {
                "params": {"bands": args.bands, "min_coverage": args.min_coverage, "z": args.z, "tukey": args.tukey},
                "meters": out["UC"].n_unique(), "measurements": out.height,
                "rate_band": float(out["out_band"].mean()), "rate_self": float(out["out_self"].mean()),
                "meters_any_band": float(any_band), "meters_any_self": float(any_self),
                "by_band": summ.to_dicts(),
            }
            f = eda.plot_outlier_bands(summ)
            f.savefig(fig_dir / "outliers_por_faixa.png", dpi=200, bbox_inches="tight")
            figs["outliers"] = "figures/eda/outliers_por_faixa.png"

    # ── atraso de comunicação ────────────────────────────────────────────────
    dfiles = eda.discover_delay_histograms(args.delay_dir, args.source)
    dh = eda.load_delay_histograms(dfiles)
    if not dh.is_empty():
        z, w, b = eda.delay_heatmap_matrix(dh, agg_minutes=args.agg)
        eda.plot_delay_heatmap(z, w, b, title=f"Atraso de chegada das leituras — {args.source}").savefig(
            fig_dir / "heatmap_atraso.png", dpi=200, bbox_inches="tight")
        arr = eda.delay_arrival_table(dh)
        eda.plot_delay_arrival(arr).savefig(fig_dir / "chegada_acumulada.png", dpi=200, bbox_inches="tight")
        dd = eda.daily_delay_summary(dh)
        m["delay"] = {
            "source": args.source, "days": dh["dt_referencia"].n_unique(),
            "first_day": str(dh["dt_referencia"].min()), "last_day": str(dh["dt_referencia"].max()),
            "readings": int(dd["readings"].sum()), "mean_delay_min": float(dd["mean_delay_min"].mean()),
            "le_1h": float(dd["share_le_1h"].mean()), "le_4h": float(dd["share_le_4h"].mean()),
            "le_24h": float(dd["share_le_24h"].mean()),
        }
        figs["heatmap"] = "figures/eda/heatmap_atraso.png"
        figs["chegada"] = "figures/eda/chegada_acumulada.png"
    return m, figs


def chapter(m: dict, figs: dict) -> str:
    L: list[str] = []
    a = L.append
    a("# Capítulos do artigo — Caracterização do Dataset AMI de Araucária\n")
    a("> Gerado por `scripts/generate_eda_report.py`. Todos os números abaixo foram calculados sobre os dados reais "
      "(métricas completas em `output/reports/eda_metrics.json`). Onde constar *n/d*, o insumo não estava disponível na execução.\n")

    # Cap. 3
    a("## 3. Descrição do Conjunto de Dados\n")
    a("### 3.1 Fontes e granularidade\n")
    a("O conjunto é composto por quatro camadas: (i) **contexto** cadastral (CIS), topológico (GEO) e histórico de instalação "
      "de medidores; (ii) **medições** de memória de massa a cada 5 minutos (288 pontos/dia) — energia ativa/reativa, "
      "tensão e corrente por fase, grandezas instantâneas e registradores; (iii) **eventos** de alarme; e (iv) **features** "
      "derivadas por UC×dia e por janela (UC×*cutoff*). A unidade analítica final é a janela de 30 dias terminada em um *cutoff*.\n")
    if "ucs" in m:
        a(f"Na extração corrente há **{_n(m['ucs'])} UCs** com telemetria em **{m['days']} dias** "
          f"({m['first_day']} a {m['last_day']}), totalizando **{_n(m['uc_day_rows'])} linhas UC×dia**, com cobertura diária média de "
          f"**{_pct(m['mean_daily_coverage'])}** e consumo diário mediano de **{m['median_daily_kwh']:.2f} kWh** por UC.\n")
    if "windows" in m:
        cut = ", ".join(m.get("cutoffs", []))
        a(f"Foram construídas **{_n(m['windows'])} janelas** UC×*cutoff* (cutoffs: {cut}); "
          f"**{_pct(m.get('clean_share'))}** são íntegras para treino (telemetria representativa).\n")
    if "earliest_install" in m:
        a(f"O histórico de instalação reúne **{_n(m['meter_records'])} registros** de medidores. A instalação mais antiga é de "
          f"**{m['earliest_install']}** (medidor eletromecânico legado) e o *rollout* de medidores inteligentes inicia em "
          f"**{m.get('earliest_smart_install')}**.\n")
    if "telecom_status" in m:
        a("### 3.2 Estados de comunicação das janelas\n")
        a("| Estado | Janelas |\n|---|---:|")
        for k, v in sorted(m["telecom_status"].items(), key=lambda kv: -kv[1]):
            a(f"| {k} | {_n(v)} |")
        a("")

    # Cap. 4
    a("## 4. Completude da Memória de Massa\n")
    a("A completude de um medidor é a soma da cobertura diária (pontos válidos ÷ 288) dividida pelo número de dias de calendário "
      "desde sua primeira leitura; dias sem qualquer registro contam como zero. Assim, um medidor recém-instalado não é "
      "penalizado pelo período anterior à sua instalação.\n")
    if "completeness" in m:
        c = m["completeness"]
        a(f"Dos **{_n(c['meters'])} medidores**, **{_pct(c['ge95'])}** possuem completude ≥ 95 %, **{_pct(c['ge80'])}** ≥ 80 % e "
          f"**{_pct(c['lt50'])}** < 50 %; a mediana é **{_pct(c['median'])}**.\n")
        if "completude" in figs:
            a(f"![Distribuição de completude da memória de massa por medidor]({figs['completude']})\n")
            a("*Figura 1 — Número de medidores por faixa de completude (0–100 %).*\n")
        a("| Faixa | Medidores | Fração |\n|---|---:|---:|")
        for r in c["histogram"]:
            a(f"| {r['bin']} | {_n(r['meters'])} | {_pct(r['share'])} |")
        a("")
        a("A concentração de medidores nas faixas extremas (quando presente) indica populações distintas de qualidade de comunicação, o que "
          "motiva separar a **qualidade da telemetria** da **análise do perfil elétrico** (quarentena por completude).\n")

    # Cap. 5
    a("## 5. Latência de Comunicação\n")
    a("A latência de uma leitura é a diferença entre o instante de gravação no ORCA e o instante da medição. Os histogramas diários "
      "(faixas de atraso × janelas de 5 minutos) permitem estimar quanto tempo as leituras levam para chegar.\n")
    if "delay" in m:
        d = m["delay"]
        a(f"Sobre **{d['days']} dias** ({d['first_day']} a {d['last_day']}) e **{_n(d['readings'])} leituras** da grandeza `{d['source']}`, "
          f"**{_pct(d['le_1h'])}** chegam em até 1 h, **{_pct(d['le_4h'])}** em até 4 h e **{_pct(d['le_24h'])}** em até 24 h; o atraso médio "
          f"é de **{d['mean_delay_min']:.1f} min**.\n")
        if "heatmap" in figs:
            a(f"![Heatmap de densidade de atraso por hora do dia]({figs['heatmap']})\n")
            a("*Figura 2 — Densidade de leituras por faixa de atraso (eixo Y) e hora de medição (eixo X), escala logarítmica.*\n")
        if "chegada" in figs:
            a(f"![Chegada acumulada das leituras]({figs['chegada']})\n")
            a("*Figura 3 — Fração de leituras por faixa de atraso e curva acumulada.*\n")
        a("> **Nota de escopo.** Os histogramas do ORCA agregam a rede inteira (retenção de 3 meses); representam o canal de comunicação "
          "da concessionária, do qual Araucária é uma amostra. A completude por medidor (Cap. 4) é, por sua vez, específica de Araucária.\n")
    else:
        a("*Histogramas de atraso não encontrados (n/d). Extraia-os no projeto `delay_analysis` e reexecute o script.*\n")

    # Cap. 6
    a("## 6. Outliers de Consumo por Faixa de Consumo Médio\n")
    if "outliers" in m:
        o = m["outliers"]; p = o["params"]
        a(f"Os medidores foram agrupados em **{p['bands']} faixas** de consumo médio (quantis da mediana diária em dias com cobertura ≥ "
          f"{p['min_coverage']:.0%}). Dentro de cada faixa, uma medição diária é *outlier* sob duas definições: **(a)** fora das cercas de Tukey "
          f"(k = {p['tukey']}) dos pares da faixa em escala log1p; **(b)** |z modificado| > {p['z']} em relação à mediana/MAD do próprio medidor.\n")
        a(f"Em **{_n(o['meters'])} medidores** e **{_n(o['measurements'])} medições**, **{_pct(o['rate_band'], 2)}** destoam dos pares da faixa e "
          f"**{_pct(o['rate_self'], 2)}** do próprio histórico; **{_pct(o['meters_any_band'])}** dos medidores têm ao menos uma medição fora do padrão "
          f"da faixa e **{_pct(o['meters_any_self'])}** fora do próprio padrão.\n")
        if "outliers" in figs:
            a(f"![Outliers de consumo por faixa]({figs['outliers']})\n")
            a("*Figura 4 — Taxa de medições outlier e fração de medidores afetados por faixa de consumo médio.*\n")
        a("| Faixa | Medidores | Medições | Outlier faixa | Outlier próprio | Medidores c/ ≥1 (faixa) |\n|---|---:|---:|---:|---:|---:|")
        for r in o["by_band"]:
            a(f"| {r['band']} | {_n(r['meters'])} | {_n(r['measurements'])} | {_pct(r['out_band_rate'], 2)} | "
              f"{_pct(r['out_self_rate'], 2)} | {_pct(r['meters_with_out_band_share'])} |")
        a("")
        a("Outliers não são, por si, irregularidades: a literatura de detecção de perdas não técnicas os trata como *candidatos* a inspeção, "
          "e esta etapa serve para dimensionar a cauda da distribuição e calibrar o detector não supervisionado.\n")

    # Cap. 7
    a("## 7. Ameaças à Validade dos Dados\n")
    a("1. **Janela temporal curta** (≈ 3 meses): sazonalidade anual não é observável; usa-se validação temporal *out-of-time*.\n"
      "2. **Ausência de rótulos confiáveis** de irregularidade; as métricas descritivas não medem acurácia de fraude.\n"
      "3. **Viés de comunicação**: medidores com baixa completude podem ter perfil elétrico subestimado; mitigado por quarentena e pela "
      "comparação com vizinhos topológicos.\n"
      "4. **Histogramas de atraso agregados**: não distinguem Araucária do restante da rede.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delay-dir", default=r"D:\Projects\delay_analysis\data")
    ap.add_argument("--source", default="VOLTAGE")
    ap.add_argument("--agg", type=int, default=60, help="Agregação do heatmap (min)")
    ap.add_argument("--bands", type=int, default=5)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--z", type=float, default=3.5)
    ap.add_argument("--tukey", type=float, default=1.5)
    args = ap.parse_args()

    metrics, figs = compute(args)
    rep = ROOT_DIR / "output" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "eda_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    out = ROOT_DIR / "docs" / "artigo_capitulos_dataset.md"
    out.write_text(chapter(metrics, figs), encoding="utf-8")
    print(f"✔ Capítulos: {out}\n✔ Métricas: {rep / 'eda_metrics.json'}\n✔ Figuras: {ROOT_DIR / 'docs' / 'figures' / 'eda'}")


if __name__ == "__main__":
    main()
