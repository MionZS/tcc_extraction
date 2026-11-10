"""Operational Isolation Rules, Spatial Telecom Contrast, Behavioral Class Transitions & Per-Category Reporting.

Implements clear, non-overlapping rules and ML/Statistical pipelines for:
1. New UC Profile Initiation (`INICIACAO_PERFIL_UC_NOVA` — 1st meter ever installed, <= 30 days)
2. Meter Replacement Transition (`TRANSICAO_TROCA_MEDIDOR` — >= 2nd meter on existing UC, <= 30 days)
3. Isolated Communication Anomaly vs. Healthy Neighbors (`ANOMALIA_COMUNICACAO_ISOLADA` — spatial/transformer contrast)
4. Collective RF Mesh Bottleneck (`GARGALO_COLETIVO_RF_MESH` — transformer/neighborhood-wide packet loss)
5. Temporal Behavioral Class Transitions (`MUDANCA_CLASSE_COMPORTAMENTAL` — UCs that changed consumption class across cutoffs)
6. Pure Electrical Anomaly / Non-Technical Loss (`ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT` — PyOD 6-detector ensemble on V, I, kWh)
7. Unregistered Distributed Generation (`POSSIVEL_GD_NAO_CADASTRADA` — Reverse active energy flow)
8. Stuck Meter / Metrological Fault (`FALHA_METROLOGICA_MEDIDOR_TRAVADO` — Energized line voltage with zero consumption)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass(frozen=True)
class CategoryMethodologySpec:
    """Explicit specification of each category, its report file, and whether ML or Rules apply."""

    code: str
    report_slug: str
    title: str
    paradigm: str
    uses_ml: str
    justification: str
    recommended_action: str


CATEGORY_METHODOLOGY_SPECS: tuple[CategoryMethodologySpec, ...] = (
    CategoryMethodologySpec(
        code="INICIACAO_PERFIL_UC_NOVA",
        report_slug="relatorio_01_iniciacao_perfil_ucs_novas",
        title="1. Iniciação de Perfil (UC Nova / 1ª Instalação de Medidor)",
        paradigm="Regra Determinística Cadastral + Classificação Preliminar de Porte",
        uses_ml="NÃO para isolar (Regra Determinística: N_METERS_HISTORICAL == 1 e Idade <= 30d); SIM (Clusterização) após >= 7 dias confiáveis para atribuir classe inicial.",
        justification=(
            "Saber se uma UC é recém-ligada (1º medidor da história) é um fato cadastral determinístico "
            "(DATA_INSTALACAO e contagem de NIOs = 1). Usar ML de anomalia aqui geraria falsos positivos "
            "porque a janela de 30 dias está incompleta (rampa de entrada). O papel analítico correto é "
            "isolar a UC em 'Iniciação de Perfil' e observar sua classe comportamental preliminar."
        ),
        recommended_action="Aguardar maturação de 30 dias para liberar ao detector de fraude; validar se a classe preliminar condiz com a tarifa cadastrada.",
    ),
    CategoryMethodologySpec(
        code="TRANSICAO_TROCA_MEDIDOR",
        report_slug="relatorio_02_transicao_troca_medidor",
        title="2. Transição por Troca de Medidor (UC Existente com Substituição de Hardware)",
        paradigm="Regra de Ciclo de Vida (N_METERS >= 2) + Comparação Longitudinal Antes/Depois",
        uses_ml="HÍBRIDO: Regra Determinística para detectar a troca + Teste de Salto de Patamar (Antes vs. Depois) para inferir submedição prévia.",
        justification=(
            "A substituição de um medidor em UC já existente mistura dias do medidor antigo (muitas vezes "
            "defeituoso ou fraudado) com o medidor novo. Isolar a janela de transição evita confundir a "
            "troca física com uma anomalia espontânea, e comparar o consumo pós-troca contra o pré-troca "
            "revela se o medidor antigo estava submedindo."
        ),
        recommended_action="Auditar UCs com salto positivo expressivo pós-troca (> +50% kWh/dia) para recuperação de receita retroativa.",
    ),
    CategoryMethodologySpec(
        code="ANOMALIA_COMUNICACAO_ISOLADA",
        report_slug="relatorio_03_comunicacao_isolada_vizinhanca_boa",
        title="3A. Anomalia de Comunicação Isolada (Vizinhos no Mesmo Trafo/Rua Comunicam Bem)",
        paradigm="Estatística Espacial/Topológica de Vizinhança (Posto Operacional + Coordenadas LAT/LON)",
        uses_ml="NÃO precisa de caixa-preta: Contraste Espacial Robusto (Cobertura UC < 60% vs. Mediana do Transformador/Vizinhos >= 80%, Déficit >= 30%).",
        justification=(
            "A queda de pacotes RF Mesh só é anomalia individual do medidor quando provamos que os medidores "
            "vizinhos (mesmo Posto Operacional/Transformador ou raio geográfico de ~250m) comunicam normalmente "
            "(>= 80%). Esse contraste espacial separa defeito/sabotagem de antena individual de problemas da rede."
        ),
        recommended_action="Inspeção de campo no medidor específico (verificar transceptor óptico/RF, caixa metálica/gaiola de Faraday ou sabotagem de antena).",
    ),
    CategoryMethodologySpec(
        code="GARGALO_COLETIVO_RF_MESH",
        report_slug="relatorio_03b_gargalo_coletivo_malha_mesh",
        title="3B. Gargalo Coletivo de Rede RF Mesh (Vizinhança/Transformador Inteiro Degradado)",
        paradigm="Agregação Topológica por Transformador (Posto Operacional) e Alimentador",
        uses_ml="NÃO (Regra Topológica Coletiva: Cobertura UC < 60% e Mediana dos Vizinhos < 80%).",
        justification=(
            "Quando vários medidores do mesmo transformador ou rua falham simultaneamente na entrega de pacotes, "
            "a causa é infraestrutura de telecomunicações (concentrador DCU, repetidor ou sombra topográfica), "
            "jamais fraude elétrica individual."
        ),
        recommended_action="Encaminhar à equipe de Telecom/Automação para manutenção de repetidor/concentrador RF Mesh (não enviar equipe comercial).",
    ),
    CategoryMethodologySpec(
        code="MUDANCA_CLASSE_COMPORTAMENTAL",
        report_slug="relatorio_04_mudanca_classe_comportamental",
        title="4. Mudança de Classe Comportamental ao Longo do Tempo (Transição Longitudinal entre Cutoffs)",
        paradigm="Classificação de Arquétipos de Consumo + Matriz de Transição Temporal entre Cutoffs",
        uses_ml="SIM (Agrupamento/Classificação de Perfil Comportamental + Detecção Longitudinal de Mudança de Regime).",
        justification=(
            "Consumidores semelhantes formam classes comportamentais bem definidas (Mínimo/Desocupado, Residencial "
            "Baixo, Residencial Padrão, Comercial/Elevado, Alta Carga, Prosumidor). Rastrear quais UCs mudaram de "
            "classe entre Ago/2026, Set/2026, Out/2026 e Nov/2026 permite flagrar tanto trocas de titularidade/hábito "
            "quanto quedas abruptas de classe causadas por derivação clandestina (bypass)."
        ),
        recommended_action="Cruzar quedas de classe (ex: C3/C4 -> C1/C2 sem troca de titularidade) com suspeita de PNT; validar subidas de classe com atualização cadastral.",
    ),
    CategoryMethodologySpec(
        code="ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT",
        report_slug="relatorio_05_anomalias_eletricas_fraude_pnt",
        title="5. Anomalia Elétrica Pura — Suspeita de Fraude / PNT / Desequilíbrio de Fase",
        paradigm="ML Não-Supervisionado PyOD Ensemble (IsoForest + LOF + ECOD + COPOD + HBOS + PCA) sobre Features Elétricas Puras",
        uses_ml="SIM — Onde o ML Não-Supervisionado é Essencial e Insubstituível.",
        justification=(
            "Fraudes elétricas (derivação parcial, corte de bobina de potencial, bypass em 1 fase de medidor "
            "bifásico/trifásico) criam combinações multivariadas sutis entre Consumo (FA), Dispersão (MAD/CV), "
            "Tendência (Slope), Fator de Carga e Desbalanço de Corrente sob carga (>= 1A) que nenhuma regra fixa "
            "isolada captura. O comitê PyOD opera 100% livre de variáveis de comunicação."
        ),
        recommended_action="Priorizar para inspeção comercial antifraude (ordenadas pelo Consensus Score PyOD e severidade de desbalanço/queda).",
    ),
    CategoryMethodologySpec(
        code="POSSIVEL_GD_NAO_CADASTRADA",
        report_slug="relatorio_06_gd_nao_cadastrada_reversao",
        title="6. Possível Geração Distribuída (GD) Não Cadastrada / Inversão de Fluxo Ativo",
        paradigm="Híbrido: Física do Medidor (RA_REVERSAL_RATIO > 0.08 e Dias Reversos >= 3) + Score de Cauda ECOD/COPOD",
        uses_ml="HÍBRIDO (Gatilho Físico de Energia Reversa + Score de Intensidade COPOD/ECOD).",
        justification=(
            "A injeção de energia ativa reversa (RA > 0) na rede é um fenômeno físico direto medido pelo canal "
            "reverso do smart meter. A regra física identifica a injeção, enquanto o score estatístico quantifica "
            "a recorrência e intensidade diurna para separar micro-inversões transitórias de geração solar real."
        ),
        recommended_action="Verificar no cadastro comercial se a UC possui parecer de acesso de microgeração; se não, notificar por GD clandestina.",
    ),
    CategoryMethodologySpec(
        code="FALHA_METROLOGICA_MEDIDOR_TRAVADO",
        report_slug="relatorio_07_falha_metrologica_medidor_travado",
        title="7. Indício de Falha Metrológica / Medidor Travado ou Shunt Total com Tensão Normal",
        paradigm="Regra Física/Metrológica + Contraste de Vizinhança Elétrica",
        uses_ml="HÍBRIDO (Regra Física de Tensão Energizada + Consumo Zero Consecutivo + Score HBOS/LOF).",
        justification=(
            "Um medidor que comunica regularmente (>= 60% de cobertura), apresenta tensão de linha normal "
            "(> 100V), mas registra consumo estritamente zero ou quase zero (< 0.05 kWh/dia) com MAD = 0 "
            "enquanto a UC consta ativa indica falha de TC/registrador ou ponteamento total (shunt)."
        ),
        recommended_action="Cruzar com status de ocupação/faturamento; se imóvel ativo, gerar ordem de serviço para verificação metrológica ou fraude total.",
    ),
)


def _read_context_flexible(path_no_ext: Path) -> pl.DataFrame:
    p_file = path_no_ext.with_suffix(".parquet")
    c_file = path_no_ext.with_suffix(".csv")
    if p_file.exists() and p_file.stat().st_size > 0:
        return pl.read_parquet(p_file)
    if c_file.exists() and c_file.stat().st_size > 0:
        return pl.read_csv(c_file, separator=";", infer_schema_length=5000, ignore_errors=True)
    return pl.DataFrame()


def enrich_lifecycle_and_spatial_context(
    df: pl.DataFrame,
    base_dir: Path | str = "output",
) -> pl.DataFrame:
    """Enrich a `training_dataset` DataFrame with historical meter count, UC start date, coordinates, and spatial/transformer peer coverage."""
    if df.is_empty():
        return df

    root = Path(base_dir)
    work = df.clone()

    uc_col = "id__uc_id" if "id__uc_id" in work.columns else "UC"
    work = work.with_columns(
        pl.col(uc_col).cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias(uc_col)
    )
    if "meta__cutoff_date" in work.columns and work["meta__cutoff_date"].dtype == pl.String:
        work = work.with_columns(pl.col("meta__cutoff_date").str.to_date(strict=False))

    # 1. Enrich N_METERS_HISTORICAL and FIRST/LATEST INSTALLATION DATE from meter_installation_history
    if "meta__n_meters_historical" not in work.columns:
        mh_df = _read_context_flexible(root / "context" / "meter_installation_history")
        if not mh_df.is_empty() and "UC" in mh_df.columns and "DATA_INSTALACAO" in mh_df.columns:
            nio_expr = (
                pl.col("NIO").cast(pl.String).str.strip_chars().str.replace(r"^0+", "")
                if "NIO" in mh_df.columns
                else pl.col("DATA_INSTALACAO").cast(pl.String)
            )
            mh_clean = (
                mh_df.select([
                    pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_MH_UC"),
                    pl.col("DATA_INSTALACAO").cast(pl.Date, strict=False).alias("_DT_INST"),
                    nio_expr.alias("_NORM_NIO"),
                ])
                .filter(pl.col("_MH_UC").is_not_null() & (pl.col("_MH_UC") != "") & pl.col("_DT_INST").is_not_null())
            )
            # Compute per-UC total distinct meters and earliest/latest installation date
            mh_summary = mh_clean.group_by("_MH_UC").agg([
                pl.col("_NORM_NIO").drop_nulls().n_unique().clip(lower_bound=1).cast(pl.Int16).alias("meta__n_meters_historical"),
                pl.col("_DT_INST").min().alias("meta__first_meter_install_date"),
                pl.col("_DT_INST").max().alias("meta__latest_meter_install_date"),
            ])
            work = work.join(mh_summary, left_on=uc_col, right_on="_MH_UC", how="left")

    if "meta__n_meters_historical" not in work.columns:
        work = work.with_columns(pl.lit(1, dtype=pl.Int16).alias("meta__n_meters_historical"))
    else:
        work = work.with_columns(pl.col("meta__n_meters_historical").fill_null(1).cast(pl.Int16))

    # 2. Enrich INICIO_UC, BAIRRO, LATITUDE, LONGITUDE from uc_context
    ctx_df = _read_context_flexible(root / "context" / "uc_context")
    if not ctx_df.is_empty() and "UC" in ctx_df.columns:
        ctx_exprs: list[pl.Expr] = [
            pl.col("UC").cast(pl.String).str.strip_chars().str.replace(r"^0+", "").alias("_CTX_UC")
        ]
        if "INICIO_UC" in ctx_df.columns and "meta__inicio_uc" not in work.columns:
            ctx_exprs.append(pl.col("INICIO_UC").cast(pl.Date, strict=False).alias("meta__inicio_uc"))
        if "BAIRRO" in ctx_df.columns and "meta__bairro" not in work.columns:
            ctx_exprs.append(pl.col("BAIRRO").cast(pl.String).alias("meta__bairro"))
        if "LATITUDE" in ctx_df.columns and "meta__latitude" not in work.columns:
            ctx_exprs.append(pl.col("LATITUDE").cast(pl.Float64, strict=False).alias("meta__latitude"))
        if "LONGITUDE" in ctx_df.columns and "meta__longitude" not in work.columns:
            ctx_exprs.append(pl.col("LONGITUDE").cast(pl.Float64, strict=False).alias("meta__longitude"))
        if len(ctx_exprs) > 1:
            ctx_sub = ctx_df.select(ctx_exprs).unique(subset=["_CTX_UC"], keep="first")
            work = work.join(ctx_sub, left_on=uc_col, right_on="_CTX_UC", how="left")

    # 3. Compute Spatial & Topological Peer Communication Coverage (Posto Operacional -> Spatial 250m Grid -> Feeder)
    cov_col = (
        pl.col("x__avg_interval_coverage").fill_null(0.0)
        if "x__avg_interval_coverage" in work.columns
        else pl.lit(1.0)
    )
    cutoff_key = "meta__cutoff_date" if "meta__cutoff_date" in work.columns else uc_col

    # Spatial ~250m grid bucket from LATITUDE/LONGITUDE (0.0025 deg ~ 250m) if coordinates exist
    if "meta__latitude" in work.columns and "meta__longitude" in work.columns:
        work = work.with_columns(
            pl.when(pl.col("meta__latitude").is_not_null() & pl.col("meta__longitude").is_not_null())
            .then(
                (pl.col("meta__latitude") / 0.0025).round(0).cast(pl.Int32).cast(pl.String)
                + pl.lit("_")
                + (pl.col("meta__longitude") / 0.0025).round(0).cast(pl.Int32).cast(pl.String)
            )
            .otherwise(None)
            .alias("_SPATIAL_GRID_250M")
        )
    else:
        work = work.with_columns(pl.lit(None, dtype=pl.String).alias("_SPATIAL_GRID_250M"))

    has_posto = "meta__posto" in work.columns
    has_feeder = "meta__feeder" in work.columns

    posto_med = cov_col.median().over([cutoff_key, "meta__posto"]) if has_posto else pl.lit(None, dtype=pl.Float64)
    posto_cnt = pl.len().over([cutoff_key, "meta__posto"]) if has_posto else pl.lit(0)
    posto_good_ratio = (
        (cov_col >= 0.75).cast(pl.Float64).mean().over([cutoff_key, "meta__posto"])
        if has_posto
        else pl.lit(None, dtype=pl.Float64)
    )

    grid_med = cov_col.median().over([cutoff_key, "_SPATIAL_GRID_250M"])
    grid_cnt = pl.len().over([cutoff_key, "_SPATIAL_GRID_250M"])
    grid_good_ratio = (cov_col >= 0.75).cast(pl.Float64).mean().over([cutoff_key, "_SPATIAL_GRID_250M"])

    feeder_med = cov_col.median().over([cutoff_key, "meta__feeder"]) if has_feeder else cov_col.median().over(cutoff_key)
    feeder_cnt = pl.len().over([cutoff_key, "meta__feeder"]) if has_feeder else pl.len().over(cutoff_key)
    feeder_good_ratio = (
        (cov_col >= 0.75).cast(pl.Float64).mean().over([cutoff_key, "meta__feeder"])
        if has_feeder
        else (cov_col >= 0.75).cast(pl.Float64).mean().over(cutoff_key)
    )

    use_posto = pl.col("meta__posto").is_not_null() & (posto_cnt >= 3) if has_posto else pl.lit(False)
    use_grid = pl.col("_SPATIAL_GRID_250M").is_not_null() & (grid_cnt >= 3)

    work = work.with_columns([
        pl.when(use_posto)
        .then(posto_med)
        .when(use_grid)
        .then(grid_med)
        .otherwise(feeder_med)
        .fill_null(0.95)
        .round(4)
        .alias("meta__peer_median_coverage"),
        pl.when(use_posto)
        .then(posto_good_ratio)
        .when(use_grid)
        .then(grid_good_ratio)
        .otherwise(feeder_good_ratio)
        .fill_null(0.95)
        .round(4)
        .alias("meta__peer_healthy_ratio"),
        pl.when(use_posto)
        .then(posto_cnt)
        .when(use_grid)
        .then(grid_cnt)
        .otherwise(feeder_cnt)
        .fill_null(1)
        .cast(pl.Int32)
        .alias("meta__peer_neighbors_count"),
        pl.when(use_posto)
        .then(pl.lit("POSTO_TRANSFORMADOR"))
        .when(use_grid)
        .then(pl.lit("VIZINHANCA_GEO_250M"))
        .otherwise(pl.lit("ALIMENTADOR"))
        .alias("meta__peer_proximity_basis"),
    ]).with_columns(
        (pl.col("meta__peer_median_coverage") - cov_col)
        .clip(0.0, 1.0)
        .round(4)
        .alias("meta__coverage_deficit_vs_peers")
    ).drop("_SPATIAL_GRID_250M")

    return work


def assign_behavioral_consumption_classes(df: pl.DataFrame) -> pl.DataFrame:
    """Assign each UC × Cutoff window to an interpretable electrical consumption class (C1..C6)."""
    if df.is_empty():
        return df

    fa_med = (
        pl.col("x__hist_fa_sum_median").fill_null(0.0)
        if "x__hist_fa_sum_median" in df.columns
        else pl.lit(0.0)
    )
    ra_max = (
        pl.col("x__hist_ra_reversal_ratio_max").fill_null(0.0)
        if "x__hist_ra_reversal_ratio_max" in df.columns
        else pl.lit(0.0)
    )

    class_expr = (
        pl.when(ra_max > 0.05)
        .then(pl.lit("C6_PROSUMIDOR_INJECAO_REVERSA"))
        .when(fa_med < 0.50)
        .then(pl.lit("C1_MINIMO_OU_DESOCUPADO"))
        .when(fa_med < 3.50)
        .then(pl.lit("C2_RESIDENCIAL_BAIXO_CONSUMO"))
        .when(fa_med < 10.00)
        .then(pl.lit("C3_RESIDENCIAL_PADRAO"))
        .when(fa_med < 25.00)
        .then(pl.lit("C4_COMERCIAL_OU_ELEVADO"))
        .otherwise(pl.lit("C5_ALTA_CARGA_CONTINUA"))
    )

    tier_num_expr = (
        pl.when(ra_max > 0.05)
        .then(pl.lit(6, dtype=pl.Int8))
        .when(fa_med < 0.50)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(fa_med < 3.50)
        .then(pl.lit(2, dtype=pl.Int8))
        .when(fa_med < 10.00)
        .then(pl.lit(3, dtype=pl.Int8))
        .when(fa_med < 25.00)
        .then(pl.lit(4, dtype=pl.Int8))
        .otherwise(pl.lit(5, dtype=pl.Int8))
    )

    return df.with_columns([
        class_expr.alias("CLASSE_COMPORTAMENTAL"),
        tier_num_expr.alias("ORDEM_CLASSE_COMPORTAMENTAL"),
    ])


def build_temporal_class_transition_report(df_with_classes: pl.DataFrame) -> pl.DataFrame:
    """Identify all UCs that changed behavioral consumption class across analyzed cutoffs."""
    if df_with_classes.is_empty() or "meta__cutoff_date" not in df_with_classes.columns:
        return pl.DataFrame()

    uc_col = "id__uc_id" if "id__uc_id" in df_with_classes.columns else "UC"
    if "CLASSE_COMPORTAMENTAL" not in df_with_classes.columns:
        df_with_classes = assign_behavioral_consumption_classes(df_with_classes)

    # Consider windows with minimal telemetry (>= 5 reliable days or clean gate) so packet drops don't fake class changes
    if "x__coverage_reliable_days" in df_with_classes.columns:
        valid_w = df_with_classes.filter(pl.col("x__coverage_reliable_days").fill_null(0) >= 5)
    elif "meta__is_clean_for_train" in df_with_classes.columns:
        valid_w = df_with_classes.filter(pl.col("meta__is_clean_for_train").fill_null(True))
    else:
        valid_w = df_with_classes

    if valid_w.is_empty():
        valid_w = df_with_classes

    sorted_w = valid_w.sort([uc_col, "meta__cutoff_date"])

    agg_exprs: list[pl.Expr] = [
        pl.len().cast(pl.Int16).alias("N_JANELAS_AVALIADAS"),
        pl.col("CLASSE_COMPORTAMENTAL").n_unique().cast(pl.Int16).alias("N_CLASSES_DISTINTAS"),
        pl.col("meta__cutoff_date").first().alias("PRIMEIRO_CUTOFF"),
        pl.col("meta__cutoff_date").last().alias("ULTIMO_CUTOFF"),
        pl.col("CLASSE_COMPORTAMENTAL").first().alias("CLASSE_INICIAL"),
        pl.col("CLASSE_COMPORTAMENTAL").last().alias("CLASSE_FINAL"),
        pl.col("ORDEM_CLASSE_COMPORTAMENTAL").first().alias("_TIER_INI"),
        pl.col("ORDEM_CLASSE_COMPORTAMENTAL").last().alias("_TIER_FIM"),
        (
            pl.col("meta__cutoff_date").cast(pl.String)
            + pl.lit(":")
            + pl.col("CLASSE_COMPORTAMENTAL")
        ).str.join(" -> ").alias("TRAJETORIA_CLASSES_TEMPORAL"),
        pl.col("x__hist_fa_sum_median").first().round(4).alias("CONSUMO_INICIAL_KWH_DIA"),
        pl.col("x__hist_fa_sum_median").last().round(4).alias("CONSUMO_FINAL_KWH_DIA"),
        pl.col("x__hist_fa_sum_median").min().round(4).alias("CONSUMO_MIN_JANELAS"),
        pl.col("x__hist_fa_sum_median").max().round(4).alias("CONSUMO_MAX_JANELAS"),
    ]
    if "meta__feeder" in sorted_w.columns:
        agg_exprs.append(pl.col("meta__feeder").last().alias("ALIMENTADOR"))
    if "meta__posto" in sorted_w.columns:
        agg_exprs.append(pl.col("meta__posto").last().alias("POSTO_OPERACIONAL"))
    if "x__meter_changed_30d" in sorted_w.columns:
        agg_exprs.append(pl.col("x__meter_changed_30d").fill_null(False).any().alias("TEVE_TROCA_MEDIDOR_PERIODO"))
    else:
        agg_exprs.append(pl.lit(False).alias("TEVE_TROCA_MEDIDOR_PERIODO"))

    if "ANOMALY_SCORE" in sorted_w.columns:
        agg_exprs.append(pl.col("ANOMALY_SCORE").max().round(4).alias("MAX_ANOMALY_SCORE"))

    transitions = (
        sorted_w.group_by(uc_col)
        .agg(agg_exprs)
        .filter((pl.col("N_JANELAS_AVALIADAS") >= 2) & (pl.col("N_CLASSES_DISTINTAS") >= 2))
        .with_columns([
            (pl.col("CONSUMO_FINAL_KWH_DIA") - pl.col("CONSUMO_INICIAL_KWH_DIA"))
            .round(4)
            .alias("DELTA_CONSUMO_KWH_DIA"),
            (
                (pl.col("CONSUMO_FINAL_KWH_DIA") - pl.col("CONSUMO_INICIAL_KWH_DIA"))
                / (pl.col("CONSUMO_INICIAL_KWH_DIA").abs() + 0.25)
                * 100.0
            )
            .round(2)
            .alias("VARIACAO_PERCENTUAL_CONSUMO"),
            (pl.col("_TIER_FIM") - pl.col("_TIER_INI")).cast(pl.Int8).alias("SALTO_DEGRAUS_CLASSE"),
        ])
    )

    if transitions.is_empty():
        return transitions

    hipotese_expr = (
        pl.when(
            (pl.col("CLASSE_FINAL") == "C6_PROSUMIDOR_INJECAO_REVERSA")
            & (pl.col("CLASSE_INICIAL") != "C6_PROSUMIDOR_INJECAO_REVERSA")
        )
        .then(pl.lit("NOVO_PROSUMIDOR_ENTRADA_GD_SOLAR"))
        .when(pl.col("TEVE_TROCA_MEDIDOR_PERIODO"))
        .then(pl.lit("MUDANCA_CLASSE_ASSOCIADA_TROCA_MEDIDOR"))
        .when(
            (pl.col("SALTO_DEGRAUS_CLASSE") <= -1)
            & (pl.col("VARIACAO_PERCENTUAL_CONSUMO") <= -45.0)
            & (pl.col("CONSUMO_INICIAL_KWH_DIA") >= 2.0)
        )
        .then(pl.lit("QUEDA_ABRUPTA_CLASSE_SUSPEITA_BYPASS_OU_DESOCUPACAO"))
        .when(pl.col("SALTO_DEGRAUS_CLASSE") >= 1)
        .then(pl.lit("ASCENSAO_CLASSE_TROCA_TITULARIDADE_OU_NOVO_EQUIPAMENTO"))
        .otherwise(pl.lit("OSCILACAO_SAZONAL_ENTRE_CLASSES_ADJACENTES"))
    )

    return (
        transitions.with_columns(hipotese_expr.alias("HIPOTESE_MUDANCA_CLASSE"))
        .drop(["_TIER_INI", "_TIER_FIM"])
        .sort(["VARIACAO_PERCENTUAL_CONSUMO", "DELTA_CONSUMO_KWH_DIA"])
    )


def apply_delimited_isolation_and_categories(
    df: pl.DataFrame,
    *,
    base_dir: Path | str = "output",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply the 7 delimited operational rules + ML scores and attach behavioral class transitions.

    Returns:
        Tuple of `(enriched_classified_df, temporal_class_transitions_df)`.
    """
    if df.is_empty():
        return df, pl.DataFrame()

    uc_col = "id__uc_id" if "id__uc_id" in df.columns else "UC"
    work = enrich_lifecycle_and_spatial_context(df, base_dir=base_dir)
    work = assign_behavioral_consumption_classes(work)

    transitions_df = build_temporal_class_transition_report(work)
    if not transitions_df.is_empty():
        trans_lookup = transitions_df.select([
            pl.col(uc_col),
            pl.lit(True).alias("MUDOU_CLASSE_TEMPORAL"),
            pl.col("CLASSE_INICIAL"),
            pl.col("CLASSE_FINAL"),
            pl.col("TRAJETORIA_CLASSES_TEMPORAL"),
            pl.col("HIPOTESE_MUDANCA_CLASSE"),
        ])
        work = work.join(trans_lookup, on=uc_col, how="left")
    else:
        work = work.with_columns([
            pl.lit(False).alias("MUDOU_CLASSE_TEMPORAL"),
            pl.col("CLASSE_COMPORTAMENTAL").alias("CLASSE_INICIAL"),
            pl.col("CLASSE_COMPORTAMENTAL").alias("CLASSE_FINAL"),
            pl.col("CLASSE_COMPORTAMENTAL").alias("TRAJETORIA_CLASSES_TEMPORAL"),
            pl.lit("ESTAVEL_NO_PERIODO").alias("HIPOTESE_MUDANCA_CLASSE"),
        ])

    work = work.with_columns(pl.col("MUDOU_CLASSE_TEMPORAL").fill_null(False))

    # Extract columns with safe defaults for clear rule evaluation
    is_clean = (
        pl.col("meta__is_clean_for_train").fill_null(True)
        if "meta__is_clean_for_train" in work.columns
        else pl.lit(True)
    )
    cov = (
        pl.col("x__avg_interval_coverage").fill_null(1.0)
        if "x__avg_interval_coverage" in work.columns
        else pl.lit(1.0)
    )
    rel_days = (
        pl.col("x__coverage_reliable_days").fill_null(30)
        if "x__coverage_reliable_days" in work.columns
        else pl.lit(30)
    )
    peer_cov = (
        pl.col("meta__peer_median_coverage").fill_null(0.95)
        if "meta__peer_median_coverage" in work.columns
        else pl.lit(0.95)
    )
    cov_deficit = (
        pl.col("meta__coverage_deficit_vs_peers").fill_null(0.0)
        if "meta__coverage_deficit_vs_peers" in work.columns
        else pl.lit(0.0)
    )
    meter_age = (
        pl.col("x__meter_age_days").fill_null(999)
        if "x__meter_age_days" in work.columns
        else pl.lit(999)
    )
    meter_changed_30d = (
        pl.col("x__meter_changed_30d").fill_null(False)
        if "x__meter_changed_30d" in work.columns
        else pl.lit(False)
    )
    n_meters_hist = (
        pl.col("meta__n_meters_historical").fill_null(1)
        if "meta__n_meters_historical" in work.columns
        else pl.lit(1)
    )
    is_anom = (
        pl.col("IS_ANOMALY").fill_null(False)
        if "IS_ANOMALY" in work.columns
        else pl.lit(False)
    )
    fa_med = (
        pl.col("x__hist_fa_sum_median").fill_null(0.0)
        if "x__hist_fa_sum_median" in work.columns
        else pl.lit(0.0)
    )
    ra_ratio = (
        pl.col("x__hist_ra_reversal_ratio_max").fill_null(0.0)
        if "x__hist_ra_reversal_ratio_max" in work.columns
        else pl.lit(0.0)
    )
    ra_days = (
        pl.col("x__hist_ra_reversal_days").fill_null(0)
        if "x__hist_ra_reversal_days" in work.columns
        else pl.lit(0)
    )
    c_imb = (
        pl.col("x__hist_current_imbalance_max").fill_null(0.0)
        if "x__hist_current_imbalance_max" in work.columns
        else pl.lit(0.0)
    )
    fa_slope = (
        pl.col("x__trend_fa_slope").fill_null(0.0)
        if "x__trend_fa_slope" in work.columns
        else pl.lit(0.0)
    )

    # Check if UC itself started within 30 days of cutoff
    if "meta__inicio_uc" in work.columns and "meta__cutoff_date" in work.columns:
        uc_age_days = (pl.col("meta__cutoff_date") - pl.col("meta__inicio_uc")).dt.total_days()
        is_recent_uc_cadastre = uc_age_days.is_not_null() & (uc_age_days >= 0) & (uc_age_days <= 30)
    else:
        is_recent_uc_cadastre = pl.lit(False)

    is_recent_hardware = (meter_changed_30d & (meter_age > 0) & (meter_age <= 30)) | is_recent_uc_cadastre
    is_new_uc_initiation = is_recent_hardware & (n_meters_hist <= 1)
    is_meter_replacement = is_recent_hardware & (n_meters_hist >= 2)

    is_poor_comm = (~is_clean) | (cov < 0.60)
    is_isolated_comm_anomaly = is_poor_comm & (~is_recent_hardware) & (peer_cov >= 0.80) & (cov_deficit >= 0.30)
    is_collective_mesh_bottleneck = is_poor_comm & (~is_recent_hardware) & (~is_isolated_comm_anomaly)

    # Electrical categories evaluated on clean, mature telemetry windows
    is_gd_reversal = (~is_poor_comm) & (ra_ratio >= 0.08) & ((ra_days >= 2) | is_anom)
    is_stuck_meter = (~is_poor_comm) & (~is_new_uc_initiation) & (fa_med < 0.05) & is_anom
    is_elec_fraud_pnt = (
        (~is_poor_comm)
        & (~is_new_uc_initiation)
        & (~is_gd_reversal)
        & (~is_stuck_meter)
        & is_anom
    )

    categoria_expr = (
        pl.when(is_new_uc_initiation)
        .then(pl.lit("INICIACAO_PERFIL_UC_NOVA"))
        .when(is_meter_replacement)
        .then(pl.lit("TRANSICAO_TROCA_MEDIDOR"))
        .when(is_isolated_comm_anomaly)
        .then(pl.lit("ANOMALIA_COMUNICACAO_ISOLADA"))
        .when(is_collective_mesh_bottleneck)
        .then(pl.lit("GARGALO_COLETIVO_RF_MESH"))
        .when(is_gd_reversal)
        .then(pl.lit("POSSIVEL_GD_NAO_CADASTRADA"))
        .when(is_stuck_meter)
        .then(pl.lit("FALHA_METROLOGICA_MEDIDOR_TRAVADO"))
        .when(is_elec_fraud_pnt)
        .then(pl.lit("ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT"))
        .when(pl.col("MUDOU_CLASSE_TEMPORAL"))
        .then(pl.lit("MUDANCA_CLASSE_COMPORTAMENTAL"))
        .otherwise(pl.lit("OPERACAO_REGULAR"))
    )

    subtipo_eletrico_expr = (
        pl.when(is_new_uc_initiation & (rel_days >= 7))
        .then(pl.lit("UC_NOVA_COM_PERFIL_PRELIMINAR_") + pl.col("CLASSE_COMPORTAMENTAL"))
        .when(is_new_uc_initiation)
        .then(pl.lit("UC_NOVA_AGUARDANDO_MATURACAO_7D"))
        .when(is_meter_replacement)
        .then(pl.lit("SUBSTITUICAO_MEDIDOR_EM_UC_EXISTENTE"))
        .when(is_isolated_comm_anomaly)
        .then(pl.lit("FALHA_TRANSCEPTOR_OU_ANTENA_COM_VIZINHOS_SAUDAVEIS"))
        .when(is_collective_mesh_bottleneck)
        .then(pl.lit("SOMBRA_COLETIVA_CONCENTRADOR_OU_TRAFO"))
        .when(is_gd_reversal)
        .then(pl.lit("INJECAO_ATIVA_REVERSA_RA"))
        .when(is_stuck_meter)
        .then(pl.lit("CONSUMO_ZERO_COM_TENSAO_NORMAL"))
        .when(is_elec_fraud_pnt & (c_imb > 0.40))
        .then(pl.lit("DESEQUILIBRIO_SEVERO_CORRENTE_BYPASS_FASE"))
        .when(is_elec_fraud_pnt & (fa_slope < -1.0))
        .then(pl.lit("QUEDA_TENDENCIA_CONSUMO_SLOPE_NEGATIVO"))
        .when(is_elec_fraud_pnt)
        .then(pl.lit("OUTLIER_MULTIVARIADO_PYOD_CONSUMO_TENSAO_CORRENTE"))
        .when(pl.col("MUDOU_CLASSE_TEMPORAL"))
        .then(pl.col("HIPOTESE_MUDANCA_CLASSE"))
        .otherwise(pl.lit("PERFIL_ESTAVEL_DENTRO_DA_CLASSE"))
    )

    metodo_deteccao_expr = (
        pl.when(is_new_uc_initiation | is_meter_replacement)
        .then(pl.lit("REGRA_DETERMINISTICA_CICLO_VIDA_CADASTRO"))
        .when(is_isolated_comm_anomaly | is_collective_mesh_bottleneck)
        .then(pl.lit("CONTRASTE_ESPACIAL_VIZINHANCA_POSTO_GEO"))
        .when(is_gd_reversal | is_stuck_meter)
        .then(pl.lit("HIBRIDO_REGRA_FISICA_E_CAUDA_PYOD"))
        .when(is_elec_fraud_pnt)
        .then(pl.lit("ML_NAO_SUPERVISIONADO_PYOD_ENSEMBLE_6_MODELOS"))
        .when(pl.col("MUDOU_CLASSE_TEMPORAL"))
        .then(pl.lit("CLASSIFICACAO_LONGITUDINAL_TRANSICAO_CLASSE"))
        .otherwise(pl.lit("BASELINE_NORMAL"))
    )

    work = work.with_columns([
        categoria_expr.alias("CATEGORIA_DIAGNOSTICA"),
        subtipo_eletrico_expr.alias("SUBTIPO_DIAGNOSTICO"),
        metodo_deteccao_expr.alias("METODO_DETECCAO_APLICADO"),
    ])

    return work, transitions_df


def export_all_category_reports(
    classified_df: pl.DataFrame,
    transitions_df: pl.DataFrame,
    *,
    reports_dir: Path | str = "output/reports",
    write_csv: bool = True,
) -> list[dict[str, Any]]:
    """Export dedicated Parquet + CSV report files for every operational category and return manifest."""
    rep_root = Path(reports_dir)
    cat_dir = rep_root / "categorias"
    cat_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []

    for spec in CATEGORY_METHODOLOGY_SPECS:
        if spec.code == "MUDANCA_CLASSE_COMPORTAMENTAL":
            sub_df = transitions_df
        else:
            sub_df = (
                classified_df.filter(pl.col("CATEGORIA_DIAGNOSTICA") == spec.code)
                if "CATEGORIA_DIAGNOSTICA" in classified_df.columns
                else pl.DataFrame()
            )
            if not sub_df.is_empty() and "ANOMALY_SCORE" in sub_df.columns:
                sub_df = sub_df.sort("ANOMALY_SCORE", descending=True)

        p_path = cat_dir / f"{spec.report_slug}.parquet"
        c_path = cat_dir / f"{spec.report_slug}.csv"

        sub_df.write_parquet(p_path, compression="zstd")
        if write_csv:
            sub_df.write_csv(c_path, separator=";")

        manifest.append({
            "category_code": spec.code,
            "report_slug": spec.report_slug,
            "title": spec.title,
            "paradigm": spec.paradigm,
            "uses_ml": spec.uses_ml,
            "justification": spec.justification,
            "recommended_action": spec.recommended_action,
            "row_count": sub_df.height,
            "unique_ucs": (
                int(sub_df.get_column("id__uc_id").n_unique())
                if not sub_df.is_empty() and "id__uc_id" in sub_df.columns
                else (
                    int(sub_df.get_column("UC").n_unique())
                    if not sub_df.is_empty() and "UC" in sub_df.columns
                    else 0
                )
            ),
            "parquet_path": str(p_path),
            "csv_path": str(c_path) if write_csv else None,
        })

    return manifest

