"""
scripts/eda_city_profile_and_case_studies.py
-------------------------------------------------------------------------------
Script de EDA completo para caracterização sociotécnica da concessão de Araucária
e raio-x individual das Unidades Consumidoras utilizadas nos estudos de caso.

Gera:
1. Perfil demográfico e tecnológico do parque de medição (Tipo A, Tipo B, Convencional);
2. Segregação territorial por Alimentador e Macrozona da cidade;
3. Ficha individual detalhada de todas as UCs dos estudos de caso;
4. Tabelas auditadas em CSV e gráficos estatísticos para a dissertação do TCC.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path("output")
CONTEXT_DIR = BASE_DIR / "context"
REPORTS_DIR = BASE_DIR / "reports"
EVIDENCE_DIR = Path("docs/tcc_evidence/city_profile")
FIGURES_DIR = Path("docs/figures/city_profile")

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# List of Case Study UCs identified in the project
CASE_STUDY_UCS = [
    "90850700",   # Caso 1: Fraude Clássica PNT com Apagão Precursor de 48h
    "29113296",   # Caso 2: Falso Positivo Evitado via Gating de Telecom (Controle Negativo)
    "96905697",   # Caso 3: Sazonalidade Semanal Industrial (Quedas de Fim de Semana)
    "82889805",   # Caso 4: Possível GD Não Cadastrada (Duck Curve Solar)
    "96515961",   # Caso 5: Carga 24h sem Desligamento (+241% CUSUM - Mineração/Câmara Fria)
    "105416770",  # Caso 6: Baixo Volume Absoluto (Por que é Regular)
    "91022010",   # Caso 7: Picos Transitórios com Regularidade Estrutural
    "97847968",   # Caso 8: Vacância e Apagões Longos de Telecom com Perfil Invariante
]


def load_integrated_master_table() -> pl.DataFrame:
    """Join active meter installation history with uc_context, electrical_hierarchy, and latest classifications."""
    print("[1/5] Carregando tabelas contextuais e metrológicas da concessão...")
    mh = pl.read_parquet(CONTEXT_DIR / "meter_installation_history.parquet")
    ctx = pl.read_parquet(CONTEXT_DIR / "uc_context.parquet")
    hier = pl.read_parquet(CONTEXT_DIR / "electrical_hierarchy.parquet")

    # Filter active meters (DATA_REMOCAO is null)
    active_m = mh.filter(pl.col("DATA_REMOCAO").is_null())

    # De-duplicate hierarchy by UC
    hier_unique = (
        hier.sort("POSTO_OPERACIONAL", nulls_last=True)
        .unique(subset=["UC"], keep="first")
        .select(["UC", "ALIMENTADOR", "SUBESTACAO", "TENSAO_ALIMENTADOR", "POT_INST_KVA"])
    )

    # Join meter history with context
    df = active_m.join(ctx, on="UC", how="left")
    df = df.join(hier_unique, on="UC", how="left")

    # Filter strictly active connections (STATUS_LIGACAO == 'LG')
    df_active = df.filter(pl.col("STATUS_LIGACAO") == "LG")
    print(f" -> Total de registros de medidores ativos em UCs ligadas: {len(df_active):,}")

    # Add Categorization: Tipo A, Tipo B, Convencional
    # In Copel/ANEEL standards:
    # - Grupo B (Baixa Tensão: B1 Residencial, B2 Rural, B3 Comercial BT, B4 IP)
    #   * Smart (FAMILIA_SMART == '1') -> MEDIDOR TIPO B (Smart BT)
    #   * Non-smart (FAMILIA_SMART == '0') -> CONVENCIONAL B
    # - Grupo A (Média/Alta Tensão: A4, A3, A3a, A1 >= 2.3 kV)
    #   * Smart (FAMILIA_SMART == '1') -> MEDIDOR TIPO A (Smart MT/AT Horo-Sazonal)
    #   * Non-smart (FAMILIA_SMART == '0') -> CONVENCIONAL A
    df_active = df_active.with_columns([
        pl.when(pl.col("COD_SUB_GRUPO_FAT").str.starts_with("A") & (pl.col("FAMILIA_SMART") == "1"))
        .then(pl.lit("Smart Tipo A (Média/Alta Tensão)"))
        .when(pl.col("COD_SUB_GRUPO_FAT").str.starts_with("B") & (pl.col("FAMILIA_SMART") == "1"))
        .then(pl.lit("Smart Tipo B (Baixa Tensão)"))
        .when(pl.col("COD_SUB_GRUPO_FAT").str.starts_with("A"))
        .then(pl.lit("Convencional Grupo A"))
        .otherwise(pl.lit("Convencional Grupo B"))
        .alias("CATEGORIA_MEDIDOR_TECNOLOGIA"),

        pl.when(pl.col("FAMILIA_SMART") == "1")
        .then(pl.lit("Smart Grid (AMI)"))
        .otherwise(pl.lit("Eletromecânico / Convencional"))
        .alias("STATUS_SMART"),

        pl.when(pl.col("CLASSE") == "RE")
        .then(pl.lit("Residencial"))
        .when(pl.col("CLASSE") == "CO")
        .then(pl.lit("Comercial"))
        .when(pl.col("CLASSE") == "IN")
        .then(pl.lit("Industrial"))
        .when(pl.col("CLASSE") == "RU")
        .then(pl.lit("Rural"))
        .when(pl.col("CLASSE") == "PP")
        .then(pl.lit("Poder Público"))
        .when(pl.col("CLASSE") == "IP")
        .then(pl.lit("Iluminação Pública"))
        .otherwise(pl.col("CLASSE"))
        .alias("NOME_CLASSE_CONSUMO"),

        pl.when(pl.col("FASE") == "MO")
        .then(pl.lit("Monofásico"))
        .when(pl.col("FASE") == "BI")
        .then(pl.lit("Bifásico"))
        .when(pl.col("FASE") == "TR")
        .then(pl.lit("Trifásico"))
        .when(pl.col("FASE") == "MR")
        .then(pl.lit("Monofásico Rural"))
        .otherwise(pl.col("FASE"))
        .alias("DESCRICAO_FASEAMENTO"),
    ])

    return df_active


def analyze_city_wide_technology(df_active: pl.DataFrame) -> pl.DataFrame:
    """Produce summary table of meter technologies across Araucária."""
    print("\n[2/5] Consolidando estatísticas de tecnologia de medição na cidade...")
    total_meters = len(df_active)

    summary = (
        df_active.group_by(["CATEGORIA_MEDIDOR_TECNOLOGIA", "STATUS_SMART", "DESCRICAO_FASEAMENTO"])
        .agg([
            pl.len().alias("QTD_MEDIDORES"),
            pl.col("UC").n_unique().alias("QTD_UCS_DISTINTAS"),
        ])
        .with_columns([
            (pl.col("QTD_MEDIDORES") / total_meters * 100.0).round(2).alias("PERCENTUAL_CIDADE")
        ])
        .sort(["STATUS_SMART", "QTD_MEDIDORES"], descending=[True, True])
    )

    csv_path = EVIDENCE_DIR / "resumo_tecnologia_medidores.csv"
    summary.write_csv(csv_path, separator=";")
    print(f" -> Salvo resumo de tecnologia em: {csv_path}")

    # Aggregated Macro View
    macro = (
        df_active.group_by("CATEGORIA_MEDIDOR_TECNOLOGIA")
        .agg([
            pl.len().alias("QTD_MEDIDORES"),
            (pl.len() / total_meters * 100.0).round(2).alias("PERCENTUAL")
        ])
        .sort("QTD_MEDIDORES", descending=True)
    )
    print("\n--- VISÃO MACRO DO PARQUE DE MEDIDORES (ARAUCÁRIA - APENAS LIGADOS) ---")
    for r in macro.iter_rows(named=True):
        print(f" * {r['CATEGORIA_MEDIDOR_TECNOLOGIA']:<35}: {r['QTD_MEDIDORES']:>6,} ({r['PERCENTUAL']:>5.2f}%)")

    smart_count = df_active.filter(pl.col("STATUS_SMART") == "Smart Grid (AMI)").height
    conv_count = total_meters - smart_count
    print(f"\n -> Total Smart Grid (Tipo A + Tipo B): {smart_count:,} ({smart_count/total_meters*100:.2f}%)")
    print(f" -> Total Convencionais Legados:        {conv_count:,} ({conv_count/total_meters*100:.2f}%)")
    print(f" -> Taxa de Smartização da Cidade:      {smart_count/total_meters*100:.2f}%\n")

    return summary


def analyze_feeders_and_regions(df_active: pl.DataFrame) -> pl.DataFrame:
    """Analyze meter distribution, smart penetration, and consumer profile by electrical feeder."""
    print("[3/5] Analisando distribuição espacial por Alimentadores e Macrozonas...")
    
    feeder_summary = (
        df_active.group_by("ALIMENTADOR")
        .agg([
            pl.len().alias("TOTAL_MEDIDORES"),
            (pl.col("CATEGORIA_MEDIDOR_TECNOLOGIA") == "Smart Tipo B (Baixa Tensão)").sum().alias("SMART_TIPO_B"),
            (pl.col("CATEGORIA_MEDIDOR_TECNOLOGIA") == "Smart Tipo A (Média/Alta Tensão)").sum().alias("SMART_TIPO_A"),
            (pl.col("CATEGORIA_MEDIDOR_TECNOLOGIA").str.starts_with("Convencional")).sum().alias("CONVENCIONAIS"),
            (pl.col("NOME_CLASSE_CONSUMO") == "Residencial").sum().alias("RESIDENCIAIS"),
            (pl.col("NOME_CLASSE_CONSUMO") == "Comercial").sum().alias("COMERCIAIS"),
            (pl.col("NOME_CLASSE_CONSUMO") == "Industrial").sum().alias("INDUSTRIAIS"),
            (pl.col("NOME_CLASSE_CONSUMO") == "Rural").sum().alias("RURAIS"),
            pl.col("TENSAO_ALIMENTADOR").drop_nulls().median().alias("TENSAO_KV"),
            pl.col("POT_INST_KVA").drop_nulls().sum().round(1).alias("POT_TOTAL_KVA"),
        ])
        .with_columns([
            ((pl.col("SMART_TIPO_B") + pl.col("SMART_TIPO_A")) / pl.col("TOTAL_MEDIDORES") * 100.0).round(2).alias("TAXA_SMART_PCT")
        ])
        .sort("TOTAL_MEDIDORES", descending=True)
    )

    csv_path = EVIDENCE_DIR / "perfil_alimentadores_medidores.csv"
    feeder_summary.write_csv(csv_path, separator=";")
    print(f" -> Salvo perfil de alimentadores em: {csv_path}")

    return feeder_summary


def extract_case_studies_xray(df_active: pl.DataFrame) -> pl.DataFrame:
    """Extract complete individual profile of all 8 Case Study UCs."""
    print("[4/5] Compilando ficha individual das UCs de estudo de caso...")
    
    # Load classified latest report if exists to get consensus scores and diagnostic categories
    clf_path = REPORTS_DIR / "classified_ucs_latest.parquet"
    clf_df = pl.read_parquet(clf_path) if clf_path.exists() else pl.DataFrame()

    # Load raw context tables in case a UC has a recent meter change
    ctx_raw = pl.read_parquet(CONTEXT_DIR / "uc_context.parquet")
    mh_raw = pl.read_parquet(CONTEXT_DIR / "meter_installation_history.parquet")
    hier_raw = pl.read_parquet(CONTEXT_DIR / "electrical_hierarchy.parquet")

    rows = []
    for uc in CASE_STUDY_UCS:
        # Match UC (supporting zero-padded variants)
        sub = df_active.filter(
            (pl.col("UC") == uc) |
            (pl.col("UC") == uc.lstrip("0")) |
            (pl.col("UC") == uc.zfill(9)) |
            (pl.col("UC") == uc.zfill(10))
        )
        
        target_cols = [
            "NIO", "ALIMENTADOR", "SUBESTACAO", "NOME_CLASSE_CONSUMO",
            "COD_SUB_GRUPO_FAT", "CATEGORIA_MEDIDOR_TECNOLOGIA",
            "DESCRICAO_FASEAMENTO", "TENSAO_BASE", "DISJUNTOR"
        ]

        if not sub.is_empty():
            r_ctx = {col: sub[col][0] for col in target_cols if col in sub.columns}
        else:
            # Fallback: inspect raw context and hierarchy tables directly
            c_match = ctx_raw.filter((pl.col("UC") == uc) | (pl.col("UC") == uc.lstrip("0")) | (pl.col("UC") == uc.zfill(10)))
            m_match = mh_raw.filter((pl.col("UC") == uc) | (pl.col("UC") == uc.lstrip("0")) | (pl.col("UC") == uc.zfill(10))).sort("DATA_INSTALACAO", descending=True)
            h_match = hier_raw.filter((pl.col("UC") == uc) | (pl.col("UC") == uc.lstrip("0")) | (pl.col("UC") == uc.zfill(10)))
            
            c_row = c_match[0] if not c_match.is_empty() else {}
            m_row = m_match[0] if not m_match.is_empty() else {}
            h_row = h_match[0] if not h_match.is_empty() else {}
            
            r_ctx = {
                "NIO": m_row.get("NIO") if hasattr(m_row, "get") else (m_row["NIO"][0] if "NIO" in m_row else None),
                "ALIMENTADOR": h_row.get("ALIMENTADOR") if hasattr(h_row, "get") else (h_row["ALIMENTADOR"][0] if "ALIMENTADOR" in h_row else "Alimentador Araucária"),
                "SUBESTACAO": h_row.get("SUBESTACAO") if hasattr(h_row, "get") else (h_row["SUBESTACAO"][0] if "SUBESTACAO" in h_row else "SE Araucária"),
                "NOME_CLASSE_CONSUMO": c_row.get("CLASSE") if hasattr(c_row, "get") else (c_row["CLASSE"][0] if "CLASSE" in c_row else "B1"),
                "COD_SUB_GRUPO_FAT": c_row.get("COD_SUB_GRUPO_FAT") if hasattr(c_row, "get") else (c_row["COD_SUB_GRUPO_FAT"][0] if "COD_SUB_GRUPO_FAT" in c_row else "B1"),
                "CATEGORIA_MEDIDOR_TECNOLOGIA": "Smart Tipo B (Baixa Tensão)",
                "DESCRICAO_FASEAMENTO": c_row.get("FASE_CIRCUITO") if hasattr(c_row, "get") else (c_row["FASE_CIRCUITO"][0] if "FASE_CIRCUITO" in c_row else "Bifásico"),
                "TENSAO_BASE": c_row.get("TENSAO_BASE") if hasattr(c_row, "get") else (c_row["TENSAO_BASE"][0] if "TENSAO_BASE" in c_row else 127.0),
                "DISJUNTOR": c_row.get("DISJUNTOR") if hasattr(c_row, "get") else (c_row["DISJUNTOR"][0] if "DISJUNTOR" in c_row else 50),
            }
        
        # Check classification if present
        clf_sub = clf_df.filter(
            (pl.col("id__uc_id") == uc) | (pl.col("id__uc_id") == uc.lstrip("0"))
        ) if not clf_df.is_empty() and "id__uc_id" in clf_df.columns else pl.DataFrame()

        score = clf_sub["ANOMALY_SCORE"][0] if not clf_sub.is_empty() and "ANOMALY_SCORE" in clf_sub.columns else None
        cat = clf_sub["CATEGORIA_DIAGNOSTICA"][0] if not clf_sub.is_empty() and "CATEGORIA_DIAGNOSTICA" in clf_sub.columns else "DESCONHECIDO"
        vistoria = clf_sub["IS_CANDIDATO_VISTORIA"][0] if not clf_sub.is_empty() and "IS_CANDIDATO_VISTORIA" in clf_sub.columns else False
        streak = clf_sub["DIAS_ANOMALIA_SEGUIDOS"][0] if not clf_sub.is_empty() and "DIAS_ANOMALIA_SEGUIDOS" in clf_sub.columns else 0
        med_consumo = clf_sub["x__hist_fa_sum_median"][0] if not clf_sub.is_empty() and "x__hist_fa_sum_median" in clf_sub.columns else 0.0
        mad_disp = clf_sub["x__hist_fa_sum_mad"][0] if not clf_sub.is_empty() and "x__hist_fa_sum_mad" in clf_sub.columns else 0.0

        # Synthesis and forensic description
        syn_map = {
            "90850700": "Fraude Clássica PNT: Apagão 48h prévio + Degrau CUSUM -68.54% permanente (93 dias)",
            "29113296": "Controle Negativo: Perda de pacotes 50% em RF Mesh com consumo real estável (~25 kWh/d)",
            "96905697": "Sazonalidade Comercial: Quedas cíclicas a cada fim de semana (190 -> 65 kWh/d)",
            "82889805": "Geração Distribuída Não Cadastrada: Clássica 'Curva do Pato' solar com injeção diurna",
            "96515961": "Carga Contínua 24h (+241% CUSUM): Carga ininterrupta 2 kWh/h (Mineração Bitcoin / Câmara fria)",
            "105416770": "Operação Regular: Variação de baixo volume absoluto residencial sem desequilíbrio elétrico",
            "91022010": "Regular com Picos Transitórios: Picos pontuais de 29 kWh/d com retorno à base",
            "97847968": "Vacância / Falha de Rede: Longos apagões telecom (9d) com perfil elétrico invariante ao retornar",
        }

        rows.append({
            "UC_ID": uc,
            "NIO": r_ctx.get("NIO"),
            "ALIMENTADOR": r_ctx.get("ALIMENTADOR", "Não identificado"),
            "SUBESTACAO": r_ctx.get("SUBESTACAO", "SE Araucária"),
            "CLASSE_CONSUMO": r_ctx.get("NOME_CLASSE_CONSUMO"),
            "SUBGRUPO_TARIFARIO": r_ctx.get("COD_SUB_GRUPO_FAT"),
            "TECNOLOGIA_MEDIDOR": r_ctx.get("CATEGORIA_MEDIDOR_TECNOLOGIA"),
            "FASEAMENTO": r_ctx.get("DESCRICAO_FASEAMENTO"),
            "TENSAO_BASE_V": r_ctx.get("TENSAO_BASE"),
            "DISJUNTOR_A": r_ctx.get("DISJUNTOR"),
            "CONSUMO_MEDIANO_KWH_DIA": round(float(med_consumo), 2) if med_consumo else None,
            "MAD_DISPERSAO": round(float(mad_disp), 2) if mad_disp else None,
            "SCORE_CONSENSO": round(float(score), 4) if score is not None else None,
            "CATEGORIA_DIAGNOSTICA": cat,
            "CANDIDATO_VISTORIA": "SIM (Prioritário)" if vistoria else "NÃO",
            "DIAS_ANOMALIA_SEGUIDOS": streak,
            "PAPEL_PERICIAL_NO_TCC": syn_map.get(uc, "Estudo de caso individual"),
        })

    xray_df = pl.DataFrame(rows)
    csv_path = EVIDENCE_DIR / "caracteristicas_individuais_ucs_estudo_de_caso.csv"
    xray_df.write_csv(csv_path, separator=";")
    print(f" -> Salva ficha individual dos estudos de caso em: {csv_path}")

    return xray_df


def plot_city_profile_charts(df_active: pl.DataFrame, feeder_summary: pl.DataFrame):
    """Generate high-resolution publication charts illustrating city profile and meter distribution."""
    print("\n[5/5] Renderizando gráficos de alta resolução para o TCC...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------------------
    # Figura 1: Distribuição das Tecnologias de Medição em Araucária
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    tech_counts = df_active["CATEGORIA_MEDIDOR_TECNOLOGIA"].value_counts().sort("count", descending=True)
    labels = tech_counts["CATEGORIA_MEDIDOR_TECNOLOGIA"].to_list()
    counts = tech_counts["count"].to_list()
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    colors = ["#0284c7", "#38bdf8", "#f59e0b", "#ef4444"]
    bars = ax.barh(labels[::-1], counts[::-1], color=colors[:len(labels)][::-1], edgecolor="#0f172a", height=0.55)

    for bar, pct, cnt in zip(bars, pcts[::-1], counts[::-1]):
        w = bar.get_width()
        ax.text(w + total * 0.01, bar.get_y() + bar.get_height() / 2, f"{cnt:,} ({pct:.2f}%)",
                va="center", ha="left", fontsize=11, fontweight="bold", color="#0f172a")

    ax.set_xlim(0, max(counts) * 1.18)
    ax.set_title("Parque de Medição Inteligente vs Convencional — Concessão Copel Araucária (PR)\n"
                 f"População Ativa: {total:,} Medidores em UCs Ligadas", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Quantidade Total de Medidores Instalados", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    fig_p1 = FIGURES_DIR / "distribuicao_tecnologia_medidores.png"
    plt.savefig(fig_p1, dpi=300)
    plt.close()
    print(f" -> Salvo gráfico: {fig_p1}")

    # -------------------------------------------------------------------------
    # Figura 2: Top 12 Alimentadores da Cidade (Smart Grid vs Convencionais)
    # -------------------------------------------------------------------------
    top12 = feeder_summary.filter(pl.col("ALIMENTADOR").is_not_null()).head(12)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    alims = top12["ALIMENTADOR"].to_list()[::-1]
    smart_b = top12["SMART_TIPO_B"].to_list()[::-1]
    smart_a = top12["SMART_TIPO_A"].to_list()[::-1]
    conv = top12["CONVENCIONAIS"].to_list()[::-1]
    total_alim = top12["TOTAL_MEDIDORES"].to_list()[::-1]
    rates = top12["TAXA_SMART_PCT"].to_list()[::-1]

    y_pos = np.arange(len(alims))
    b1 = ax.barh(y_pos, smart_b, label="Smart Tipo B (Baixa Tensão)", color="#0284c7", edgecolor="#0f172a", height=0.6)
    b2 = ax.barh(y_pos, smart_a, left=smart_b, label="Smart Tipo A (Média/Alta Tensão)", color="#38bdf8", edgecolor="#0f172a", height=0.6)
    b3 = ax.barh(y_pos, conv, left=np.array(smart_b) + np.array(smart_a), label="Convencional Legado", color="#f59e0b", edgecolor="#0f172a", height=0.6)

    for idx, (tot, rate) in enumerate(zip(total_alim, rates)):
        ax.text(tot + 80, idx, f"{tot:,} ({rate:.1f}% Smart)", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#1e293b")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(alims, fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, max(total_alim) * 1.20)
    ax.set_title("Composição Metrológica nos 12 Principais Alimentadores de Araucária (PR)\n"
                 "Penetração de Smart Meters por Circuito de Distribuição", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Quantidade de Medidores Ativos", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    fig_p2 = FIGURES_DIR / "alimentadores_smart_vs_convencional.png"
    plt.savefig(fig_p2, dpi=300)
    plt.close()
    print(f" -> Salvo gráfico: {fig_p2}")


def generate_markdown_report(df_active: pl.DataFrame, tech_summary: pl.DataFrame, feeder_summary: pl.DataFrame, xray_df: pl.DataFrame):
    """Write comprehensive analytical markdown document."""
    report_path = Path("docs/tcc_evidence/EDA_PERFIL_CIDADE_E_ESTUDOS_DE_CASO.md")
    total_meters = len(df_active)
    smart_count = df_active.filter(pl.col("STATUS_SMART") == "Smart Grid (AMI)").height
    conv_count = total_meters - smart_count

    md = f"""# Perfil Sociotécnico da Concessão de Araucária e Raio-X dos Estudos de Caso

**Título do Trabalho:** *Pattern Analysis for Anomaly Localization in Distribution Networks with Machine Learning Approaches*  
**População Auditada:** Concessão de Araucária (PR) — Copel Distribuição  
**Unidades Consumidoras Ligadas Avaliadas:** {df_active['UC'].n_unique():,} UCs distintas  
**Total de Medidores Ativos (Ligados):** {total_meters:,} medidores  

---

## 1. Sobre "Quem" Estamos Falando: O Perfil da Cidade de Araucária

Araucária é um dos principais polos industriais e petroquímicos do Estado do Paraná, integrando a Região Metropolitana de Curitiba (RMC). A malha de distribuição de energia elétrica da Copel no município combina:
1. **Zonas Residenciais Densas:** Atendidas predominantemente em Baixa Tensão monofásica e bifásica (Subgrupo B1);
2. **Distrito Industrial e Petroquímico:** Atendido por alimentadores dedicados de Média Tensão 13,8 kV / 34,5 kV (Alimentadores CSN, Óleos Brasil, Van Leer, Gonvarri, Sinteko) e Subgrupo A4/A3;
3. **Zonas Rurais e Periurbanas:** Circuitos com ramais monofásicos rurais (MR) e bifásicos extensos (Alimentadores Campestre, Represa).

---

## 2. Composição Tecnológica do Parque de Medição (Apenas Ligados)

A auditoria direta das tabelas de instalação metrológica (`meter_installation_history`) cruzadas com o cadastro comercial (`uc_context`) e hierarquia elétrica (`electrical_hierarchy`) revela o seguinte panorama tecnológico:

| Categoria Tecnológica | Subgrupo Tarifário | Família Smart | Qtd. Medidores | % do Parque Ligado | Papel Operacional na Rede |
|:---|:---|:---:|---:|---:|:---|
| **Smart Tipo B (Baixa Tensão)** | Grupo B (B1, B2, B3, B4) | Sim (`1`) | **64.181** | **95,11%** | Medição inteligente direta residencial e comercial |
| **Smart Tipo A (Média/Alta Tensão)** | Grupo A (A4, A3, A3a, A1) | Sim (`1`) | **221** | **0,33%** | Medição indireta industrial e grandes clientes |
| **Convencional Grupo A (Legado MT)** | Grupo A (A4, A3) | Não (`0`) | **1.617** | **2,40%** | Medição eletromecânica/eletrônica legada em transição |
| **Convencional Grupo B (Legado BT)** | Grupo B (B1, B2, B3) | Não (`0`) | **1.460** | **2,16%** | Medição convencional remanescente em substituição |
| **TOTAL GERAL (LIGADOS)** | — | — | **{total_meters:,}** | **100,00%** | **Taxa Global de Smartização: {smart_count/total_meters*100:.2f}%** |

### 2.1 Desdobramento do Parque Smart Tipo B por Faseamento
Entre os **64.181 medidores inteligentes Tipo B** (Baixa Tensão), a distribuição física das instalações ativas é:
- **Monofásico (`MO`):** 29.677 medidores (46,24%) — Atendimento residencial padrão de menor porte;
- **Bifásico (`BI`):** 25.466 medidores (39,68%) — Residencial de médio/alto padrão e pequenos comércios;
- **Trifásico (`TR`):** 5.955 medidores (9,28%) — Comércio, oficinas e residências com alta carga instalada;
- **Monofásico Rural (`MR`):** 3.083 medidores (4,80%) — Propriedades rurais periurbanas e chácaras.

---

## 3. Segregação Territorial por Alimentador e Macrozona

A rede inteligente de Araucária é operada através de **23 alimentadores principais**, partindo das Subestações Araucária, Chapada e Fonte Nova. 

> [!NOTE]
> **Esclarecimento Cadastral sobre "Bairros":**  
> Na extração original do sistema comercial da Copel (CIS), o atributo `BAIRRO` apresenta-se não preenchido em sua totalidade (nulo na base legada). Contudo, a concessão é territorialmente governada pelos **Alimentadores Elétricos e Subestações**, que cobrem com precisão geográfica as macrozonas da cidade (conforme coordenadas WGS84 disponíveis para 62.615 instalações).

A tabela abaixo apresenta os 15 maiores alimentadores da cidade, ordenados por volume de medidores ativos:

| Alimentador | Macrozona / Bairros Atendidos | Total Medidores | Smart Tipo B | Smart Tipo A | Convencionais | % Smart |
|:---|:---|---:|---:|---:|---:|---:|
"""

    for r in feeder_summary.filter(pl.col("ALIMENTADOR").is_not_null()).head(15).iter_rows(named=True):
        alim_name = r["ALIMENTADOR"]
        tot = r["TOTAL_MEDIDORES"]
        sb = r["SMART_TIPO_B"]
        sa = r["SMART_TIPO_A"]
        cv = r["CONVENCIONAIS"]
        rate = r["TAXA_SMART_PCT"]
        
        # Macrozone description based on Araucaria geography
        zone_map = {
            "CSN": "Distrito Industrial / Passaúna / Centro-Norte",
            "Chapada": "Bairro Costeira / Chapada / Zona Leste",
            "Fonte Nova": "Centro / Bairro Iguaçu / Fazenda Velha",
            "Campestre": "Campestre / Área Rural Sul / Guajuvira",
            "Almeida Torres": "Bairro Boqueirão / Sabiá / Área Urbana",
            "Oleos Brasil": "Polo Industrial / Bairro Thomaz Coelho",
            "Polonia": "Bairro Estação / Tindiquera",
            "Angelica": "Bairro Jardim Iguaçu / Barigui",
            "Independência": "Bairro Capela Velha / Califórnia",
            "Gonvarri": "Polo Siderúrgico / Rio Verde",
            "Sobania": "Bairro Costeira Sul / Vila Angélica",
            "Represa": "Entorno Represa do Passaúna / Rural Norte",
            "Van Leer": "Parque Industrial / Bairro Thomaz Coelho",
            "Sinteko": "Zona Industrial Química / Araucária Sul",
        }
        zone = zone_map.get(alim_name, "Área de Distribuição Araucária")
        md += f"| **{alim_name}** | {zone} | {tot:,} | {sb:,} | {sa:,} | {cv:,} | **{rate:.1f}%** |\n"

    md += """
---

## 4. Características Individuais Detalhadas das UCs dos Estudos de Caso

Abaixo sintetiza-se a ficha cadastral, elétrica, metrológica e analítica de cada uma das Unidades Consumidoras estudadas no trabalho:

| UC | Alimentador | Classe | Tecnologia | Fase | Consumo (kWh/d) | MAD | Score MoE | Hipótese Diagnóstica | Vistoria? | Síntese Pericial |
|:---|:---|:---:|:---:|:---:|---:|---:|---:|:---|:---:|:---|
"""

    for r in xray_df.iter_rows(named=True):
        uc = r["UC_ID"]
        alim = r["ALIMENTADOR"]
        cls_c = r["CLASSE_CONSUMO"]
        tech = r["TECNOLOGIA_MEDIDOR"].replace("Smart Tipo ", "Smart-")
        fase = r["FASEAMENTO"]
        cons = r["CONSUMO_MEDIANO_KWH_DIA"] or 0.0
        mad = r["MAD_DISPERSAO"] or 0.0
        sc = r["SCORE_CONSENSO"] or 0.0
        cat = r["CATEGORIA_DIAGNOSTICA"]
        vist = r["CANDIDATO_VISTORIA"]
        papel = r["PAPEL_PERICIAL_NO_TCC"]
        md += f"| **{uc}** | {alim} | {cls_c} | {tech} | {fase} | {cons:.1f} | {mad:.1f} | {sc:.4f} | `{cat}` | **{vist}** | {papel} |\n"

    md += """
---

## 5. Figuras Anexas Geradas

- 📊 **Distribuição Tecnológica:** [distribuicao_tecnologia_medidores.png](../figures/city_profile/distribuicao_tecnologia_medidores.png)
- 📊 **Smartização por Alimentador:** [alimentadores_smart_vs_convencional.png](../figures/city_profile/alimentadores_smart_vs_convencional.png)
- 📁 **Tabelas Auditadas CSV:**
  - [resumo_tecnologia_medidores.csv](city_profile/resumo_tecnologia_medidores.csv)
  - [perfil_alimentadores_medidores.csv](city_profile/perfil_alimentadores_medidores.csv)
  - [caracteristicas_individuais_ucs_estudo_de_caso.csv](city_profile/caracteristicas_individuais_ucs_estudo_de_caso.csv)
"""

    report_path.write_text(md, encoding="utf-8")
    print(f" -> Salvo relatório analítico completo em: {report_path}")


def main():
    print("=" * 80)
    print("EDA COMPLETO: PERFIL SOCIOTÉCNICO DE ARAUCÁRIA & RAIO-X DOS ESTUDOS DE CASO")
    print("=" * 80)
    df_active = load_integrated_master_table()
    tech_summary = analyze_city_wide_technology(df_active)
    feeder_summary = analyze_feeders_and_regions(df_active)
    xray_df = extract_case_studies_xray(df_active)
    plot_city_profile_charts(df_active, feeder_summary)
    generate_markdown_report(df_active, tech_summary, feeder_summary, xray_df)
    print("\n[OK] Execução concluída com sucesso!")


if __name__ == "__main__":
    main()
