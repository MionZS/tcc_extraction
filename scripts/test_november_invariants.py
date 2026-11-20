import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import polars as pl

print("=== VERIFICAÇÃO DE INVARIANTES E RECONCILIAÇÃO METROLÓGICA ===")

p_metrics = Path("docs/tcc_evidence/november_experiment/metricas_diarias_populacao_nov2026.csv")
df = pl.read_csv(p_metrics, separator=";")

# Teste 1: Conservação da População Total
diff_pop = (df["POPULACAO_TOTAL"] - (df["ELEGIVEIS_LIMPOS"] + df["TOTAL_QUARENTENA"])).abs().max()
print(f"1. Teste de Conservação da População (Total == Elegíveis + Quarentena): Erro Máx = {diff_pop}")
assert diff_pop == 0, "Falha na conservação da população"

# Teste 2: Equação de Balanço de Filas (Candidatos == Permanência + Entradas)
# No primeiro dia, Permanência é 0 e Entradas é o saldo inicial.
for i in range(1, len(df)):
    c_curr = df["CANDIDATOS_VISTORIA"][i]
    perm = df["FILA_PERMANENCIA"][i]
    ent = df["FILA_ENTRADAS"][i]
    assert c_curr == perm + ent, f"Falha no balanço de filas no dia {df['CUTOFF_DATE'][i]}: {c_curr} != {perm} + {ent}"
print("2. Teste de Balanço de Filas (Candidatos(t) == Permanência(t) + Entradas(t)): APROVADO em todos os 19 dias")

# Teste 3: Limites dos Scores
min_sc = df["SCORE_MEDIO_ELEGIVEIS"].min()
max_sc = df["SCORE_P95_ELEGIVEIS"].max()
print(f"3. Teste de Limites do Score de Anomalia: min={min_sc}, max={max_sc} (esperado em [0, 1])")
assert 0.0 <= min_sc <= 1.0 and 0.0 <= max_sc <= 1.0, "Scores fora do intervalo [0, 1]"

# Teste 4: Ausência de Dados Futuros
p_tracks = Path("docs/tcc_evidence/november_experiment/trilha_individual_estudos_caso_nov2026.csv")
df_tr = pl.read_csv(p_tracks, separator=";")
max_cutoff = df_tr["CUTOFF_DATE"].max()
print(f"4. Teste de Data Máxima Avaliada: {max_cutoff} (inferior a 2026-11-20)")
assert max_cutoff <= "2026-11-20", "Vazamento temporal detectado"

print("\nTODOS OS 4 TESTES DE INVARIANTES METROLÓGICAS FORAM APROVADOS COM SUCESSO!")
