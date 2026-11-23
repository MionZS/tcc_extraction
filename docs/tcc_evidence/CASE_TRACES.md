# Trilhas Longitudinais e Reconciliação dos Estudos de Caso (Mascarados)

## 1. Matriz de Diferenças (Anterior × Recálculo × Interpretação Corrigida)
| UC Mascarada | Afirmação Anterior | Recálculo Verificado | Interpretação Corrigida |
|:---|:---|:---|:---|
| `908***00` | 93 dias consecutivos; outro relatório registrava REGULAR ou INICIACAO_PERFIL_UC_NOVA | Ago=17.41 kWh/d; Out->Nov queda FA 18.05->5.72 kWh/d (-68.3%) + início de fluxo reverso RA=14–31 kWh/d em 12 dias (OOT score=0.9601); DATA_INSTALACAO CIS=2026-10-31 no mesmo NIO é atualização cadastral administrativa (não UC nova); persistência real=20d (nunca 93d); categoria corrigida: POSSIVEL_GD_NAO_CADASTRADA | Entrada efetiva de microgeração bidirecional após 31/10 (12 dias de injeção reversa RA ativa), anteriormente mascarada como INICIACAO_PERFIL_UC_NOVA por atualização cadastral administrativa, agora corretamente restabelecida como POSSIVEL_GD_NAO_CADASTRADA e candidata à vistoria. |
| `291***96` | 50% de perda de pacotes crônica em RF Mesh com reconfiguração de firmware | Cadência no arquivo diário não persistida (CADENCE_MINUTES: None); consumo diário estável em 28 kWh/dia com 100% integridade no registrador acumulador | Completude horária de 50% reflete intervalo de 10 min frente à grade de 5 min; sem evidência causal de reconfiguração de firmware; consumo e registrador sem anomalia elétrica |
| `969***97` | Desequilíbrio/bypass por quedas cíclicas fim de semana | Ratio dia útil/fim de semana = 1.45x | Sazonalidade comercial regular de funcionamento (não anomalia de furto) |
| `828***05` | Geração distribuída solar não cadastrada | Fluxo reverso = 56.0% do fluxo total em 30d | Hipótese diagnóstica de microgeração solar fotovoltaica não homologada |

## 2. Features Incrementais (Seção 9) até 2026-11-19
| UC Mascarada | Medidores Hist. | Cadência Bruta | Materialidade 30d (kWh) | Gap Máx (d) | Delta Degrau (kWh/d) | Dias Pós-Mudança | Razão Útil/FDS | Razão Reversa |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|
| `908***00` | 1 | 10.0 min | 217.97 | 2 | -12.515 | 20 | 1.1271 | 0.5208 |
| `291***96` | 1 | None min | 28.0 | 15 | 0.0 | 0 | 1.0 | 0.5381 |
| `969***97` | 1 | 10.0 min | 4658.26 | 0 | 0.0 | 0 | 1.4536 | 0.0 |
| `828***05` | 1 | 10.0 min | 1920.95 | 0 | 0.0 | 0 | 3.1047 | 0.5603 |
| `965***61` | 1 | 10.0 min | 1180.61 | 0 | -30.675 | 6 | 1.1285 | 0.0 |
| `105****70` | 1 | 10.0 min | 253.11 | 0 | -4.522 | 21 | 0.9351 | 0.4788 |
| `910***10` | 1 | None min | 144.25 | 10 | -4.376 | 25 | 0.8864 | 0.6026 |
| `978***68` | 2 | None min | 601.31 | 6 | 0.0 | 0 | 1.2539 | 0.5368 |
