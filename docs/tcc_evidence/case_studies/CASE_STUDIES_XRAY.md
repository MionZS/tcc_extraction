# Estudos de Caso Reais e Raio-X da UC (Anonimizados)

## Caso 1: Operação Regular (`753***95`)
- **Alimentador:** Fonte Nova | **Cutoff:** 2026-11-04 | **Cobertura:** 98.6% | **NIOs mapeados:** 2
- **Métricas:** Mediana=5.277 kWh/dia | MAD=0.748 | CV=0.1391 | Slope=0.004 | Desb.I=0.0 | Reversão RA=0.0
- **Modelo:** Score=0.0012 | Streak=0d | Hipótese=`REGULAR` | Candidato=False
- **Ressalva:** Hipótese diagnóstica automática baseada em telemetria; não constitui confirmação de irregularidade sem inspeção física.

## Caso 2: Anomalia Multivariada Global (`670***50`)
- **Alimentador:** Fonte Nova | **Cutoff:** 2026-11-04 | **Cobertura:** 96.6% | **NIOs mapeados:** 10
- **Métricas:** Mediana=139.592 kWh/dia | MAD=2.818 | CV=0.0202 | Slope=-0.4916 | Desb.I=0.1359 | Reversão RA=0.0
- **Modelo:** Score=0.5784 | Streak=7d | Hipótese=`ANOMALIA_MULTIVARIADA` | Candidato=True
- **Ressalva:** Hipótese diagnóstica automática baseada em telemetria; não constitui confirmação de irregularidade sem inspeção física.

## Caso 3: Desequilíbrio / Possível Bypass (`114****04`)
- **Alimentador:** None | **Cutoff:** 2026-08-31 | **Cobertura:** 60.2% | **NIOs mapeados:** 4
- **Métricas:** Mediana=216.936 kWh/dia | MAD=20.165 | CV=0.0929 | Slope=-8.526 | Desb.I=1.0 | Reversão RA=0.0
- **Modelo:** Score=0.6518 | Streak=7d | Hipótese=`DESEQUILIBRIO_CARGA_OU_BYPASS` | Candidato=True
- **Ressalva:** Hipótese diagnóstica automática baseada em telemetria; não constitui confirmação de irregularidade sem inspeção física.

## Caso 4: Fraude Clássica PNT com Apagão Precursor de Telecom (`908***00` / UC 90850700)
- **Alimentador:** Oleos Brasil | **Cutoff:** 2026-11-04 | **Cobertura Diária:** 91.1% | **NIOs mapeados:** 2
- **Métricas:** Mediana=17.160 kWh/dia | MAD=4.576 | CV=0.2651 | Slope=-0.1961 kWh/dia | Desb.I=1.4035 | Reversão RA=0.4298
- **Modelo:** Score=0.5412 | Streak=**93 dias seguidos** | Hipótese=`POSSIVEL_GD_NAO_CADASTRADA` / `ANOMALIA_ELETRICA` | **Candidato=SIM (Prioritário)**
- **Rastro Pericial AMI:** Apagão absoluto de 48h na telemetria (28/10 21h a 30/10 21h com cobertura 0%), seguido imediatamente de degrau descendente CUSUM de **-68.54%** em 31/10/2026, onde o consumo desaba de 17.16 para 5.10 kWh/dia e permanece deprimido indefinidamente. Padrão físico compatível com desenergização do padrão para implantação de by-pass clandestino / shunt de corrente em 2 fases.
- **Documentação Aprofundada:** Ver estudo pericial completo em [ESTUDO_DE_CASO_FRAUDE_VS_TELECOM.md](../ESTUDO_DE_CASO_FRAUDE_VS_TELECOM.md).
- **Ressalva:** Hipótese diagnóstica automática baseada em telemetria; priorização de campo fundamentada em evidência física e estatística.

## Caso 5: Degrau Descendente / Queda Sustentada (`216***98`)
- **Alimentador:** Van Leer | **Cutoff:** 2026-11-04 | **Cobertura:** 60.7% | **NIOs mapeados:** 2
- **Métricas:** Mediana=365.1 kWh/dia | MAD=24.61 | CV=0.0674 | Slope=-1.8687 | Desb.I=0.3503 | Reversão RA=0.0
- **Modelo:** Score=0.6182 | Streak=1d | Hipótese=`DEGRAU_DESCENDENTE_CONSUMO` | Candidato=True
- **Ressalva:** Hipótese diagnóstica automática baseada em telemetria; não constitui confirmação de irregularidade sem inspeção física.

## Caso 6: Medidor Travado / Consumo Zero (`721***08`)
- **Alimentador:** Gonvarri | **Cutoff:** 2026-11-04 | **Cobertura:** 60.6% | **NIOs mapeados:** 2
- **Métricas:** Mediana=0.0 kWh/dia | MAD=0.0 | CV=0.0 | Slope=0.0 | Desb.I=0.0 | Reversão RA=0.0
- **Modelo:** Score=0.5436 | Streak=3d | Hipótese=`MEDIDOR_TRAVADO_OU_CONSUMO_ZERO` | Candidato=True
- **Ressalva:** Hipótese diagnóstica automática baseada em telemetria; não constitui confirmação de irregularidade sem inspeção física.

## Caso 7: Falha / Quarentena de Telecomunicação (`255***30`)
- **Alimentador:** Campestre | **Cutoff:** 2026-11-04 | **Cobertura:** 0.0% | **NIOs mapeados:** 2
- **Métricas:** Mediana=0.0 kWh/dia | MAD=0.0 | CV=0.0 | Slope=0.0 | Desb.I=0.0 | Reversão RA=0.0
- **Modelo:** Score=0.3058 | Streak=0d | Hipótese=`TELECOM_QUARANTINE` | Candidato=False
- **Ressalva:** Hipótese diagnóstica automática baseada em telemetria; não constitui confirmação de irregularidade sem inspeção física.

## Caso 8: Alta Dispersão com Consumo Médio Normal (116781157) (`116****57`)
- **Alimentador:** CSN | **Cutoff:** 2026-11-04 | **Cobertura:** 99.4% | **NIOs mapeados:** 2
- **Métricas:** Mediana=19.224 kWh/dia | MAD=16.564 | CV=0.8572 | Slope=1.1269 | Desb.I=1.0 | Reversão RA=0.0
- **Modelo:** Score=0.4963 | Streak=0d | Hipótese=`REGULAR` | Candidato=False
- **Ressalva:** Hipótese diagnóstica automática baseada em telemetria; não constitui confirmação de irregularidade sem inspeção física.

## Caso 9: O Falso Positivo Evitado via Gating de Telecom (`291***96` / UC 29113296)
- **Alimentador:** Araucária | **Cutoff:** 2026-11-04 | **Cobertura Horária:** 50.0% (168 janelas degradadas) | **NIOs mapeados:** 2
- **Métricas:** Mediana=25.453 kWh/dia | MAD=10.528 | CV=0.4136 | Slope=-0.6712 kWh/dia | Desb.I=0.0 | Reversão RA=0.0
- **Modelo:** Score=0.6553 | Streak=24d | Hipótese=`TELECOM_QUARANTINE` | **Candidato=NÃO (Despacho Bloqueado)**
- **Rastro Pericial AMI:** Consumo real estável e regular em 30d diário (~25.4 kWh/dia com totalizador acumulando normalmente). Em nível horário, a malha RF mesh perde 50% dos pacotes, gerando aparente lacuna. O Gating de Quarentena evita o despacho incorreto de equipe de campo, economizando custos de viatura e prevenindo atrito com cliente adimplente.
- **Documentação Aprofundada:** Ver estudo pericial completo em [ESTUDO_DE_CASO_FRAUDE_VS_TELECOM.md](../ESTUDO_DE_CASO_FRAUDE_VS_TELECOM.md).

## Caso 10: Perfil Típico Sazonal de Fim de Semana — Contraste com Fraude (`969***97` / UC 96905697)
- **Alimentador:** Fonte Nova | **Cutoff:** 2026-11-04 | **Cobertura Diária:** 99.6% | **NIOs mapeados:** 2
- **Métricas:** Mediana=175.420 kWh/dia | MAD=12.544 | Slope=-0.8713 kWh/dia | Desb.I=1.0 | Reversão RA=0.0
- **Modelo:** Score=0.5589 | Streak=67d | Hipótese=`DESEQUILIBRIO_CARGA_OU_BYPASS` | **Candidato=SIM (Prioritário)**
- **Rastro Pericial AMI:** Perfil comercial/industrial de alta carga. Apresenta quedas periódicas severas rigorosamente a cada 7 dias (sábados e domingos o consumo cai de ~190 para ~65 kWh/dia). Na série horária, 5 dias úteis de atividade plena (~16-20 kWh/h) e 2 dias de repouso (~2-3 kWh/h). O detector CUSUM atesta: `Sem degrau CUSUM crítico`, pois a oscilação é estritamente sazonal de fim de semana (ao contrário do degrau irreversível da UC 90850700).

## Caso 11: Possível Microgeração Distribuída (GD) Não Cadastrada (`828***05` / UC 82889805)
- **Alimentador:** Araucária | **Cutoff:** 2026-11-04 | **Cobertura Horária:** 49.9% (Cadência 10 min) | **NIOs mapeados:** 2
- **Métricas:** Mediana Horária=1.8245 kWh/h | MAD=0.6085 | Slope=+0.4415 kWh/dia
- **Modelo:** Score=0.5761 | Streak=71 dias seguidos | Hipótese=`POSSIVEL_GD_NAO_CADASTRADA` | **Candidato=SIM (Prioritário)**
- **Rastro Pericial AMI:** Caso arquetípico de injeção solar/microgeração não cadastrada na distribuidora. Apresenta nítido perfil de "Curva do Pato" (Duck Curve): depressão profunda de consumo nos horários de máxima irradiância solar (11h às 15h) e picos de demanda no início da noite (19h às 21h).

## Caso 12: Salto Repentino (+241%) com Carga Contínua 24h sem desligamento (`965***61` / UC 96515961)
- **Alimentador:** Araucária | **Cutoff:** 2026-11-04 | **Cobertura Diária:** 98.4% | **NIOs mapeados:** 2
- **Métricas:** Mediana=22.044 kWh/dia | MAD=11.414 | Slope=+1.566 kWh/dia
- **Modelo:** Score=0.5509 | Streak=55 dias | Hipótese=`DESEQUILIBRIO_CARGA_OU_BYPASS` | **Candidato=SIM (Prioritário)**
- **Rastro Pericial AMI:** Em 21/10/2026, sofre um **degrau CUSUM abrupto de +241.08%**, saltando de ~12 kWh/dia para ~55 kWh/dia. Na telemetria horária de 14 dias, exibe consumo linear e ininterrupto de base em torno de ~1.5 a 2.5 kWh/h durante as 24 horas do dia, sem qualquer redução no período noturno (ausência total de fundo de vale). Assinatura compatível com carga computacional ininterrupta (mineração de Bitcoin / servidores 24/7), câmara fria/refrigeração comercial com termostato travado, ou maquinário operando ininterruptamente sob tarifa residencial.

## Caso 13: Por que a UC 105416770 é Categorizada como Regular (`105***70` / UC 105416770)
- **Alimentador:** Araucária | **Cutoff:** 2026-11-04 | **Cobertura Diária:** 98.9% | **NIOs mapeados:** 2
- **Métricas:** Mediana=7.609 kWh/dia | MAD=3.359 | Slope=-0.1503 kWh/dia
- **Modelo:** Score=0.5327 | Streak=29 dias | Hipótese=`REGULAR` | **Candidato=NÃO**
- **Fundamentação Metodológica:** Embora registre degrau CUSUM de -63.21% em 30/10, o volume absoluto prévio é baixo (mediana base de apenas 7.6 kWh/dia). Na curva horária, o consumo resume-se a picos estreitos de chuveiro/eletrodoméstico (~3 kWh/h) em residência humilde. A redução no final do mês coincide com viagem de feriado prolongado ou economia doméstica voluntária. Sem desequilíbrio de corrente ($I_{imb} \approx 0$) e sem reversão ativa ($RA = 0$), o Ensemble reconhece que variações percentuais em baixo volume são ruído de estilo de vida humano e não fraude, desarmando o alarme falso com segurança.

## Caso 14: Regularidade Estrutural com Picos Transitórios (`910***10` / UC 91022010)
- **Alimentador:** Araucária | **Cutoff:** 2026-11-04 | **Cobertura Diária:** 99.1% | **NIOs mapeados:** 2
- **Métricas:** Mediana=9.420 kWh/dia | MAD=1.931 | Slope=-0.1305 kWh/dia
- **Modelo:** Score=0.5324 | Streak=0 dias | Hipótese=`REGULAR` | **Candidato=NÃO**
- **Rastro Pericial AMI:** Apresenta consumo regular de 9.4 kWh/dia com pico anômalo transitório em 10/10 a 12/10 (atingindo 29 kWh/dia). O modelo classifica corretamente como regular porque o consumo retorna espontaneamente ao baseline mediano em seguida, sem quebra de regime permanente. A nova funcionalidade de deslizamento temporal (`◀ -7d` / `+7d ▶`) e visão `30d Horário` permite navegar e inspecionar exatamente o intervalo dos picos em alta resolução.
