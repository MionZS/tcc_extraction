# O Dilema da Série Temporal: Associação UC × NIO ao Longo do Tempo

> Rascunho para inclusão no artigo. Estrutura CAR (Contexto, Ação, Resultado).

## 1. Contexto

Em redes AMI, duas entidades são frequentemente confundidas:

| Entidade | Natureza | Ciclo de vida |
|---|---|---|
| **UC** (Unidade Consumidora) | Ponto de entrega contratual/elétrico | Perene; pode ser criada, suspensa ou encerrada |
| **NIO** (número do medidor) | Ativo físico de medição | Substituível; emite a telemetria |

O MDM (ORCA) indexa as medições por **NIO** (`meter_asset_no`), nunca por UC. O objeto de estudo
(anomalias, desbalanceamento, perdas) é, porém, a **UC**. A série temporal de uma UC é portanto
uma *composição* das séries dos NIOs que a atenderam, e essa composição muda no tempo.

### Eventos que quebram a hipótese "1 UC = 1 NIO = 1 série"

1. **Upgrade convencional → inteligente.** Antes da troca a UC não possui telemetria; depois, passa a
   possuir. A UC "entra" na análise em um instante específico (entrada censurada à esquerda).
2. **Troca inteligente → inteligente.** A UC é a mesma, mas a fonte dos dados muda de NIO. Os contadores
   acumulados (registradores de energia) são reiniciados ou assumem outro offset.
3. **Inclusão/encerramento de UC.** A população muda diariamente.
4. **Troca intradia.** No dia da troca existem *dois* NIOs válidos, cada um cobrindo um trecho do dia.

## 2. O problema

### 2.1 Viés de sobrevivência (snapshot cadastral)
Extrair o CIS "como está hoje" e consultar o MDM do passado apenas com os NIOs vigentes hoje
descarta, para datas anteriores à troca, os medidores então instalados. A UC aparece sem dados
(falso "apagão de telemetria") ou com uma série truncada.

### 2.2 Descontinuidades espúrias
Sem tratar a troca, a série de uma UC exibe:
- queda a zero (saída do NIO antigo) seguida de reaparecimento (NIO novo);
- *reset* de registradores acumulados (energia ativa/reativa), que um modelo interpretaria como
  inversão de fluxo ou geração distribuída;
- mudança de cadência (por exemplo, 15 min para 5 min) quando muda a família do medidor.

Todas podem ser erroneamente rotuladas como anomalia.

### 2.3 Vazamento temporal (leakage)
Atributos cadastrais atuais da UC (classe, fase, tensão, situação) aplicados a datas passadas
introduzem informação do futuro nos *features*. O cadastro é uma dimensão de variação lenta (SCD),
não uma constante.

## 3. Ação

### 3.1 Regra de vigência inclusiva
Cada vínculo é descrito por `(UC, NIO, DATA_INSTALACAO, DATA_RETIRADA)`. Para um dia de referência `D`,
o NIO é consultado se:

```
DATA_INSTALACAO <= D  E  (DATA_RETIRADA é NULL  OU  DATA_RETIRADA >= D)
```

O operador `>=` na retirada é deliberado: **no dia da troca ambos os NIOs são extraídos**, pois o
antigo cobre o início do dia e o novo o restante.

### 3.2 Granularidade correta
- **Telemetria** (`ami_interval`, `ami_instantaneous`, `ami_registers`): grão **NIO × tempo**. A medição
  pertence ao aparelho.
- **Vínculo** (`meter_installation_history`): grão **UC × NIO × [instalação, retirada]**.
- **Série da UC**: obtida por *junção temporal* (as-of join) entre telemetria e vínculo, não por chave
  estática.

### 3.3 Implementação
- A consulta ao CIS passou a incluir medidores retirados dentro do horizonte de estudo
  (`dta_reti_reu IS NULL OR dta_reti_reu >= DATE '2026-01-01'`).
- A população é única por `(UC, NIO)` (antes, por NIO).
- O *loop* diário de extração filtra a população pela regra de vigência antes de montar os lotes.

## 4. Resultado

- Na carga de 2026-08-10, a fila diária contém 70.869 medidores vigentes dentre 72.473 vínculos
  conhecidos: os demais ainda não haviam sido instalados ou já haviam sido retirados naquela data.
- A telemetria deixa de ser truncada nas datas de troca e a população diária passa a refletir a rede real.

## 5. Questões em aberto (a discutir no artigo)

1. **Fusão intradia.** Somar energia de dois NIOs no dia da troca é trivial; para tensão e corrente
   é preciso definir a política (concatenar por *timestamp* e marcar a origem).
2. **Registradores acumulados.** Calcular deltas *por NIO* e só depois agregar à UC; nunca diferenciar
   a série já fundida.
3. **Hora da troca.** O cadastro registra apenas a data. O instante pode ser inferido do primeiro/último
   pacote de cada NIO no MDM. Isso evita sobreposição e lacuna.
4. **Cadastro histórico.** Atributos da UC devem ser versionados (SCD tipo 2) ou *snapshots* diários
   do CIS devem ser guardados. Hoje o cadastro é lido com o estado atual.
5. **Censura.** UCs que entram com medidor inteligente no meio do período têm histórico curto.
   Estratégias: janela mínima de observação, *flag* de "tempo desde a instalação" como feature, ou
   exclusão dos primeiros dias pós-troca (período de comissionamento).
6. **Medidor não inteligente.** Não gera telemetria; deve ser filtrado antes da consulta ao MDM e
   tratado como ausência estrutural, não como falha.
7. **Reuso de NIO.** Um NIO pode ser retirado e reinstalado em outra UC. A chave de vínculo deve
   ser sempre `(UC, NIO, intervalo)`, nunca `NIO` isolado.

## 6. Formulação

Seja `X_u(t)` a série da UC `u` e `M_u(t)` o NIO vigente em `t`. Então

```
X_u(t) = Σ_k  1[ t ∈ [ins_k, ret_k) ] · Y_k(t)
```

onde `Y_k` é a série bruta do NIO `k` vinculado a `u`. A construção ingênua usa `k = M_u(hoje)`
para todo `t`, o que só é correto se nunca houve troca.
