# Perfil Sociotécnico da Concessão de Araucária e Raio-X dos Estudos de Caso

**Título do Trabalho:** *Pattern Analysis for Anomaly Localization in Distribution Networks with Machine Learning Approaches*  
**População Auditada:** Concessão de Araucária (PR) — Copel Distribuição  
**Unidades Consumidoras Ligadas Avaliadas:** 65,762 UCs distintas  
**Total de Medidores Ativos (Ligados):** 67,479 medidores  

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
| **TOTAL GERAL (LIGADOS)** | — | — | **67,479** | **100,00%** | **Taxa Global de Smartização: 95.44%** |

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
| **CSN** | Distrito Industrial / Passaúna / Centro-Norte | 8,734 | 8,581 | 3 | 150 | **98.3%** |
| **Chapada** | Bairro Costeira / Chapada / Zona Leste | 7,459 | 7,371 | 3 | 85 | **98.9%** |
| **Fonte Nova** | Centro / Bairro Iguaçu / Fazenda Velha | 6,606 | 6,446 | 8 | 152 | **97.7%** |
| **Campestre** | Campestre / Área Rural Sul / Guajuvira | 5,429 | 5,111 | 10 | 308 | **94.3%** |
| **Almeida Torres** | Bairro Boqueirão / Sabiá / Área Urbana | 5,046 | 4,827 | 14 | 205 | **95.9%** |
| **Oleos Brasil** | Polo Industrial / Bairro Thomaz Coelho | 4,695 | 4,534 | 12 | 149 | **96.8%** |
| **Polonia** | Bairro Estação / Tindiquera | 4,122 | 3,985 | 15 | 122 | **97.0%** |
| **Angelica** | Bairro Jardim Iguaçu / Barigui | 3,955 | 3,916 | 1 | 38 | **99.0%** |
| **Independência** | Bairro Capela Velha / Califórnia | 3,806 | 3,514 | 19 | 273 | **92.8%** |
| **Gonvarri** | Polo Siderúrgico / Rio Verde | 3,675 | 3,467 | 20 | 188 | **94.9%** |
| **Sobania** | Bairro Costeira Sul / Vila Angélica | 2,344 | 2,294 | 2 | 48 | **98.0%** |
| **Represa** | Entorno Represa do Passaúna / Rural Norte | 2,158 | 2,003 | 15 | 140 | **93.5%** |
| **Van Leer** | Parque Industrial / Bairro Thomaz Coelho | 1,666 | 1,555 | 9 | 102 | **93.9%** |
| **Sinteko** | Zona Industrial Química / Araucária Sul | 1,601 | 1,491 | 8 | 102 | **93.6%** |
| **Risotolandia** | Área de Distribuição Araucária | 378 | 204 | 15 | 159 | **57.9%** |

---

## 4. Características Individuais Detalhadas das UCs dos Estudos de Caso

Abaixo sintetiza-se a ficha cadastral, elétrica, metrológica e analítica de cada uma das Unidades Consumidoras estudadas no trabalho:

| UC | Alimentador | Classe | Tecnologia | Fase | Consumo (kWh/d) | MAD | Score MoE | Hipótese Diagnóstica | Vistoria? | Síntese Pericial |
|:---|:---|:---:|:---:|:---:|---:|---:|---:|:---|:---:|:---|
| **90850700** | Oleos Brasil | 5442 | Smart-B (Baixa Tensão) | Trifásico | 17.4 | 1.3 | 0.4279 | `REGULAR` | **NÃO** | Fraude Clássica PNT: Apagão 48h prévio + Degrau CUSUM -68.54% permanente (93 dias) |
| **29113296** | CSN | 9101 | Smart-B (Baixa Tensão) | BI | 21.8 | 6.9 | 0.4964 | `REGULAR` | **NÃO** | Controle Negativo: Perda de pacotes 50% em RF Mesh com consumo real estável (~25 kWh/d) |
| **96905697** | Oleos Brasil | 9101 | Smart-B (Baixa Tensão) | Trifásico | 156.8 | 6.6 | 0.4027 | `REGULAR` | **NÃO** | Sazonalidade Comercial: Quedas cíclicas a cada fim de semana (190 -> 65 kWh/d) |
| **82889805** | Van Leer | 4443 | Smart-B (Baixa Tensão) | Trifásico | 71.6 | 42.7 | 0.5927 | `POSSIVEL_GD_NAO_CADASTRADA` | **SIM (Prioritário)** | Geração Distribuída Não Cadastrada: Clássica 'Curva do Pato' solar com injeção diurna |
| **96515961** | Chapada | 4845 | Smart-B (Baixa Tensão) | Bifásico | 29.2 | 2.7 | 0.4225 | `REGULAR` | **NÃO** | Carga Contínua 24h (+241% CUSUM): Carga ininterrupta 2 kWh/h (Mineração Bitcoin / Câmara fria) |
| **105416770** | Oleos Brasil | 9101 | Smart-B (Baixa Tensão) | Trifásico | 11.2 | 5.6 | 0.5627 | `POSSIVEL_GD_NAO_CADASTRADA` | **SIM (Prioritário)** | Operação Regular: Variação de baixo volume absoluto residencial sem desequilíbrio elétrico |
| **91022010** | Polonia | 9101 | Smart-B (Baixa Tensão) | TR | 11.8 | 2.2 | 0.4459 | `REGULAR` | **NÃO** | Regular com Picos Transitórios: Picos pontuais de 29 kWh/d com retorno à base |
| **97847968** | Gonvarri | 9101 | Smart-B (Baixa Tensão) | TR | 40.2 | 8.5 | 0.6074 | `DESEQUILIBRIO_CARGA_OU_BYPASS` | **SIM (Prioritário)** | Vacância / Falha de Rede: Longos apagões telecom (9d) com perfil elétrico invariante ao retornar |

---

## 5. Figuras Anexas Geradas

- 📊 **Distribuição Tecnológica:** [distribuicao_tecnologia_medidores.png](../figures/city_profile/distribuicao_tecnologia_medidores.png)
- 📊 **Smartização por Alimentador:** [alimentadores_smart_vs_convencional.png](../figures/city_profile/alimentadores_smart_vs_convencional.png)
- 📁 **Tabelas Auditadas CSV:**
  - [resumo_tecnologia_medidores.csv](city_profile/resumo_tecnologia_medidores.csv)
  - [perfil_alimentadores_medidores.csv](city_profile/perfil_alimentadores_medidores.csv)
  - [caracteristicas_individuais_ucs_estudo_de_caso.csv](city_profile/caracteristicas_individuais_ucs_estudo_de_caso.csv)
