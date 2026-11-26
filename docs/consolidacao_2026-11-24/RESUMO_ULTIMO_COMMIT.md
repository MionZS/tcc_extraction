# Resumo do commit c1539b4

Data: 24/11/2026. Tema: saneamento metodológico de LOF e PCA, reconciliação textual e correções pré-defesa.

## Melhorias implementadas

- **LOF:** pontuação bruta convertida por `log1p(max(-score_samples - 1, 0))`, reduzindo picos causados por distâncias nulas e empates antes da normalização.
- **PCA residual:** padronização das variáveis no ajuste e na inferência; erro médio quadrático calculado no espaço padronizado; número de componentes limitado também pela quantidade de amostras.
- **Normalização:** quando P01 e P99 coincidem, tenta P99,9 antes de recorrer aos extremos mínimo/máximo.
- **Imputação:** `keep_empty_features=True` preserva colunas inteiramente vazias no ajuste.
- **Manuscrito:** adicionados fonte LaTeX modular (`main.tex`, capítulos, pré-textuais, bibliografia, figuras e template), mapa de feedback e documentação de pendências. Texto distingue comitê OOT de seis detectores, variante histórica de quatro e ensaio de ablação; separa resultados retrospectivos de projeções operacionais e validação futura.
- **Pré-textuais:** título, orientador, data prevista e banca foram registrados; campos administrativos ainda exigem conferência.

## Arquivo único disponível

`TCC_AMI_Consolidado.pdf`: PDF A4 de 37 páginas incluído neste commit. O projeto LaTeX modular documenta uma versão de 78 páginas, mas não há compilador LaTeX disponível neste ambiente e o PDF versionado não comprova correspondência com esse fonte. Portanto, não tratar o PDF de 37 páginas como compilação atual do projeto modular.

## Pendências principais

- Compilar `TCC_AMI_eekBR/main.tex` com LaTeX/Biber e conferir correspondência com o fonte, sumário, figuras e citações.
- Reconciliar cobertura do caso `685***20`; fechar séries dos estudos de caso e painéis metodológicos.
- Guardar manifesto, artefatos e tabelas de fatos dos ensaios; confirmar dados da banca e autorização institucional antes de publicação.
- Não há neste commit nova execução documentada do pipeline ou validação por inspeções de campo.
