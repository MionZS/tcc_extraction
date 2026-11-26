# Integração final — 26/11/2026

Base: `TCC_AMI_BANCA_2026-11-26_eekBR_FONTES.zip`. Complemento: `manuscrito_final.zip` enviado nesta solicitação. O complemento contém 27 conflitos Git não resolvidos em 12 arquivos TeX. A árvore anterior, já compilada e reconciliada, foi preservada como ponto de partida.

## Intervenções
- Preservados 13 capítulos ativos, resultados da campanha congelada de 25/11, perfis AMI reais, IF–LOF OOT, apêndice de proveniência e correções de granularidade.
- Incorporadas as três demonstrações controladas de IF, LOF e percentis por circuito como Apêndice B, com explicação e separação inequívoca dos resultados reais.
- Mantida a redação quantitativa do benchmark local, com delimitação explícita das etapas não medidas, explicação histórica de normalização/LOF/PCA, distinção entre qualidade de comunicação e cobertura intervalar, contexto da série legada e correção da conclusão dos casos.
- Mantidas a tabela de sensibilidade com limiar fixo de 0,5226 e a indicação de vistoria observada no OOT (limiar 0,5131), sem atribuir seus resultados ao código pós-merge.
- Não restaurada a versão sintética integral de cap09, nem as figuras antigas de PCA, resíduos e arquétipos, que não são necessárias para os resultados reais; o PDF `resid.pdf` do complemento ainda apresenta erro sintático na leitura. Não restaurados os capítulos 10 e 11 absorvidos por outros capítulos.
- Não adotadas comparações causais entre populações diferentes, afirmações de fluxo estável após retirada de medidor, ou justificativas não verificadas para as 16 exclusões.
- Metadados pré-textuais e template preservados.

## Limites
Os painéis e resultados são provenientes dos documentos fornecidos, não de uma nova execução dos scripts. O ZIP recebido não contém as tabelas de fatos e os modelos congelados; reprodução independente não realizada. A validação de campo, OS/TOI e desempenho pericial permanecem fora do escopo. Termo de aprovação, ficha catalográfica e conferência institucional da banca continuam pendentes.

## Validação da entrega

- Compilação concluída por latexmk/pdfLaTeX + Biber, 86 páginas A4, 13 capítulos ativos, 20 figuras, 13 tabelas e dois apêndices.
- Log final sem erros de compilação, referências indefinidas, caracteres ausentes ou caixas overfull; permanece aviso de opções globais não utilizadas, herdado do template.
- Nenhum marcador de conflito Git nos fontes TeX.
- Inspeção visual das páginas e dos novos apêndices.
