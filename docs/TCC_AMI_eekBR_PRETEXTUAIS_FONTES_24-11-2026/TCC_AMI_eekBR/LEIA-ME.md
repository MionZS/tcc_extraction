# Projeto LaTeX — TCC AMI integrado

Arquivo principal: `main.tex`. Conteúdo: `cap01.tex` a `cap15.tex`, carregados por `\include` em `main.tex`. Metadados, resumo e abstract: `00-dados.tex`. Bibliografia: `referencias.bib`.

Compilar na raiz deste projeto com TeX Live ou ambiente equivalente, com abntex2, biblatex-abnt, Biber e latexmk instalados:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternativa: executar `pdflatex main.tex`, `biber main`, `pdflatex main.tex` e `pdflatex main.tex`.

Consulte `MAPA_FEEDBACK.md` para rastreabilidade das mudanças e pendências externas. Orientação, título em português, tipo TCC II, data prevista e nomes da banca estão registrados em `00-dados.tex`. As listas técnicas estão em `00-siglas.tex` e `00-simbolos.tex`. O `README.md` original e `TEMPLATE_ORIGIN.md` identificam o template. O ZIP contém o fonte atualizado, sem os manuscritos Markdown antigos que poderiam gerar divergência.

## Organização dos capítulos

- `cap01.tex` — Introdução
- `cap02.tex` — Problema de pesquisa e objetivos
- `cap03.tex` — Fundamentação metodológica
- `cap04.tex` — Evolução metodológica e comparação com a abordagem original
- `cap05.tex` — Arquitetura de dados
- `cap06.tex` — Engenharia do pipeline: das fontes ao dataset analítico
- `cap07.tex` — Análise exploratória do conjunto de dados
- `cap08.tex` — Arquitetura analítica multivisão
- `cap09.tex` — Demonstração experimental das técnicas de aprendizado de máquina
- `cap10.tex` — Saídas como padrões de compatibilidade
- `cap11.tex` — Estudos de caso e anatomia da detecção
- `cap12.tex` — Aplicação operacional e TUI
- `cap13.tex` — Resultados e discussão
- `cap14.tex` — Projeção de operacionalização, governança e avaliação econômica
- `cap15.tex` — Limitações e conclusões

A divisão é byte a byte do antigo `conteudo.tex`, sem alteração científica. Compilação completa comparada com PDF mestre anterior: 78 páginas, texto e renderização a 36 dpi idênticos em todas as páginas. O aviso de opções globais não utilizadas permanece herdado do projeto anterior. Para compilação parcial, use `\includeonly{capXX}` no preâmbulo, após uma compilação integral. Remova a opção antes da entrega.

Consulte `DADOS_PRETEXTUAIS.md` para as informações confirmadas e pendências. O termo de aprovação, a ficha catalográfica e os elementos pessoais opcionais permanecem desativados.
