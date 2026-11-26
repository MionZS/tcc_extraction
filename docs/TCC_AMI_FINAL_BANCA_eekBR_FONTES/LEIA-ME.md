# TCC AMI — projeto final integrado

Arquivo mestre: `main.tex`. Projeto eekBR/UFPR modular, com os capítulos ativos da versão da banca, Apêndice A (proveniência) e Apêndice B (demonstrações controladas complementares).

Compilar com pdfLaTeX e Biber:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternativa: `pdflatex main.tex`, `biber main`, `pdflatex main.tex` (duas vezes).

Para Overleaf, importe o ZIP e selecione `main.tex` como documento principal.

Consulte `INTEGRACAO.md` e `DADOS_PRETEXTUAIS.md`. Termo de aprovação e ficha catalográfica permanecem desativados até confirmação institucional.
