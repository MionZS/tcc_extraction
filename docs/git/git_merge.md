Para aplicar mudanças de um branch para outro, você tem várias opções, dependendo do que precisa fazer. Aqui está um guia completo:

## **Métodos Principais**

### **1. Merge (Recomendado para mudanças completas)**
```bash
# No branch de destino
git checkout target-branch
git merge source-branch
```
- **O que faz:** Aplica todos os commits do source-branch para o target-branch
- **Bom para:** Mudanças completas, incluindo moves, rewrites, renames
- **Resultado:** Histórico combinado, possível conflito se houver mudanças sobrepostas

### **2. Rebase (Limpa histórico)**
```bash
# No source-branch
git checkout source-branch
git rebase target-branch
# Resolve conflitos se necessário
git checkout target-branch
git merge source-branch --ff-only
```
- **O que faz:** Re-escreve commits do source-branch sobre o target-branch
- **Bom para:** Histórico limpo, quando você quer reordenar commits
- **Cuidado:** Re-escreve histórico (problema se já foi pushado)

### **3. Cherry-pick (Commits específicos)**
```bash
# No branch de destino
git checkout target-branch
git cherry-pick commit-hash-1 commit-hash-2 ...
```
- **O que faz:** Aplica commits individuais
- **Bom para:** Mudanças específicas, quando você quer seletividade

## **Moves e Rewrites**

### **Git Detecta Automaticamente Moves**
```bash
# Se você renomeou um arquivo no source-branch:
git mv old-name.py new-name.py  # No source-branch
# Git rastreia isso como um delete + add, mas geralmente detecta como um move
```

### **Forçando Git a Reconhecer Moves**
```bash
# No source-branch antes de merge/rebase
git add -A  # Stage todas as mudanças
git commit -m "Move/rename changes"

# Ou use filter-branch (cuidado - re-escreve histórico)
git filter-branch --tree-filter 'git mv old-path new-path'
```

## **Fluxo de Trabalho Recomendado**

### **Cenário: Você tem mudanças completas no feature-branch que quer aplicar ao main**

```bash
# 1. Prepare o branch de destino
git checkout main
git pull origin main  # Mantém main atualizado

# 2. Merge as mudanças
git merge feature-branch

# 3. Resolva conflitos se ocorrerem
# (Git mostrará onde as mudanças se sobrepõem)

# 4. Faça push (se você tiver permissão)
git push origin main
```

### **Cenário: Você quer um histórico limpo (rebase)**

```bash
# 1. Rebase feature-branch sobre main
git checkout feature-branch
git rebase main

# 2. Resolva conflitos
# (Git pedirá para resolver conflitos durante rebase)

# 3. Force push (cuidado - re-escreve histórico)
git push origin feature-branch --force

# 4. Abra um Pull Request
# (GitHub/GitLab vai revisar antes de merge)
```

## **Casos Específicos**

### **Apenas Moves/Renames**
```bash
# No branch de origem
git checkout feature-branch
git mv directory/old-file.txt directory/new-file.txt
git add directory/new-file.txt
git rm directory/old-file.txt  # Opcional - Git geralmente detecta automaticamente

# No branch de destino
git checkout main
git merge feature-branch  # Git detecta o move automaticamente
```

### **Rewrites (Mudar conteúdo de arquivos)**
```bash
# No branch de origem
git checkout feature-branch
# Edite arquivo.py completamente
git add arquivo.py
git commit -m "Rewrite arquivo.py"

# No branch de destino
git checkout main
git merge feature-branch  # Aplica o rewrite
```

## **Problemas e Soluções**

### **Conflitos de Merge**
```bash
# Git irá parar e pedir para você resolver
# Edite os arquivos conflitantes
git add arquivo-conflito.py
git merge --continue  # Ou git commit
```

### **Força Push (Cuidado!)**
```bash
# Apenas se você controlou o branch
git push origin feature-branch --force
# PERIGO: Re-escreve histórico para outros que já clonaram
```

## **Melhor Prática para seu Caso**

**Se você está começando e o branch nunca foi pushado:**
```bash
git checkout main
git merge feature-branch
```

**Se você já pushou o feature-branch:**
```bash
# 1. Rebase localmente (apenas para o seu clone)
git checkout feature-branch
git rebase main

# 2. Force push (apenas se você controlou o branch)
git push origin feature-branch --force

# 3. Abra um Pull Request para revisão
```

## **Verificação Rápida (Dry-run Real)**

```bash
# 1. Veja os commits que seriam aplicados (sem tocar no worktree)
git log --oneline main..feature-branch

# 2. Veja estatísticas do que mudaria (sem tocar no worktree)
git diff --stat main..feature-branch

# 3. DRY-RUN REAL: Faz o merge mas PARA antes de commitar
#    Permite inspecionar conflitos, arquivos modificados, etc.
git checkout main
git merge --no-commit --no-ff feature-branch

#    - --no-commit: Para antes de criar o commit de merge
#    - --no-ff:   Força criar commit de merge mesmo se fast-forward possível
#    - Resultado: Worktree/índice têm o merge aplicado, mas NÃO commitado

# 4. Inspecione o resultado
git status                    # Arquivos staged/unstaged
git diff --cached             # Mudanças que seriam commitadas
git diff                      # Mudanças não staged (conflitos)

# 5. DECIDA:
#    - Se OK:     git commit          # Finaliza o merge
#    - Se NÃO OK: git merge --abort   # Cancela tudo, volta ao estado original
```

> **Nota:** `git merge --stat` **NÃO** é um dry-run — só mostra estatísticas. O merge real com `--no-commit --no-ff` é a forma correta de testar antes de commitar.

**Resumo:** Para mudanças completas (incluindo moves, rewrites, renames), **merge** é o método mais seguro. Use **rebase** apenas se você quiser um histórico limpo e controlar completamente o branch.