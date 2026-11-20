# Pendências Reais Externas (Fora do Alcance Local / Offline)

1. **Rótulos Independentes de Campo (Inspeções Executadas / PNT Confirmadas):**
   - O repositório local não contém tabela de retorno de ordens de serviço (OS) executadas em campo com resultado (`procedente` / `improcedente` / `kWh recuperado`).
   - Sem esses rótulos independentes, **não é possível afirmar** *Precision@K*, *Recall* real de fraude, *Yield@K* alcançado ou taxa de falsos negativos de PNT.
2. **Acesso Oracle Corporativo (CIS / GEO / ORCA):**
   - Consultas diretas a bancos produtivos não são executadas no gerador de evidências por política de segurança e isolamento local.
3. **Execução Automatizada em Ambientes com GPO Restrita:**
   - Caso o ambiente de terminal corporativo bloqueie subshell do agente, o usuário deve executar os comandos listados em `REPRODUCAO.md` diretamente no CMD/PowerShell autorizado.
4. **Capturas Atualizadas da TUI e do Cockpit Reconciliado:**
   - Após a correção dos KPIs no Cockpit (`scripts/run_web_cockpit.py`), recomenda-se capturar novas screenshots das telas principais para substituir as capturas anteriores no documento final.
