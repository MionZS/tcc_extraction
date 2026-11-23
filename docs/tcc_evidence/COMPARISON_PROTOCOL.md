# Protocolo de Comparação Entre Execuções

1. Comparar apenas execuções com manifesto `status == "validated"` e mesmo `unit_of_analysis`.
2. Verificar interseção populacional (`pop_intersection`) antes de interpretar variações em taxa de candidatos.
3. Decompor variações em: mudança populacional, alteração de cobertura telecom, troca de versão de modelo/features e alteração de limiar.
