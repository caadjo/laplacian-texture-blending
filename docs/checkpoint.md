# Checkpoint técnico de Computação Gráfica

## Demonstração de quatro minutos

1. **0:00–0:40 — Repositório e pergunta.** Mostre o histórico Git, abra o README e apresente a pergunta. Quanto de fidelidade se preserva ao usar os níveis 0, 2, 4 e 6? A otimização aparece na Seção 6.2 do artigo de Wronski.
2. **0:40–1:30 — Baseline e protótipo.** Rode `python -m unittest discover -s tests -v` e `python src/laplacian_blend.py --outdir output/baseline`. Abra o comparativo gerado. Na demo WebGL, fixe `mip=6`, mostre o lerp e alterne o passo de 1 para 2 com o Laplaciano habilitado.
3. **1:30–2:50 — Experimento e resultado.** Abra `results/materiais_perlin.png` e `results/metrics.csv`. Mostre a linha inferior na ordem lerp, completo e salto. O caso menos próximo da referência para o salto foi materiais com Perlin, com SSIM de 0,999888 e PSNR de 52,25 dB. Mostre `results/materiais_perlin_erro5x.png`. A contagem teórica passa de 21 para 12 amostras lógicas por fragmento.
4. **2:50–4:00 — Artigo e conclusão.** Abra `article/main.tex` e mostre a estrutura e a tabela. Explique que o piloto é offline, com dois pares de albedo, e que o shader foi exercitado em renderização por software. Apresente os critérios de conclusão.

## Plano para concluir

| Ordem | Dificuldade | Entrega verificável |
| --- | --- | --- |
| 1 | Isolar o custo da mistura e dos passes de máscara | Fixar GPU, navegador, resolução e entradas. Usar `EXT_disjoint_timer_query_webgl2` se disponível e descartar medições disjuntas. Registrar separadamente a preparação da máscara e a mistura. |
| 2 | Separar variação de tempo do efeito da otimização | Aquecer por pelo menos 60 quadros. Coletar 100 consultas válidas por variante em cinco repetições, alternando a ordem das variantes. Publicar valores brutos, mediana e intervalo interquartil. |
| 3 | Generalizar a avaliação visual | Acrescentar três pares de albedo com licenças registradas. Variar a largura da transição. Comparar recortes, SSIM, PSNR e variância. Tratar mapas de normais separadamente, com renormalização e erro angular. |
| 4 | Fechar o manuscrito | Vincular a revisão Git avaliada, incorporar a tabela de tempos e discutir casos com artefatos. Investigar LOD adaptativo e filtros conforme o tempo após o experimento principal. |

Com o servidor local aberto, use `http://localhost:8000/demo_webgl/`. Para apresentar sem o CDN, use o painel e o mapa de erro gerados no piloto.
