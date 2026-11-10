# GPU-Friendly Laplacian Texture Blending

Projeto de Pesquisa Aplicada em Computação Gráfica, IFNMG (2026/2).
Equipe de Icaro Vinicius, Vinicius Macedo e Caio Dias.
Professor Wagner Ferreira de Barros.

## Pergunta de pesquisa

Quanto a omissão de níveis mip altera a mistura Laplaciana de texturas e quanto ela pode economizar em GPU? Partimos do [artigo de Bartlomiej Wronski](https://jcgt.org/published/0014/01/02/) (JCGT, 2025). A omissão de níveis vem da Seção 6.2. A contribuição investigada é a avaliação quantitativa da relação entre fidelidade e custo.

A interpolação linear por pixel (lerp) usa uma máscara em escala única. A técnica Laplaciana mistura diferenças entre mipmaps com máscaras de escalas correspondentes e soma as bandas ao residual. O piloto compara os níveis [0, 1, 2, 3, 4, 5, 6] com [0, 2, 4, 6]. Cada banda agrupada usa a máscara do seu nível mais fino.

## Reprodução

Na raiz do repositório, no WSL/Linux, com Python 3.12:

~~~
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/laplacian_blend.py --outdir output/baseline
python src/experiment.py --outdir results
~~~

No PowerShell, crie o ambiente com `py -3.12 -m venv .venv` e use `.\.venv\Scripts\python.exe` no lugar de `python`. O executor original aceita `--tex0`, `--tex1`, `--mask`, `--levels` e `--outdir`. Use `--levels 6 --stride 2` para gerar uma comparação individual com níveis alternados.

O experimento cruza dois pares de albedo, `1.jpg`/`2.jpg` e `1.jpg`/`1rot.jpg`, com as máscaras Perlin e rampa. Gera o [CSV com 12 medições](results/metrics.csv), o [registro do ambiente](results/environment.json), painéis e mapas de erro absoluto amplificado em 5×. A linha superior dos painéis contém textura A, máscara e textura B. A linha inferior mostra lerp, Laplaciano completo e salto. Veja os [créditos das texturas](demo_webgl/data_credits.txt).

## Demo WebGL

Execute `python -m http.server 8000` e abra `http://localhost:8000/demo_webgl/`. A demo requer WebGL e acesso ao CDN do Three.js. `enable` alterna entre lerp e mistura Laplaciana. Com `enable` ligado, fixe `mip` em `6` e alterne `Passo dos níveis` entre `1` e `2`.

A demo usa viés de LOD e renderiza a máscara em alvos separados. No lerp da demo, a máscara é suavizada pelo nível selecionado. A comparação quantitativa usa as mesmas entradas e a mesma máscara base no executor offline. O protótipo WebGL foi exercitado no Chrome com SwiftShader por comparação do canvas antes e depois da troca de passo.

## Resultados preliminares

| Cenário | SSIM do lerp | SSIM do salto | PSNR do lerp | PSNR do salto |
| --- | ---: | ---: | ---: | ---: |
| Materiais e Perlin | 0,990065 | 0,999888 | 30,09 dB | 52,25 dB |
| Materiais e rampa | 0,999996 | 0,999999 | 68,40 dB | 76,89 dB |
| Rotação e Perlin | 0,998673 | 0,999913 | 39,99 dB | 53,33 dB |
| Rotação e rampa | 0,999998 | 0,999999 | 73,10 dB | 77,12 dB |

A referência é o Laplaciano completo de seis bandas. SSIM e PSNR usam a região em que a máscara base está entre 0,05 e 0,95. O salto foi mais próximo da referência nos quatro cenários. Com Perlin no par de materiais, a variância local média foi de 0,009528 no lerp, 0,009354 na referência e 0,009375 no salto. Essa métrica responde tanto ao contraste das texturas quanto às bordas da transição.

A contagem teórica passa de 21 para 12 amostras lógicas por fragmento, queda de 42,9%. O tempo de GPU e o custo de preparar a máscara ainda precisam ser medidos.

## Artigo e apresentação

O [manuscrito LaTeX](article/main.tex) contém introdução, referências, método, resultados e limitações. Compile a partir de `article/` com `latexmk -pdf main.tex` ou `tectonic main.tex`. As figuras usam as saídas presentes em `results/`.

O [roteiro do checkpoint](docs/checkpoint.md) organiza uma demonstração de quatro minutos e define as etapas restantes. A pasta `slides/` guarda os slides da etapa de proposta.

## Apoio de IA

O Google Gemini (Antigravity Assistant) auxiliou na compreensão do artigo e no protótipo inicial. O Codex auxiliou na adaptação dos executores, no piloto e na redação do checkpoint. O CSV, os testes e o protocolo registram as evidências desta etapa.
