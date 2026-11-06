# Pixel-Depth

Experimento separado: blocos UND do seu PixelSeg (projeção de patches, RMSNorm,
GQA, MRoPE e SwiGLU) com quatro saídas intermediárias, atenção local/global
entre vistas e cabeça DA3 para depth + ray + confiança.

Os arquivos do DA3 e do SegPixelUMM existentes na pasta pai não são alterados.
O pacote é independente de `depth_anything_3` e `seg_pixelumm` na execução.
O DPT, DualDPT, CameraDec e utilitários necessários foram copiados com as licenças
originais para `pixel_depth/da3_heads`. Os hashes das fontes do DA3 estão em
`DA3_PROVENANCE.json`. A única alteração nos arquivos copiados é um import
local no DPT. RMSNorm e a matemática rotary vieram do seu SegPixelUMM.

## Instalação

Use Python >=3.10 e PyTorch >=2.5 (GQA no SDPA).

**Execute na subpasta `pixel-depth`, não na pasta pai `Pixel-depth`.**
A pasta pai ainda tem o `pyproject.toml` do DA3 original. Instalar nela baixa
as dependências do DA3, incluindo bibliotecas CUDA/3D que este pacote não usa.
Se você copiou o conjunto para `~/Pixel-depth`, comece assim:

```bash
cd ~/Pixel-depth/pixel-depth
python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['name'])"
# Deve imprimir: pixel-depth (e não depth-anything-3).
```

A verificação acima usa Python >=3.11. Com Python 3.10, confira o campo
`name = "pixel-depth"` no `pyproject.toml` manualmente.

Dentro da subpasta correta:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Instale antes o PyTorch adequado à sua GPU/CUDA, se necessário.
pip install -e .
pip install pytest
python -m pytest -q
```

Não precisa instalar o DA3 original, Transformers, xformers ou baixar pesos.
A implementação usa pesos novos por padrão. Não é compatível diretamente com
checkpoints oficiais do DA3 ou PixelUMM.

## API

```python
import torch
from pixel_depth import PixelDepth, PixelDepthConfig, depth_loss

model = PixelDepth(PixelDepthConfig())
images = torch.randn(1, 2, 3, 224, 320)  # B, vistas, canais, H, W
output = model(images, export_feat_layers=(7, 11))
# depth/depth_conf: [B, vistas, H, W]
# extrinsics: [B, vistas, 3, 4], world-to-camera
# intrinsics: [B, vistas, 3, 3]
# pose_encoding: [B, vistas, 9], camera-to-world translation, xyzw quaternion, FOV
# aux: dict camada -> [B, vistas, H_padded/p, W_padded/p, hidden_size]
loss = depth_loss(output, torch.ones(1, 2, 224, 320))
loss.backward()
model.save('runs/example/model.pt')
restored = PixelDepth.load('runs/example/model.pt').eval()
```

Entrada monocular `[B,C,H,W]` também é aceita; a saída mantém a dimensão de
vistas. Padding replicate até múltiplos do patch é interno; a profundidade
retorna na resolução original. O DPT e a geometria de câmera operam em FP32
mesmo sob autocast. `model.half()` não é suportado; use autocast.

A normalização é externa: prepare dados de treino, validação e inferência
com as mesmas estatísticas, calculadas no treino. Para RGB em [0,1], uma
possibilidade é `(image - 0.5) / 0.5`; não há conversão automática de uint8.
Os canais podem ser multiespectrais via `in_channels`, com normalização por banda.

## Arquitetura e configurações

Para gerar um desenho simplificado da arquitetura ativa em SVG:

```bash
python -m pixel_depth.cli architecture \
  --config configs/small.json \
  --output architecture.svg
```

O diagrama resume o caminho `input → patch embedding → backbone local/global →
DualDPT` e mostra as saídas de depth, ray, confidence e câmera. Linhas
tracejadas representam condicionamento ou supervisão opcionais.

- `configs/small.json`: 12 blocos, largura 384, quatro features locais/globais
  de 768 canais, patches 16, profundidade/confiança e predição de câmera.
  Após `alt_start=4`, blocos ímpares fazem atenção global entre todas as vistas;
  os demais fazem atenção independente por vista. A concatenação preserva
  a última feature local e a feature corrente normalizada, como no DA3.
- `configs/mono.json`: quatro blocos, largura 128, sem atenção entre vistas,
  câmeras ou confiança. Útil para primeiro treino/ablação.
- `out_layers` são quatro índices distintos em ordem crescente, começando em zero.
- `gradient_checkpointing=true` reduz memória de ativações no backbone.
- `patch_size=14` também é permitido: não usamos o decoder progressivo do
  PixelSeg, que impunha potência de dois. Isso muda a projeção de entrada e
  impede transferência direta de um checkpoint treinado com patches 16.

A vista zero é sempre a referência. Não há seleção/reordenação automática de
referência. Os tokens de referência e das demais vistas são aprendidos; o eixo
temporal do MRoPE permanece zero. Cada vista usa a mesma grade espacial, com
coordenadas zero reservadas ao token de câmera. Isto é uma adaptação para
geometria multivista, não o processamento de tubelets do PixelUMM original.

O decoder usa uma cabeça DualDPT: a branch principal prevê profundidade e
confiança; a branch auxiliar prevê ray map de seis canais e confiança. Ainda não
há neste pacote uma cabeça de Gaussian Splatting nem a API
`DepthAnything3.inference`.

## Transferir seu checkpoint de segmentação

`model.load_pixelseg_backbone(state_dict)` aceita `SegPixelUMM.state_dict()` ou
`SegPixelUMM.backbone.state_dict()`. Exige igualdade de canais, patch, largura,
camadas, cabeças e dimensões de atenção; valida nomes e shapes antes de carregar.
Os pesos do decoder de segmentação são descartados e o token de câmera é novo.
MRoPE theta/epsilon não estão nos pesos: mantenha-os iguais à configuração original.
Adicionar câmera e atenção entre vistas muda o comportamento mesmo com os pesos
transferidos; isso não garante retenção da qualidade de segmentação.

O seu experimento Sentinel-2 tem **13 canais**, largura 128 e quatro blocos.
Para transferi-lo, parta de `configs/mono.json` e defina `in_channels=13`; para
uma experiência RGB, a projeção de 13 bandas não pode ser carregada diretamente.

O CLI aceita `--pixelseg-checkpoint backbone.pt`, carregado com `weights_only=True`.
Para checkpoints completos do seu trainer que contenham estados NumPy/Python,
exporte primeiro apenas o state_dict usando seu ambiente e um checkpoint local
confiável:

```python
import torch
saved = torch.load('run/best.pt', map_location='cpu', weights_only=False)
torch.save(saved['model'], 'backbone.pt')
```

## Dados e treino posterior

Cada amostra é um `.npz` com:

| Chave | Formato | Conteúdo |
| --- | --- | --- |
| `image` | `[C,H,W]` ou `[S,C,H,W]` | float32 já normalizado |
| `depth` | `[H,W]` ou `[S,H,W]` | profundidade positiva, nas mesmas unidades em todo o dataset |
| `valid` | igual a `depth`, opcional | máscara 0/1 de supervisão |
| `teacher_depth` | igual a `depth`, opcional | pseudo-depth produzido offline pelo DA3 Teacher |
| `teacher_valid` | igual a `depth`, opcional | máscara da supervisão teacher |
| `ray` | `[H,W,6]` ou `[S,H,W,6]` | origem + direção do raio por pixel |
| `ray_valid` | `[H,W]` ou `[S,H,W]`, opcional | máscara da supervisão de rays |
| `extrinsics` | `[S,3,4]` ou `[S,4,4]`, opcional | world-to-camera |
| `intrinsics` | `[S,3,3]`, opcional | calibração em pixels |

Na API, câmeras também exigem eixo de batch. Extrínsecas devem estar no sistema
da primeira vista, com translações nas mesmas unidades da profundidade. Alinhe-as
no preparo do dataset. O modelo não estima essa transformação de referência.
NaN, Inf e profundidades <=0 são ignorados pela loss. Amostras/batches inteiros
sem supervisão válida geram erro. Entradas de imagem precisam ser finitas.
Para batch >1, as amostras precisam ter as mesmas dimensões e número de vistas.

Crie manifests CSV, com caminhos relativos ao próprio CSV:

```csv
sample
samples/scene_001.npz
samples/scene_002.npz
```

```bash
pixel-depth train --config configs/small.json \
  --train-manifest /dados/train.csv --val-manifest /dados/val.csv \
  --output runs/pixel-small-seed42 --epochs 20 --device cuda

# Treinamento DA3-style completo: exige teacher_depth e ray em cada NPZ.
# Primeiro gere os pseudo-labels com o DA3MONO-LARGE (ou seu checkpoint teacher):
python scripts/generate_teacher_labels.py \
  --manifest /dados/train.csv --output-manifest /dados/train_full.csv \
  --output-dir /dados/teacher_npz --model depth-anything/DA3MONO-LARGE

pixel-depth train --full-da3 --config configs/small.json \
  --train-manifest /dados/train_full.csv --val-manifest /dados/val_full.csv \
  --output runs/pixel-full-seed42 --epochs 20 --device cuda

pixel-depth evaluate --checkpoint runs/pixel-small-seed42/last.pt \
  --manifest /dados/test.csv --output runs/pixel-small-seed42/test.json

pixel-depth infer --checkpoint runs/pixel-small-seed42/last.pt \
  --input /dados/samples/scene_001.npz --output predictions/scene_001.npz
```

O trainer usa AdamW, clipping, loss de profundidade em log com componente
invariante à escala e supervisão de confiança. Com `--ray-weight > 0`, também
supervisiona o ray map; com `--teacher-weight > 0`, usa `teacher_depth` como
pseudo-label adicional. `--full-da3` ativa ambos e falha se os alvos não
existirem. Com poses anotadas, também é possível supervisionar a câmera por
rotação, translação e calibração via `--camera-weight`. Sem poses,
o decoder de câmera **não é treinado** e suas predições não são utilizáveis.
`--condition-cameras` injeta câmeras conhecidas no backbone (exige ambas as
anotações); por padrão o modelo aprende sem esse condicionamento. No protocolo
de avaliação, use o mesmo regime e não forneça câmeras que o baseline não recebe.

Gera `run.json` (configuração, parâmetros e seed), `metrics.jsonl` e `last.pt`.
Salva o modelo a cada época; o checkpoint deste trainer não inclui estado do
otimizador nem retomada. O diretório de saída deve estar vazio. Não há DDP,
scheduler ou AMP neste trainer inicial; a API suporta gradient checkpointing e
autocast externo. Seeds não garantem determinismo entre GPUs/backends.

## Comparação futura

A avaliação registra AbsRel, RMSE e delta1 **sem realinhar escala**; as métricas
são agregadas por pixel válido. Uma avaliação relativa com alinhamento de escala
precisa de um protocolo separado, igual para ambos os modelos. Confiança é um
score aprendido, sem garantia de calibração probabilística.

Separe treino/validação/teste por cena ou região e mantenha os mesmos splits,
resolução, normalização, dados, número de vistas, regime de câmeras, orçamento
de treino e seeds. O CLI verifica sobreposição de caminhos entre treino e
validação, mas não detecta cenas correlacionadas ou arquivos duplicados.
Compare também parâmetros, memória e tempo. O preset small aproxima a largura
DA3-Small, mas não iguala automaticamente parâmetros/FLOPs.

Comparar Pixel-Depth do zero com DA3 pré-treinado mede o sistema disponível;
isolar o efeito do backbone exige uma comparação adicional com treino e
inicialização controlados. Nenhum resultado de qualidade foi produzido aqui.

## Verificação

```bash
python -m pytest -q
```

Os testes usam tensores sintéticos pequenos: forward/backward, padding,
comunicação entre vistas, condicionamento de câmera, checkpoint round-trip,
gradient checkpointing, transferência de pesos, patches 14 e fluxo CLI de
uma atualização sintética seguida de avaliação/inferência. Não iniciam um
experimento com dados reais e não validam qualidade científica.

## Baixar dados e comparar dois treinamentos

O novo script executa o experimento completo, **sequencialmente**: baixa NYUv2,
cria a validação, treina DA3 monocular, treina Pixel-Depth monocular, seleciona o
melhor checkpoint de cada um pela validação e avalia ambos no teste reservado.
Ele só inicia downloads/treinos quando você o executar.

Dentro da subpasta `pixel-depth`, com seu ambiente ativado:

```bash
python -m pip install --no-cache-dir -e '.[comparison]'

# Primeiro inspecione a configuração, sem baixar ou treinar:
bash scripts/compare_nyu.sh --work-dir /disco/com-espaco/pixel-depth-nyu --dry-run

# Executa 6 treinos (2 arquiteturas × seeds 42, 43 e 44) e exporta a galeria:
bash scripts/compare_nyu.sh \
  --work-dir /disco/com-espaco/pixel-depth-nyu \
  --device cuda --epochs 30 --batch-size 2 --accumulate 4 \
  --gradient-checkpointing
```

Substitua `/disco/com-espaco/...` por um caminho real. Não precisa reinstalar o
DA3 original. O extra `comparison` adiciona apenas `h5py` e `scipy` às dependências
do Pixel-Depth; o PyTorch para sua GPU deve já estar corretamente instalado.
O script usa o Python do ambiente ativado e pode ser chamado de qualquer pasta.
Também está disponível como `pixel-depth-compare` após reinstalar o pacote.

### O que está sendo comparado

| Item | DA3 monocular | Pixel-Depth monocular |
| --- | --- | --- |
| Inicialização | Do zero | Do zero |
| Encoder | Código ViT/DINOv2 do DA3 | Seus blocos UND: RMSNorm, GQA, MRoPE, SwiGLU |
| Patches | 14 | 14 |
| Preset padrão | ViT-S: largura 384, 12 blocos, 6 cabeças | Largura 384, 12 blocos, 6 cabeças Q / 3 KV |
| Features intermediárias | Blocos 5, 7, 9, 11 | Blocos 5, 7, 9, 11 |
| Decoder | DPT depth-only (protocolo legado) | DPT depth-only (protocolo legado) |
| Otimização | AdamW + cosine + clipping | Idêntica |
| Loss | Log-depth, peso SI=0.5 | Idêntica |
| Atenção entre vistas/câmeras | Desativada | Desativada |

**É uma ablação monocular controlada, não uma reprodução do pré-treinamento
completo do artigo DA3 nem uma comparação multivista.** O baseline Small é uma
adaptação monocular escalada do encoder original, não um checkpoint oficial
chamado DA3-Mono-Small. `--size large` usa os parâmetros do encoder
DA3-Mono-Large: largura 1024, 24 blocos, 16 cabeças, saídas 4/11/17/23. O head
de sky é desativado em ambos por não haver supervisão de céu neste experimento.
O número de parâmetros/FLOPs não é forçado a ser igual; os parâmetros reais
são registrados. `--size tiny` existe somente para verificar o fluxo rapidamente.

Para comparar as branches completas depth+ray+teacher, não use `compare_nyu.sh`:
ele permanece deliberadamente depth-only. Use os manifests NPZ com câmeras e
pseudo-labels `teacher_depth` no comando `pixel-depth train --full-da3`, com o
mesmo protocolo e os mesmos alvos para os dois modelos.

Os arquivos do encoder estão em `pixel_depth/da3_encoder`, com os avisos
originais de licença. `DA3_ENCODER_PROVENANCE.json` registra os hashes das
fontes. As adaptações são imports locais, logger padrão e a constante de seleção
de referência; os cálculos do encoder original foram preservados. A seleção de
referência não é utilizada no modo monocular. O checkpointing é aplicado por
um wrapper no experimento, sem editar o fonte original.

### Dataset e protocolo

Usa o [NYUv2 rotulado oficial](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html):
1.449 pares RGB/profundidade, arquivo MAT de aproximadamente 3 GB. O download
vai direto ao diretório escolhido, sem duplicação em cache pip ou conversão em
milhares de NPZs. Arquivos `.part` permitem retomar downloads se o servidor
suportar HTTP Range; se ignorar Range, o script reinicia o arquivo corretamente.
Um arquivo MAT oficial local pode ser reutilizado com `--labeled-mat /dados/nyu.mat`.

Os [splits distribuídos pelo BTS](https://github.com/cleinc/bts/blob/master/utils/splits.mat)
separam 795 imagens de treino e 654 de teste. A URL, estrutura e SHA256 do arquivo
de splits foram verificados. O script retira cerca de 10% **das cenas de treino**
para validação e verifica ausência de cenas compartilhadas entre as partições.
A quantidade final de imagens de treino/validação depende do tamanho das cenas.
O teste permanece reservado; nunca seleciona checkpoints. `--split-seed` controla
essa divisão, independentemente das seeds dos modelos.

Os dados HDF5 são lidos sob demanda por amostra, sem carregar todo o dataset na
RAM. Imagens são redimensionadas para 224×294 e recebem a mesma normalização
ImageNet nos dois modelos. Não há augmentations diferentes entre arquiteturas.
Profundidade é em metros, com supervisão dentro da região válida. No treino,
profundidade/máscara são redimensionadas com nearest. Na avaliação, a predição
é reamostrada à resolução original 480×640, usando crop `y=45:471, x=41:601` e
profundidades válidas entre 0.001 e 10 m. Reporta média por imagem de:

- `raw_abs_rel`, `raw_rmse`, `raw_delta1`: escala bruta, com clipping [0.001,10].
- `median_aligned_*`: alinhamento por mediana feito separadamente por imagem,
  usando a profundidade de referência, seguido do mesmo clipping.

O alinhamento usa o ground truth apenas para calcular a métrica relativa, sem
alterar os checkpoints. A seleção usa **AbsRel bruto na validação**. A resolução
reduzida e este orçamento de treino são escolhas do experimento; os números
não devem ser apresentados como reprodução dos resultados publicados do DA3.
NYUv2 rotulado é pequeno: uma derrota ou vitória aqui não estabelece superioridade
universal da arquitetura nem substitui avaliação multivista posterior.

### Espaço, memória e retomada

O script verifica espaço antes de começar: para Small, reserva aproximadamente
3 GB de download mais 2 GiB por seed para checkpoints (6 GiB nas três seeds
padrão); Large exige uma reserva de 20 GiB por seed para checkpoints, além dos dados. Instalação do PyTorch/CUDA e cache pip usam
espaço adicional. O número exato depende dos modelos e do sistema de arquivos.
Small é o padrão; Large custa muito mais memória/tempo.

O batch efetivo padrão é 8: `batch-size=2 × accumulate=4`. Para pouca VRAM, use
`--batch-size 1 --accumulate 8 --gradient-checkpointing`. Opcionalmente:
`--amp bf16` em GPU compatível ou `--amp fp16`; o padrão é FP32. Ambos recebem o
mesmo regime de precisão. Não altera automaticamente a configuração se faltar
memória. Os tempos incluem carregamento dos dados e podem sofrer efeito de
cache/sistema; a memória registrada é o pico alocado pelo PyTorch, não o total
reservado no dispositivo.

Para retomar após interrupção, repita **o mesmo comando** acrescentando `--resume`.
Este pipeline de comparação salva otimizador, scheduler, scaler, RNG e época em
`last.pt`. Retoma da próxima época concluída, refazendo uma época incompleta.
Arquiteturas que já têm `test.json` são reutilizadas. Não altera silenciosamente
hiperparâmetros de uma experiência existente: use outro `--work-dir` para mudar
épocas, resolução, batch, LR ou configuração do modelo.

Para só baixar e preparar: `--prepare-only`. O padrão agora é `--seeds 42 43 44`:
seis treinos, dois por seed, mantendo a mesma divisão de dados. Use `--seeds 42`
para executar apenas dois treinos. Não é necessário alterar os arquivos de
configuração. A retomada continua compatível com os checkpoints anteriores.

### Resultados gerados

```text
work-dir/
  downloads/nyu_depth_v2_labeled.mat
  downloads/splits.mat
  split.json
  plan.json
  runs/da3-seed42/{run.json,metrics.jsonl,last.pt,best.pt,test.json}
  runs/pixel_depth-seed42/{run.json,metrics.jsonl,last.pt,best.pt,test.json}
  comparison.csv
  comparison.json
  comparison.md
  visuals/index.html
  visuals/manifest.json
  visuals/seed42/sample_XXXX.png
  visuals/seed43/sample_XXXX.png
  visuals/seed44/sample_XXXX.png
```

`comparison.csv` contém uma linha por arquitetura/seed. `comparison.json` inclui
médias e desvios entre seeds, além da melhoria percentual do Pixel-Depth em
AbsRel bruto (positivo significa menor erro). Também registra parâmetros,
passos de otimizador, tempo de treino/teste e pico de memória CUDA.

Os checkpoints de comparação usam formato próprio para identificar o backbone;
carregue-os assim, em vez de `PixelDepth.load`:

```python
from pixel_depth.comparison_models import load_comparison_checkpoint
model = load_comparison_checkpoint('/dados/experimento/runs/da3-seed42/best.pt', 'cuda')
```

Nenhum download de 3 GB nem treinamento real foi executado durante a implementação.
Os testes do pipeline usam um MAT sintético pequeno com o layout original.


### Reaproveitar a seed 42 e adicionar as outras

Depois de atualizar esta pasta no servidor e instalar o extra atualizado:

```bash
cd ~/Pixel-depth/pixel-depth
source .venv/bin/activate
python -m pip install --no-cache-dir -e '.[comparison]'

bash scripts/compare_nyu.sh \
  --work-dir ~/dados/comparacao-pixel-depth \
  --device cuda --epochs 30 --gradient-checkpointing \
  --seeds 42 43 44 --resume
```

Preserve os mesmos argumentos do primeiro treino (incluindo batch, precisão,
resolução e taxa de aprendizado, se foram alterados). `--resume` aproveita os
`test.json` concluídos da seed 42 e executa apenas as rodadas que faltam.
O relatório final contém as três seeds; positivo na melhoria percentual em
AbsRel significa vantagem do Pixel-Depth. `comparison.md` traz uma tabela de
média ± desvio padrão amostral, e `comparison.json` também registra a diferença
pareada por seed. Isso não constitui um teste de significância estatística.

### Inspeção visual sem treinar novamente

Para visualizar somente os dois checkpoints que você já treinou:

```bash
bash scripts/compare_nyu.sh \
  --work-dir ~/dados/comparacao-pixel-depth \
  --device cuda --seeds 42 --visuals-only --visual-samples 8
```

A exportação usa os melhores checkpoints segundo a validação, lê a resolução
de entrada salva nos checkpoints e **não executa nenhum passo de treino**.
Se você mudou a divisão de validação originalmente, repita também os argumentos
`--split-seed` e `--val-fraction` originais. O modo visual não sobrescreve o plano
de treinamento: escreve `visual_plan.json`.

Abra `~/dados/comparacao-pixel-depth/visuals/index.html` no navegador após copiar
os resultados para sua máquina. Cada PNG mostra:

- RGB original, profundidade real, predição DA3 e predição Pixel-Depth.
- Máscara válida e erro absoluto de cada arquitetura.
- A seed, o índice da amostra e as épocas dos checkpoints selecionados.

Os exemplos são sorteados deterministicamente do teste usando `--split-seed`,
com os mesmos índices para todas as seeds, sem selecionar melhores ou piores
casos pelo desempenho. A escala de profundidade é fixa em **0–10 m**, e a de
erro em **0–2 m**, com saturação acima desse limite. Pixels inválidos aparecem
pretos; não há ajuste de escala por imagem ou alinhamento por mediana.
As predições usam o regime AMP salvo quando executadas em CUDA, ou FP32 em CPU.
`manifest.json` registra os exemplos e o AbsRel individual. Os números da galeria
não substituem as métricas calculadas em todo o teste.

A exportação acontece automaticamente depois dos treinos; use
`--visual-samples 0` para desativá-la. Um modelo de cada vez ocupa a GPU durante
a geração. A galeria não é um critério adicional de seleção de checkpoint e
não deve ser usada para ajustar hiperparâmetros a partir do teste.
