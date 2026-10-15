# TF3 MetalFX: upscaling temporal da Apple no Transport Fever 3 (macOS)

Este mod troca o upscaler do Transport Fever 3 no Mac (o FSR 1 do jogo) pelo **MetalFX Temporal**, o
upscaler temporal da Apple, que é o mais próximo de um DLSS disponível em Macs com Apple Silicon. A imagem
fica muito mais limpa: bordas sem serrilhado, linhas finas contínuas e texturas mais nítidas, inclusive
com o jogo renderizando em 50-67% da resolução.

É um fork, para macOS, do projeto [tf3-dlss](https://github.com/lucasmokwa/tf3-dlss) de Lucas (licença
MIT), que faz o mesmo no Windows com DLSS e FSR 3.1. Não é oficial e não tem ligação com a Urban Games,
a Apple, a NVIDIA ou a AMD.

---

## 1. Requisitos

- Mac com Apple Silicon (M1 ou mais novo). Testado num Apple M4 com macOS 27.
- Transport Fever 3 da Steam, na versão nativa para Mac (a que usa Vulkan via MoltenVK, o padrão).
- Para compilar a partir do código: Xcode Command Line Tools (`xcode-select --install`) e
  [Homebrew](https://brew.sh). Quem usa o zip pronto não precisa de nada disso.

## 2. Instalação

### 2.1 Com o zip pronto (para jogadores)

1. Baixe `TF3-MetalFX-macOS-<versão>.zip` na página **Releases** do repositório e descompacte.
2. Abra o Terminal na pasta `TF3-MetalFX` descompactada e rode:
   ```sh
   ./install.sh
   ```
   O instalador copia os arquivos para `~/.local/share/tf3-metalfx`. Ele **não mexe na pasta do jogo**.
3. Na Steam: botão direito em **Transport Fever 3 → Propriedades → Geral → Opções de inicialização** e
   cole a linha que o instalador mostrou no final, por exemplo:
   ```
   /Users/SEU_USUARIO/.local/share/tf3-metalfx/launch.sh %command%
   ```
4. No jogo, em **Opções → Gráficos**: escala de resolução entre **50% e 67%** e **MSAA desligado**.

### 2.2 A partir do código (para quem quer compilar)

```sh
git clone https://github.com/AltairTF/tf3-metalfx.git
cd tf3-metalfx
macos/build.sh      # instala as dependências pelo Homebrew e compila (build-macos/bin)
macos/install.sh    # instala em ~/.local/share/tf3-metalfx e mostra a opção de inicialização
```

Depois siga os passos 3 e 4 da seção anterior.

### 2.3 Desinstalar

- Para só desligar: apague o texto das opções de inicialização da Steam. O jogo volta ao normal.
- Para remover tudo: `./install.sh --uninstall`. Suas configurações ficam salvas em
  `~/tf3_dlss.cfg.backup`.

## 3. Uso

- Ao abrir o jogo, o MetalFX já está ligado (`upscaler=auto` escolhe o MetalFX no Mac).
- Atalhos (com o jogo em primeiro plano):

  | Atalho | Efeito |
  |---|---|
  | **Ctrl+Shift+D** | MetalFX liga/desliga (desligado = upscaler do próprio jogo, bom para comparar) |
  | **Ctrl+Shift+O** | Caixa de informação: upscaler, resoluções, tempo do MetalFX na GPU, nitidez, FPS |
  | **Ctrl+Shift+↑ / ↓** | Nitidez +/- 0,1 |
  | **Ctrl+Shift+S** | Tipo de nitidez: RCAS (padrão) ou CAS (a da versão 0.1), para comparar |
  | **Ctrl+Shift+A** | Nitidez extra no que se move: liga/desliga |
  | **Ctrl+Shift+R** | Máscara reativa: ligada → ligada e mostrada em vermelho → desligada |
  | **Ctrl+Shift+F** | Geração de frames liga/desliga (se o jogo abriu com `frame_gen=1`) |
  | **Ctrl+Shift+G** | Grava um frame gerado e os vizinhos em `mvcap/`, ~3 s depois (para relatar problemas) |

- As configurações ficam em `~/.local/share/tf3-metalfx/tf3_dlss.cfg`. Cada linha está explicada no
  próprio arquivo, e as mudanças valem assim que o arquivo é salvo, mesmo com o jogo aberto.
  As mais úteis:
  - `enabled=0/1`: liga ou desliga o upscaler.
  - `sharpness=0.5`: nitidez depois do MetalFX (0 a 1), com o RCAS da AMD: não cria halos nas bordas.
  - `motion_sharpness=0.4`: nitidez extra para o que se move (o MetalFX suaviza o movimento);
    0 desliga.
  - `overlay=1`: mostra a caixa de informação ao iniciar.
  - `mip_bias=auto`: deixa as texturas nítidas apesar da resolução menor (vale após reiniciar o jogo).
  - `reactive=1`: máscara reativa (padrão; macOS 14.4+). Onde o jogo desenha fumaça, partículas,
    chuva, pessoas, animais e água, que se movem sem vetores de movimento, o MetalFX passa a confiar
    mais no frame novo e menos nos anteriores, e o rastro some. A força por tipo fica em
    `reactive_particles`, `reactive_animated`, `reactive_water` e `reactive_transparent` (0 a 1);
    `reactive_debug=1` pinta de vermelho onde a máscara age. Custa ~0,7 ms no M4.
  - `upscale_to=1080`: o MetalFX gera no máximo essa altura e o jogo estica o resto; a interface
    continua na resolução da tela (útil numa tela 4K com escala baixa).
  - `fps_limit=30`: trava o jogo em 30 fps com vsync. Numa tela de 60 Hz cada frame fica exatamente
    2 atualizações na tela, então o movimento fica regular (sem os pulos e o tearing de 37-47 fps).
    Acrescenta ~1 frame de atraso. Vale após reiniciar o jogo; 0 desliga.
  - `frame_gen=1` (experimental): geração de frames, explicada logo abaixo. Vale após reiniciar o jogo.

### Geração de frames (experimental)

Com `frame_gen=1`, o MetalFX cria um frame intermediário entre cada dois frames do jogo e o mostra
entre eles, então o movimento fica duas vezes mais fluido. A interface do jogo não é interpolada:
o MetalFX a separa da imagem e a coloca por cima do frame gerado.

- Precisa do macOS 26 ou mais novo e do MetalFX como upscaler.
- Liga o vsync. Cada frame do jogo vira dois na tela, então o jogo roda no máximo na metade da
  taxa do monitor: num monitor de 60 Hz, 30 fps do jogo viram 60 na tela.
- Custa cerca de 6-8 ms de GPU por frame num M4 a 2560x1440 (medido fora do jogo; a caixa de
  informação mostra "FG x MS"). Se o jogo cair abaixo de 30 fps, a tela fica abaixo de 60 e o
  movimento perde regularidade.
- Acrescenta cerca de um frame de atraso entre o mouse e a tela.
- A caixa de informação do mod faz parte da imagem que o MetalFX desloca, então ela treme quando
  a câmera se move. A interface do próprio jogo fica parada.

### Desempenho medido (Apple M4, saída 2560x1440)

| Escala do jogo | Resolução interna | Custo do MetalFX na GPU | FPS observado com MetalFX |
|---|---|---|---|
| 50% | 1280x720 | ~3,4 ms | ~30-57 (depende da cena) |
| 65-67% | ~1700x950 | ainda não medido no jogo | ~25 |

O custo de 50% foi medido no jogo, em ~8.600 frames (3,3-3,5 ms na maior parte do tempo). Os outros passos do mod
somam ~1 ms: nitidez ~0,7 ms, vetores de movimento ~0,3 ms. O número da sua máquina aparece na
caixa de informação (Ctrl+Shift+O, "GPU x MS" ao lado das resoluções). O custo do MetalFX cresce com a
resolução interna. Por isso **50-58%** dá o melhor equilíbrio: a imagem com MetalFX a 50% já fica mais
limpa que a de 65-75% sem ele.

## 4. Como funciona

O Transport Fever 3 não tem os dados de que um upscaler temporal precisa (vetores de movimento e
*jitter* de câmera). O mod cria esses dados por fora, como uma **layer Vulkan**: um módulo que o
carregador Vulkan do jogo coloca entre o jogo e o driver (MoltenVK) e que vê todos os comandos de
desenho.

```
Jogo (Vulkan) ──► [layer TF3 MetalFX] ──► MoltenVK (Vulkan sobre Metal) ──► GPU Apple
                        │
                        └─► fila Metal própria: MetalFX Temporal
```

A cada frame a layer:

1. **Reconhece os passes do jogo** pelos nomes de depuração dos shaders: a cena 3D, o tonemap, o FSR 1
   (EASU e RCAS) e o blit final para a tela.
2. **Aplica jitter**: desloca a câmera em uma fração de pixel, num padrão Halton, a cada frame. É isso
   que permite ao upscaler recuperar detalhe que um único frame em resolução menor não tem.
3. **Calcula os vetores de movimento da câmera** num compute shader, a partir do depth e das matrizes da
   câmera deste frame e do anterior (a reprojeção é montada em double precision na CPU).
4. **Calcula o movimento dos objetos** (trens, ônibus, carros): redesenha os modelos que mudaram de
   posição com a matriz do frame anterior.
5. **Entrega tudo ao MetalFX**: cor, depth (cópia R32F), vetores de movimento (RG16F) e jitter. A saída,
   em resolução nativa, substitui a imagem do FSR 1 no blit final, antes da interface do jogo.

### Sincronização Vulkan ↔ Metal

O MetalFX é uma API Metal e o jogo desenha em Vulkan. Para o MetalFX rodar **no meio** do command buffer
do jogo, a layer:

- grava `vkCmdSetEvent(entrada)` logo depois de preparar as entradas e `vkCmdWaitEvents(saída)` antes de
  qualquer leitura da saída;
- exporta esses `VkEvent` como `MTLSharedEvent` (extensão `VK_EXT_metal_objects` do MoltenVK);
- no `vkQueueSubmit` do jogo, ajusta o valor dos eventos e envia, numa **fila Metal própria**:
  esperar(entrada) → MetalFX → sinalizar(saída).

O MoltenVK calcula os valores desses eventos na hora em que codifica o command buffer (no envio), e a
layer usa exatamente os mesmos valores. Se algo falhar, a layer sinaliza a saída pela CPU para a GPU do
jogo nunca ficar esperando para sempre.

### Por que uma opção de inicialização da Steam

Camadas Vulkan são ativadas por variáveis de ambiente (`VK_ADD_LAYER_PATH`, `VK_INSTANCE_LAYERS`). A
Steam no macOS não aceita `VAR=valor %command%` nas opções de inicialização (dá "OS Error 260"). Por
isso o `launch.sh` define as variáveis só para o processo do jogo e então abre o jogo. Nada é instalado
no sistema nem na pasta do jogo.

## 5. Solução de problemas

| Sintoma | O que fazer |
|---|---|
| "OS Error 260" ao abrir pela Steam | A opção de inicialização deve ser só `.../launch.sh %command%`, sem `VAR=` antes. |
| O jogo abre igual, sem mudança | Veja `~/.local/share/tf3-metalfx/tf3_dlss.log`. Se existir `TF3-DLSS-DID-NOT-START.txt` ao lado, ele diz o motivo. Confira se a escala de resolução está abaixo de 100%. |
| Não aparece nenhum log | A layer não carregou: confira o caminho na opção de inicialização e rode `xattr -dr com.apple.quarantine ~/.local/share/tf3-metalfx`. |
| Imagem tremendo com a câmera parada | Teste `mfx_jitter_sign_x=-1` (e/ou `_y`) no cfg e informe o resultado. |
| Rastros atrás de veículos | Confirme `motion_vectors=1` e `object_motion=1`. Pessoas e animais só recebem o movimento da câmera. |
| FPS baixo | Use escala de 50-58% e desligue o MSAA. |
| Quer comparar | Ctrl+Shift+D alterna com o upscaler do jogo. Para prints use **F12 (Steam)**: o F9 do próprio jogo renderiza um frame separado, sem upscaler. |

## 6. Como compartilhar com outras pessoas

### 6.1 Repositório

O código fica em `github.com/AltairTF/tf3-metalfx`. Ele foi criado como **privado**. Opções:

- **Compartilhar com algumas pessoas:** GitHub → Settings → Collaborators → *Add people*, ou
  `gh repo edit AltairTF/tf3-metalfx --add-collaborator USUARIO` (requer gh recente; o caminho pelo site
  sempre funciona).
- **Tornar público:** `gh repo edit AltairTF/tf3-metalfx --visibility public --accept-visibility-change-consequences`.

### 6.2 Publicar uma versão pronta (Release)

Quem só quer jogar não deve precisar compilar. Para gerar o zip:

```sh
macos/package.sh
# -> dist-macos/TF3-MetalFX-macOS-<versão>.zip
gh release create v0.1.0-macos dist-macos/TF3-MetalFX-macOS-*.zip \
   --title "TF3 MetalFX 0.1.0 (macOS)" --notes "MetalFX Temporal para Transport Fever 3 no Mac."
```

O zip contém a layer (assinada ad-hoc), o manifesto, o `tf3_dlss.cfg`, o `launch.sh`, o `install.sh`, este
tutorial (como `LEIA-ME.md`) e as licenças. Quem recebe só precisa seguir a seção 2.1.

### 6.3 Regras para distribuir

- **Nunca** inclua arquivos do jogo (shaders, binários, saves) nem capturas de frames (`mvcap`).
- Mantenha o crédito e a licença MIT do projeto original (`LICENSE`) e as licenças em `package/licenses`.
  O build inclui headers da NVIDIA (só para desativar o DLSS no Mac), por isso a licença do SDK vai junto.
- Deixe claro que o mod não é oficial e que foi feito com a ajuda de IA (Claude).
- Lugares para divulgar: o próprio GitHub, o fórum oficial da Urban Games / Transport Fever, Reddit
  (r/TransportFever) e Nexus Mods. O Steam Workshop não serve, porque o mod não é um mod de conteúdo do jogo.

### 6.4 Devolver ao projeto original

A correção dos vetores de movimento com MSAA (seção 7) vale também para o Windows. Um pull request para o
[tf3-dlss](https://github.com/lucasmokwa/tf3-dlss) ajuda os dois lados:

```sh
gh repo fork lucasmokwa/tf3-dlss --clone=false
git remote add fork git@github.com:AltairTF/tf3-dlss.git
git push fork macos-metalfx
gh pr create --repo lucasmokwa/tf3-dlss --head AltairTF:macos-metalfx --title "macOS port (MetalFX) + MSAA depth fix"
```

## 7. Para desenvolvedores

### Estrutura do que foi adicionado

| Caminho | O que é |
|---|---|
| `macos/CMakeLists.txt` | Build da layer como dylib arm64 (Apple clang). |
| `macos/metalfx_bridge.{h,mm}` | Interface C sobre o MetalFX: fila própria, scaler, eventos, atalhos. |
| `macos/launch.sh` | Wrapper para a opção de inicialização da Steam. |
| `macos/build.sh`, `install.sh`, `package.sh` | Compilar, instalar e empacotar. |
| `macos/tf3_dlss.cfg` | Configuração padrão para jogadores no Mac. |
| `layer/src/layer.cpp` | Backend `MetalFx`, hand-off por eventos, correção do MSAA e ajustes de macOS (`#ifdef __APPLE__`). |
| `layer/shaders/camera_mv.comp` | Variante `DEPTH_MS` (depth multisample). |
| `harness/fake_tf3.cpp` | Opção `--msaa N`, enumeração de portabilidade (MoltenVK). |
| `tools/mvcap_check.py` | Analisa capturas de frames (Python puro, sem numpy). |
| `docs/MACOS_PORT.md` | Diário técnico completo, com todas as medições. |

### Testar sem o jogo (harness)

```sh
TF3_GAME_DIR="/caminho/para/Transport Fever 3" macos/build.sh --harness
G="$TF3_GAME_DIR"; mkdir -p logs/h && cd logs/h && cp ../../tf3_dlss.cfg .
DYLD_LIBRARY_PATH="$G" VK_DRIVER_FILES="$G/MoltenVK_icd.json" \
VK_ADD_LAYER_PATH="$PWD/../../build-harness/layer" VK_INSTANCE_LAYERS=VK_LAYER_TF3_dlss \
TF3_DLSS_FORCE=1 TF3_DLSS_CFG="$PWD/tf3_dlss.cfg" TF3_DLSS_LOG="$PWD/tf3_dlss.log" \
TF3_DLSS_CAPTURE=60,140,220 ../../build-harness/fake_tf3 --msaa 2
```

O `fake_tf3` renderiza frames com a mesma estrutura do TF3 (cena, tonemap, EASU, RCAS, blit, UI) e
confere os vetores de movimento com valores exatos. Com `TF3_DLSS_TEST_UPSCALER=1` ele usa um upscaler
bilinear de teste no lugar do MetalFX.

### Variáveis de ambiente úteis

Elas podem ir num arquivo `launch.env` ao lado do `launch.sh` (uma `VAR=valor` por linha).

| Variável | Efeito |
|---|---|
| `TF3_DLSS_CAPTURE=900,1800` + `TF3_DLSS_CAPTURE_REL=1` | Grava pares de frames (cor, depth, vetores) em `mvcap/`, contando a partir de quando o mapa aparece. Analise com `python3 tools/mvcap_check.py <pasta>`. |
| `TF3_DLSS_TRACE=300,600` | Uma linha de log por frame com a câmera e os vetores esperados. |
| `TF3_DLSS_TEST_UPSCALER=1` | Upscaler bilinear de teste. |
| `TF3_FG_DUMP=900,901` | Com `frame_gen=1`: grava o frame anterior (sem UI), o gerado e o atual em `mvcap/fg_f*.ppm`. Conta a partir do mapa com `TF3_DLSS_CAPTURE_REL=1`. |
| `TF3_DLSS_MS_DEPTH=0` | Força o caminho antigo de leitura do depth (só para comparar). |

### Bugs encontrados no caminho (e o que ensinam)

1. **MSAA + MoltenVK:** com MSAA ligado, o depth do jogo é multisample. Ler essa imagem como `sampler2D`
   é inválido no Vulkan. O driver da NVIDIA tolera, mas o MoltenVK devolve valores sem relação com a cena,
   e todos os vetores de movimento saíam errados (erro de 10-97 px). Correção: uma variante com
   `sampler2DMS` (amostra 0), e o passe de objetos passa a usar a cópia R32F. Confirmado no jogo: 0,03 px.
2. **Medição enganosa:** gravar as capturas na thread do jogo travava o frame seguinte, e a câmera (que
   anda pelo tempo) dava um salto. Agora as capturas são copiadas para a RAM e gravadas numa thread separada.
3. **Steam no Mac:** variáveis de ambiente nas opções de inicialização dão "OS Error 260". Daí o `launch.sh`.
4. **Abrir o jogo pelo terminal não funciona:** a Steam relança o jogo e as variáveis se perdem.

## 8. Limitações

- Só macOS com Apple Silicon e MoltenVK (o driver Vulkan padrão do jogo no Mac).
- Fumaça, partículas, pessoas, animais e água não têm vetores de movimento próprios; a máscara reativa
  reduz o rastro deles, mas nesses pontos a imagem fica um pouco menos suave.
- Nuvens e outros efeitos de tela inteira recebem só o movimento da câmera.
- O jogo apresenta os frames sem vsync: a 37-47 fps numa tela de 60 Hz o movimento fica irregular.
  `fps_limit=30` resolve isso enquanto o jogo aguentar 30 fps; abaixo disso volta a variar.
- O MetalFX custa ~3,4 ms de GPU por frame no M4 base a 50% (cerca de 15% do frame a 40 fps).
  Chips Pro/Max devem ter folga bem maior.
- Experimental: use por sua conta. Para voltar ao normal, basta apagar a opção de inicialização.

## 9. Créditos

- **tf3-dlss**, de Lucas (MIT): a layer, a detecção de passes, o jitter, os vetores de movimento e o harness.
- **Port para macOS / MetalFX:** AltairTF, com desenvolvimento assistido por IA (Claude, da Anthropic).
- Apple MetalFX, Khronos MoltenVK e Vulkan-Headers (Apache 2.0); headers do NVIDIA NGX (licença do SDK)
  usados só na compilação.
