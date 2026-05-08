# Relatório — Melhoria da Generalização do Agente CPP

## 1. Introdução e Problema

O ambiente de **Coverage Path Planning (CPP)** exige que o agente visite todas as células livres de um grid sem acesso ao mapa completo — apenas a uma visão local ao redor de sua posição atual. O agente de referência (V1), treinado com PPO no ambiente 5×5 com visão 3×3, apresenta baixa generalização para o 10×10:

| Ambiente | Full Coverage Rate | Cobertura Média |
|----------|--------------------|-----------------|
| 5×5      | 76/100 (76%)       | ~93%            |
| 10×10    | 64/100 (64%)       | ~82%            |

A análise do comportamento identificou três causas principais de falha:

| # | Causa | Correção |
|---|-------|----------|
| 1 | Agente treinado apenas no 5×5 não converge no espaço de estados maior do 10×10 | **Aprendizado por currículo** (transfer learning 5×5 → 10×10) |
| 2 | Penalidade de revisita desencoraja backtracking necessário para sair de becos | Eliminar penalidade de **−0.3** |
| 3 | Células livres isoladas por obstáculos tornam cobertura 100% impossível e corrompem o sinal de recompensa | **Flood fill** no `reset()` |

Uma quarta melhoria foi adicionada após análise dos resultados intermediários:

| # | Causa | Correção |
|---|-------|----------|
| 4 | Visão 3×3 insuficiente — agente não enxerga células não visitadas a mais de 1 passo | Expandir observação para **5×5** |

---

## 2. Estratégia Adotada

### 2.1 Aprendizado por Currículo (Curriculum Learning)

O agente é treinado em três fases progressivas, sempre usando os pesos da fase anterior como ponto de partida (transfer learning via `PPO.load()`):

- **Phase 1 (5×5):** aprendizado do comportamento básico de cobertura num grid pequeno com poucos obstáculos.
- **Phase 2 (10×10):** fine-tuning no ambiente alvo, partindo da política aprendida na Phase 1. As observações são normalizadas por `size` (posição `x/size`, `y/size`), o que permite que os pesos sejam reutilizados entre tamanhos diferentes sem reinterpretar os valores.
- **Phase 3 (10×10, fine-tune):** ajuste fino com episódios mais longos e `gamma` maior para resolver o problema da "última célula".

**Justificativa em RL:** o currículo reduz a variância do gradiente nas fases iniciais. O espaço de estados do 10×10 (~88 células livres) é quatro vezes maior que o do 5×5 (~22 células); partir de uma política já funcional encurta significativamente o tempo de convergência. Sem o currículo, o PPO teria dificuldade em explorar suficientemente o espaço de estados do 10×10 para aprender uma política de cobertura sistemática.

A normalização das observações é a condição técnica que permite a transferência direta de pesos entre ambientes de tamanhos diferentes: `x/size` e `y/size` sempre ficam no intervalo [0, 1] independente do tamanho do grid, e `coverage_ratio` também é sempre [0, 1]. Isso garante que o que o agente aprendeu no 5×5 seja semanticamente válido no 10×10.

O treino converge rapidamente por três fatores combinados: (1) cada fase parte de uma política já funcional; (2) a recompensa densa fornece gradiente em todo passo, não só no final do episódio; e (3) episódios de cobertura completa terminam cedo, permitindo que o PPO processe muito mais episódios por unidade de tempo.

### 2.2 Espaço de Observação

O agente possui apenas **observação parcial** do ambiente: não tem acesso ao mapa completo, somente às informações coletadas ao longo da exploração.

| Chave       | Dimensão | Conteúdo |
|-------------|----------|----------|
| `agent`     | (3,)     | `[x/size, y/size, coverage_ratio]` — posição normalizada e fração de cobertura acumulada |
| `neighbors` | (5,5)    | visão local 5×5 centrada no agente (agente sempre em (2,2)): 0 = livre/não visitado, 1 = obstáculo/parede, 2 = visitado |

O agente não tem acesso a nenhuma informação global sobre o mapa. A decisão de qual célula visitar a seguir é aprendida inteiramente pela política da rede neural a partir da visão local 5×5.

**Por que 5×5 e não 3×3:** com visão 3×3, o agente enxerga apenas 1 célula em cada direção. Quando todas as células vizinhas imediatas já foram visitadas, o agente não tem nenhuma informação sobre onde estão as células ainda não exploradas, levando a loops e timeouts. Com 5×5, o agente enxerga até 2 células em cada direção — se houver uma célula livre próxima, ele consegue identificá-la e navegar em sua direção sem depender de movimento aleatório.

### 2.3 Função de Recompensa

#### V1 (baseline)

| Condição | Recompensa |
|----------|:----------:|
| Visitar célula nova | +1.0 |
| Revisitar célula já visitada | **−0.3** |
| Colisão com parede ou obstáculo | −0.5 |
| Penalidade por passo (toda ação) | −0.1 |
| Cobertura completa (todas as células livres) | +10.0 |
| Max steps atingido sem cobertura completa | −5.0 |

**Problema identificado:** a penalidade de −0.3 por revisita tornava o backtracking caro. Para sair de um beco atravessando N células visitadas, o agente pagava N × (−0.1 − 0.3) = N × (−0.4), o que desestimulava a saída mesmo quando era a única opção.

#### V2 — Sem penalidade de revisita (versão final)

| Condição | Recompensa |
|----------|:----------:|
| Visitar célula nova | +1.0 |
| Revisitar célula já visitada | **0.0** (apenas penalidade de passo) |
| Colisão com parede ou obstáculo | −0.5 |
| Penalidade por passo (toda ação) | −0.1 |
| Cobertura completa (todas as células livres) | +10.0 |
| Max steps atingido sem cobertura completa | −5.0 |

**Justificativa:** o backtracking é frequentemente necessário em CPP — para sair de um corredor ou retornar a uma área não visitada, o agente precisa atravessar células já exploradas. Penalizar esse comportamento com −0.3 contradiz o objetivo de cobertura completa. Com revisita custando apenas a penalidade de passo (−0.1), o agente pode transitar livremente por células já visitadas.

### 2.4 Correções no Ambiente

**Garantia de início não-cercado:** `reset()` verifica se ao menos um dos quatro vizinhos diretos do agente é acessível. Se o agente estiver completamente cercado (paredes + obstáculos em todas as direções), ele é realocado. Sem essa correção, episódios raros iniciavam com 0% de progresso possível, introduzindo ruído no treinamento.

**Contagem de células alcançáveis (flood fill):** `total_free_cells` conta apenas as células **alcançáveis via flood fill** a partir da posição inicial do agente, excluindo células livres isoladas por obstáculos que nunca podem ser visitadas.

**Justificativa:** o posicionamento aleatório de obstáculos pode, em alguns episódios, criar células livres que são fisicamente inacessíveis — cercadas por obstáculos em todas as direções. Sem o flood fill, o agente receberia −5.0 ao final desses episódios por não ter visitado células que ele **nunca poderia visitar**, independente de qual política aprendesse. Isso não é uma falha do agente — é uma configuração inválida do ambiente.

O flood fill **não exclui episódios nem facilita o problema**: o episódio roda normalmente e o agente ainda precisa visitar fisicamente todas as células alcançáveis. A única mudança é que o critério de sucesso ("cobertura completa") passa a ser definido como "visitou todas as células que pode alcançar a partir de onde está", o que é a única definição matematicamente possível de 100% de cobertura. Sem essa correção, o objetivo seria impossível de atingir em episódios com células isoladas, e a penalidade de −5.0 introduziria ruído espúrio no gradiente do PPO.

Em 2 000 resets por tamanho:

| Grid | Células brutas | Média alcançável | Resets com células isoladas |
|------|:-:|:-:|:-:|
| 5×5  (3 obs.)  | 22 | 21.9 | 5.7% |
| 10×10 (12 obs.) | 88 | 87.6 | 10.7% |
| 20×20 (48 obs.) | 352 | 351.6 | 22.7% |

Apesar de 22.7% dos resets no 20×20 terem alguma célula isolada, a média de células excluídas por episódio é apenas 0.4 — o efeito prático é mínimo, mas garante que a Full Coverage Rate reflita a capacidade real do agente, não artefatos do gerador de obstáculos.

---

## 3. Detalhes de Implementação

### Arquivos criados/modificados

| Arquivo | Papel |
|---------|-------|
| `gymnasium_env/grid_world_cpp_v2.py` | Ambiente CPP-V2: observação 5×5, recompensa sem penalidade de revisita, flood fill, garantia de início não-cercado |
| `train_grid_world_cpp_v2.py` | Script de treinamento com currículo em 3 fases; teste em 5×5, 10×10 e 20×20 |
| `report/relatorio.md` | Este relatório |

### Como executar

```bash
# Treinamento completo (Phase 1 → Phase 2 → Phase 3 automático)
python train_grid_world_cpp_v2.py train

# Testar em 5x5, 10x10 e 20x20 (100 episódios cada)
python train_grid_world_cpp_v2.py test

# Visualizar o agente em um único episódio (escolha 5, 10 ou 20)
python train_grid_world_cpp_v2.py run
```

O treinamento salva um modelo por fase na pasta `data/` (phase1, phase2, phase3). **Apenas o modelo da Phase 3 deve ser usado** para `test` e `run` — ele é o resultado final do currículo completo e contém o comportamento mais refinado. Os modelos das fases anteriores são intermediários e não representam a política final.

### Hiperparâmetros de treinamento

| Parâmetro | Phase 1 (5×5) | Phase 2 (10×10) | Phase 3 (10×10 fine-tune) |
|-----------|:---:|:---:|:---:|
| Tamanho do grid | 5 | 10 | 10 |
| Obstáculos | 3 | 12 | 12 |
| Max passos (treino) | 200 | 400 | 600 |
| Total timesteps | 1 000 000 | 1 000 000 | 500 000 |
| `gamma` | 0.99 | 0.99 | **0.995** |
| `learning_rate` | 3×10⁻⁴ | 3×10⁻⁴ | **1×10⁻⁴** |
| `ent_coef` | 0.05 | 0.05 | **0.02** |

**Justificativa da Phase 3:** após a Phase 2, a maioria dos episódios falhos ocorria com exatamente 1 célula restante quando o limite de 400 passos era atingido. Com `gamma=0.99`, a recompensa de +10.0 vale apenas `10 × 0.99^200 ≈ 0.13` quando a última célula está a 200 passos — o agente racionalmente "desiste". Com `gamma=0.995`, vale `10 × 0.995^200 ≈ 3.7`, criando motivação real para concluir a cobertura. LR e entropia menores protegem o comportamento aprendido nas fases anteriores.

**Avaliação em 20×20 (zero-shot):** o modelo da Phase 3 é testado no ambiente 20×20 (48 obstáculos, max 5 000 passos) sem nenhum treinamento adicional nesse tamanho. Esse teste valida se a política generaliza para ambientes maiores que nunca foram vistos durante o treinamento.

---

## 4. Resultados

### 4.1 Baseline (V1 — sem currículo, visão 3×3, com penalidade −0.3)

Modelo original treinado somente em 5×5, testado sem modificações:

| Ambiente | Full Coverage Rate | Cobertura Média |
|----------|--------------------|-----------------|
| 5×5      | 76/100 (76%)       | ~93%            |
| 10×10    | 64/100 (64%)       | ~82%            |

### 4.2 Progressão das melhorias

Cada linha mostra o efeito acumulativo das melhorias aplicadas:

| Configuração | FCR 5×5 | FCR 10×10 | FCR 20×20 (zero-shot) |
|---|:---:|:---:|:---:|
| Baseline V1 (3×3, penalidade −0.3, sem currículo) | 76% | 64% | — |
| + Currículo + sem penalidade (visão 3×3, max 5000) | 100% | 98% | 83% |
| + Visão 5×5 (**versão final**) | **100%** | **99%** | **93%** |

### 4.3 Resultado final — V2 (visão 5×5 + currículo + sem penalidade + flood fill)

Modelo V2 completo treinado com currículo 3 fases, testado em 100 episódios cada:

| Ambiente | Treinado? | Max Steps | Full Coverage Rate | Cobertura Média | Passos Médios |
|----------|:---------:|:---------:|--------------------|-----------------|:---:|
| 5×5      | Sim       | 200       | **100/100 (100%)** | 100.0% (std=0.0%) | 31.0 (std=11.5, min=21, max=86)      |
| 10×10    | Sim       | 600       | **99/100 (99%)**   | 99.5% (std=4.5%)  | 170.0 (std=95.5, min=97, max=600)    |
| 20×20    | **Não**   | 5000      | **93/100 (93%)**   | 99.3% (std=6.6%)  | 1433.0 (std=1252.7, min=558, max=5000) |

### 4.4 Curvas de aprendizado

```bash
tensorboard --logdir log/
```

---

## 5. Análise dos Resultados

### 5.1 Impacto do currículo

Sem o currículo, o agente treinado no 5×5 generaliza mal para o 10×10 (64% FCR, cobertura média ~82%). O espaço de estados do 10×10 é quatro vezes maior, e o PPO não consegue explorar suficientemente para aprender uma política sistemática partindo do zero. O transfer learning da Phase 1 para a Phase 2 resolve esse problema: o agente já sabe cobrir grids sistematicamente e precisa apenas adaptar o comportamento para o espaço maior.

### 5.2 Impacto da visão 5×5

A expansão de 3×3 para 5×5 foi a mudança mais direta para o 20×20. Com visão 3×3, quando todas as células vizinhas imediatas já foram visitadas, o agente não tinha nenhuma informação sobre onde estão as células restantes e entrava em loop. Com 5×5, o agente enxerga até 2 células em cada direção: se houver uma região não explorada a até 2 passos, ele consegue identificá-la e navegar em sua direção. O FCR no 20×20 subiu de 83% (visão 3×3) para **93%** (visão 5×5) com o mesmo modelo e mesmos hiperparâmetros de treino.

### 5.3 Impacto da remoção da penalidade de revisita

A penalidade −0.3 por revisita criava um dilema: o agente precisava de backtracking para sair de becos, mas cada passo de retorno era penalizado. Com a remoção, o agente pode transitar livremente por células já visitadas sem custo adicional além da penalidade de passo (−0.1). Isso é essencial para CPP, onde regiões isoladas exigem retorno por caminhos já explorados.

### 5.4 Generalização zero-shot para 20×20

O resultado de 93% FCR no 20×20 sem nenhum treinamento nesse tamanho é expressivo para RL puro com observação parcial. A normalização das observações (`x/size`, `y/size`, `coverage_ratio`) é o fator técnico que permite a generalização: essas observações têm a mesma faixa e semântica em qualquer tamanho de grid, então o que o agente aprendeu no 10×10 se aplica diretamente ao 20×20.

Os 7% de falha restantes apresentam desvio padrão alto (6.6%), causado por episódios outliers com coberturas muito baixas (ex: 33.2%). Esses casos correspondem a configurações onde obstáculos criam corredores estreitos logo na posição inicial do agente, confinando-o a uma região pequena por longos períodos antes de conseguir explorar o restante do grid.

### 5.5 Limitações

1. **Visão local limitada:** mesmo com 5×5, o agente não tem como localizar células não visitadas que estejam além de 2 passos. Em grids muito grandes, isso ainda pode levar a loops nas etapas finais da cobertura.

2. **Sem garantia de otimalidade do caminho:** o agente aprende uma política que alcança alta cobertura, mas não o caminho de menor comprimento possível.

3. **Conectividade não garantida:** o placement aleatório de obstáculos pode criar regiões completamente inacessíveis. O flood fill atual exclui essas células do objetivo, mas um ambiente de produção deveria garantir conectividade completa ao gerar obstáculos.

### 5.6 Possíveis melhorias futuras

- **Política recorrente (LSTM):** memória explícita da trajetória permitiria ao agente rastrear regiões não visitadas ao longo do tempo, compensando a ausência de visão global.
- **Treinamento direto no 20×20:** adicionar uma Phase 4 de fine-tune no 20×20 para eliminar os 7% de falha restantes.
- **Garantia de conectividade no `reset()`:** validar que todas as células livres formam um grafo conectado antes de iniciar o episódio.

---

## 6. Conclusão

A combinação de quatro técnicas — **aprendizado por currículo (5×5 → 10×10)**, **visão local 5×5**, **eliminação da penalidade de revisita** e **flood fill + garantia de início** — transformou um agente com 64% FCR no 10×10 em um agente com alta cobertura em todos os tamanhos testados.

A política é aprendida inteiramente por RL (PPO com `MultiInputPolicy`), sem nenhum algoritmo clássico de planejamento guiando as ações em tempo de inferência.

Os resultados finais sobre 100 episódios cada:

| Ambiente | Full Coverage Rate | Cobertura Média | Passos Médios |
|----------|--------------------|-----------------|:---:|
| 5×5      | **100/100 (100%)** | 100.0%          | 31.0 |
| 10×10    | **99/100 (99%)**   | 99.5%           | 170.0 |
| 20×20 (zero-shot) | **93/100 (93%)** | 99.3%  | 1433.0 |

O currículo resolve o problema fundamental de generalização entre tamanhos. A visão 5×5 fornece contexto suficiente para que o agente navegue em direção a regiões não exploradas sem depender de movimento aleatório. A eliminação da penalidade de revisita remove um incentivo contraditório ao backtracking necessário. O resultado zero-shot de 93% no 20×20 — ambiente nunca visto durante o treinamento — confirma que a política aprendida é genuinamente geral.

---
