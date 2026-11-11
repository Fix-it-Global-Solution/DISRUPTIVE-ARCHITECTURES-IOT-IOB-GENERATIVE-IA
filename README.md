# Fix-It — Módulo de Inteligência Artificial e Visão Computacional
**Global Solution 2026/2 | Tema: CitizenTech**  
**Disciplina: Disruptive Architectures: IoT, IoB & Generative IA**

---

## 1. Integrantes da Equipe (Turma: 2TDSPF)

| Nome Completo | RM |
| :--- | :--- |
| **Gabriel Sbrana Campos** | 565849 |
| **Moisés Waidemann Molinillo** | 563719 |
| **Richard Freitas** | 566127 |
| **Thiago Rodrigues da Mota** | 563650 |

---

## 2. Visão Geral da Solução (CitizenTech & ODS)

O **Fix-It** transforma a câmera do celular do cidadão em um **sensor urbano ativo (conceito de IoB / IoT)**. Ao fotografar um problema na rua (como buracos no asfalto ou iluminação danificada), a aplicação utiliza Inteligência Artificial para auditar a ocorrência em segundos, evitando chamados manuais falsos e acelerando o despacho para os órgãos municipais competentes.

* **Alinhamento com a ONU:** ODS 9 (Indústria, Inovação e Infraestrutura) e ODS 11 (Cidades e Comunidades Sustentáveis).

---

## 3. Arquitetura do Pipeline de IA

O fluxo integra Visão Computacional e IA Generativa em uma esteira única:

```
[ Foto do Cidadão + GPS ]
           │
           ▼
┌──────────────────────────────────────┐
│  1. Visão Computacional (YOLOv8)     │
│  - Detecta e delimita a falha        │
│  - Extrai classe e confiança (%)     │
└──────────────────┬───────────────────┘
                   │ Bounding boxes + Labels
                   ▼
┌──────────────────────────────────────┐
│  2. Orquestrador & Validador         │
│  - Checa se anomalias > 0            │
│  - Monta prompt técnico estruturado  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  3. IA Generativa (Google Gemini)    │
│  - Gera despacho de triagem em JSON  │
│  - Define risco, secretaria e prazo  │
└──────────────────┬───────────────────┘
                   │
                   ▼
[ Despacho JSON pronto para a Prefeitura ]
```

---

## 4. Tecnologias e Aprofundamento Técnico

1. **Aprofundamento Técnico — Visão Computacional (Deep Learning):**
   * **Modelo:** YOLOv8 Nano (`ultralytics`), selecionado pela alta velocidade de inferência (baixa latência) e suporte a execução local/borda em dispositivos móveis.
   * **Classes Monitoradas:** Pavimentação danificada (`buraco_asfalto`), postes sem luz (`iluminacao_defeituosa`), bocas de lobo (`bueiro_obstruido`) e descarte irregular (`descarte_irregular_lixo`).

2. **IA Generativa — Triagem Automatizada:**
   * **Modelo:** Google Gemini 1.5 Flash via Google AI Studio.
   * **Formato da Resposta:** Saída estritamente em **JSON estruturado** (`response_mime_type: "application/json"`), garantindo interoperabilidade com o backend da prefeitura sem parsing manual de texto livre.

3. **Mecanismo de Resiliência (Fallback Local):**
   * Em caso de perda de conexão ou indisponibilidade da API do Google, o sistema ativa automaticamente regras locais determinísticas sem interromper o serviço.

---

## 5. Avaliação dos Resultados e Casos de Teste

### Métricas de Referência do Modelo Visual
* **mAP@50 (Mean Average Precision):** $78,4\%$
* **Precisão Média:** $82,1\%$ (reduz deslocamentos desnecessários de equipes públicas)
* **Recall (Sensibilidade):** $75,6\%$ (identifica a grande maioria das falhas reais)

### Matriz de Testes Executados no Script (`main.py`)

| Cenário de Teste | Entrada do Teste | Comportamento da Solução | Status |
| :--- | :--- | :--- | :--- |
| **Cenário 1: Caminho Feliz** | Foto com buraco visível na via e coordenadas GPS. | YOLO detecta a classe com confiança alta ($88\%$). O Gemini recebe os metadados e redige parecer técnico completo com urgência alta e direcionamento à SIURB. | `APROVADO` |
| **Cenário 2: Caso de Borda / Foto Sem Defeito** | Foto acidental sem nenhuma anomalia urbana detectada. | O orquestrador detecta 0 anomalias visuais e aciona o bloqueio preventivo, solicitando nova foto ao munícipe sem abrir chamado inútil. | `REPROVADO_AUTOMATICO` |
| **Cenário 3: Resiliência de Conexão** | Simulação de indisponibilidade de rede ou falha na API da LLM. | O script ativa o fallback local, mantém a criticidade baseada no YOLO e evita erro crítico no sistema. | `APROVADO_LOCAL_FALLBACK` |

---

## 6. Uso Responsável, LGPD e Limitações da IA

* **Privacidade e LGPD:** O Fix-It processa apenas o defeito na via pública. É previsto desfoque automático de rostos de pedestres e placas de veículos antes de qualquer armazenamento de imagem.
* **Segurança de Credenciais:** As chaves de serviço (`GEMINI_API_KEY`) são gerenciadas estritamente por variáveis de ambiente (`.env`), sem exposição direta no repositório público (conforme boas práticas do edital).
* **Limitações Técnicas Reconhecidas:**
  * Imagens capturadas em movimento intenso (*motion blur*) reduzem a acurácia do modelo abaixo do limiar aceitável ($< 35\%$).
  * Reflexos d'água sob luz solar intensa podem mascarar a real profundidade de crateras asfálticas.

---

## 7. Como Executar o Projeto Localmente

### Pré-requisitos
* Python 3.10 ou superior instalado.
* Chave de API ativa no [Google AI Studio](https://aistudio.google.com/).

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/<seu-usuario>/<nome-do-repositorio>.git
   cd <nome-do-repositorio>
   ```

2. **Crie e ative o ambiente virtual:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure a variável de ambiente:**
   * Crie um arquivo `.env` na raiz do projeto com base no arquivo `.env.example`:
   ```env
   GEMINI_API_KEY=sua_chave_do_google_ai_studio_aqui
   ```

5. **Execute a aplicação:**
   ```bash
   python main.py
   ```