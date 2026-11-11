import os
import json
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DA CHAVE GEMINI
# ------------------------------------------------------------------------------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not GEMINI_API_KEY:
    raise ValueError("[Fix-It] Erro: A variável GEMINI_API_KEY não foi encontrada no arquivo .env.")

print(f"[Fix-It] Chave Gemini configurada (Início: {GEMINI_API_KEY[:8]}... | {len(GEMINI_API_KEY)} caracteres).")

CLASSES_ZELADORIA = {
    0: "buraco_asfalto",
    1: "iluminacao_defeituosa",
    2: "bueiro_obstruido",
    3: "descarte_irregular_lixo"
}


# ------------------------------------------------------------------------------
# 2. VISÃO COMPUTACIONAL (YOLOv8)
# ------------------------------------------------------------------------------
class TriagemVisualFixIt:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.30):
        self.conf_threshold = conf_threshold
        print("[Fix-It] Inicializando modelo YOLOv8...")
        self.model = YOLO(model_name)

    def analisar_imagem(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        altura, largura, _ = img.shape

        results = self.model.predict(source=img_rgb, conf=self.conf_threshold, verbose=False)[0]

        deteccoes = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()

            rotulo = CLASSES_ZELADORIA.get(cls_id % len(CLASSES_ZELADORIA), "avaria_urbana")
            deteccoes.append({
                "classe": rotulo,
                "confianca": round(conf, 4),
                "box": [round(coord, 1) for coord in xyxy]
            })

        return {
            "resolucao": f"{largura}x{altura}",
            "total_anomalias": len(deteccoes),
            "deteccoes": deteccoes
        }


# ------------------------------------------------------------------------------
# 3. IA GENERATIVA (GOOGLE GEMINI VIA SDK OFICIAL GOOGLE-GENAI)
# ------------------------------------------------------------------------------
class DespachoGenerativoFixIt:
    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite"):
        self.api_key = api_key
        self.model = model
        # Inicializa o cliente oficial Google GenAI (autentica chaves AQ. nativamente via x-goog-api-key)
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                api_version="v1beta",
                timeout=15000,
                retry_options=types.HttpRetryOptions(attempts=2),
            ),
        )

    def gerar_parecer_tecnico(self, dados_visuais: dict, dados_contexto: dict) -> dict:
        if dados_visuais.get("total_anomalias", 0) == 0:
            return {
                "status_triagem": "REPROVADO_AUTOMATICO",
                "motivo": "Nenhuma evidência de anomalia urbana detectada com confiança mínima.",
                "acao": "Solicitar nova foto ao munícipe com melhor ângulo e iluminação.",
                "criticidade": "NENHUMA"
            }

        prompt = f"""
        Você é o motor de inteligência e triagem da plataforma Fix-It (CitizenTech).
        Analise a anomalia urbana detectada e gere o despacho de zeladoria pública em JSON estrito.

        DADOS DA OCORRÊNCIA:
        - Endereço aproximado: {dados_contexto.get('endereco', 'Não informado')}
        - Coordenadas: Lat {dados_contexto.get('lat')}, Long {dados_contexto.get('long')}
        - Relato do Cidadão: "{dados_contexto.get('relato')}"
        - Detecções da Visão Computacional (YOLO): {json.dumps(dados_visuais.get('deteccoes', []), ensure_ascii=False)}

        SCHEMA DO JSON OBRIGATÓRIO:
        {{
            "status_triagem": "APROVADO",
            "diagnostico_tecnico": "Resumo técnico do problema identificado",
            "nivel_criticidade": "BAIXA | MEDIA | ALTA | CRITICA",
            "risco_populacao": "Riscos potenciais a veículos e pedestres",
            "secretaria_responsavel": "Nome da secretaria competente (ex.: Obras, Iluminação Pública)",
            "prazo_estimado_resolucao_horas": 48,
            "necessidade_interdicao": false,
            "equipe_recomendada": "Tipo de equipe técnica necessária para o reparo"
        }}
        """

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            return json.loads(response.text)

        except Exception as e:
            # Fallback determinístico local em caso de falha de conexão ou indisponibilidade da API
            classe = dados_visuais["deteccoes"][0]["classe"]
            return {
                "status_triagem": "APROVADO_LOCAL_FALLBACK",
                "diagnostico_tecnico": f"Classificado preventivamente como {classe}.",
                "nivel_criticidade": "ALTA" if "buraco" in classe else "MEDIA",
                "risco_populacao": "Risco à circulação viária e integridade física de pedestres.",
                "secretaria_responsavel": "Secretaria Municipal de Obras e Serviços Públicos",
                "prazo_estimado_resolucao_horas": 48,
                "necessidade_interdicao": False,
                "equipe_recomendada": "Equipe padrão de manutenção viária",
                "aviso_resiliencia": f"Fallback ativado: {str(e)}"
            }


# ------------------------------------------------------------------------------
# 4. EXECUÇÃO DOS CENÁRIOS DE TESTE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== FIX-IT: PIPELINE DE IA E TRIAGEM MUNICIPAL ===")

    contexto_chamado = {
        "lat": -23.561684,
        "long": -46.655981,
        "endereco": "Av. Paulista, 1106 - Bela Vista, São Paulo - SP",
        "relato": "Buraco profundo na faixa de ônibus após chuva torrencial."
    }

    detector = TriagemVisualFixIt(conf_threshold=0.25)
    gerador = DespachoGenerativoFixIt(api_key=GEMINI_API_KEY)

    # CENÁRIO 1: Teste Positivo (Avaria detectada)
    print("\n" + "="*50)
    print(">>> EXECUTANDO CENÁRIO 1: DETECÇÃO DE AVARIA URBANA (CAMINHO FELIZ)")
    print("="*50)

    resultado_visual_positivo = {
        "resolucao": "1920x1080",
        "total_anomalias": 1,
        "deteccoes": [
            {
                "classe": "buraco_asfalto",
                "confianca": 0.8842,
                "box": [120.5, 340.0, 480.2, 590.8]
            }
        ]
    }
    print("[Etapa 1] Resultado da Visão Computacional:")
    print(json.dumps(resultado_visual_positivo, indent=2, ensure_ascii=False))

    print("\n[Etapa 2] Parecer Técnico da IA Generativa (Google Gemini):")
    parecer_positivo = gerador.gerar_parecer_tecnico(resultado_visual_positivo, contexto_chamado)
    print(json.dumps(parecer_positivo, indent=2, ensure_ascii=False))

    # CENÁRIO 2: Teste Negativo (Sem anomalia)
    print("\n" + "="*50)
    print(">>> EXECUTANDO CENÁRIO 2: IMAGEM SEM ANOMALIA (CASO DE BORDA / FALHA)")
    print("="*50)

    resultado_visual_negativo = {
        "resolucao": "640x640",
        "total_anomalias": 0,
        "deteccoes": []
    }
    print("[Etapa 1] Resultado da Visão Computacional:")
    print(json.dumps(resultado_visual_negativo, indent=2, ensure_ascii=False))

    print("\n[Etapa 2] Parecer Técnico da IA Generativa (Google Gemini):")
    parecer_negativo = gerador.gerar_parecer_tecnico(resultado_visual_negativo, contexto_chamado)
    print(json.dumps(parecer_negativo, indent=2, ensure_ascii=False))