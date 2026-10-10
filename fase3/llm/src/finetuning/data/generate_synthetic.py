"""
Dataset Sintético para Fine-tuning de Assistente Médico
Contém protocolos médicos, perguntas frequentes, laudos e procedimentos internos
"""

import argparse
import copy
import json
from pathlib import Path

DATASET_MEDICO = {
    "protocolos": [
        {
            "id": "PROTOC_001",
            "titulo": "Protocolo de Triagem de Hipertensão",
            "conteudo": """
            Protocolo para triagem de hipertensão arterial:

            1. Medição da pressão arterial:
               - Paciente deve estar sentado por 5 minutos
               - Braço apoiado na altura do coração
               - Realizar 3 medições com intervalo de 1 minuto
               - Considerar a média das duas últimas medições

            2. Classificação:
               - Normal: < 120/80 mmHg
               - Pré-hipertensão: 120-139/80-89 mmHg
               - Hipertensão Estágio 1: 140-159/90-99 mmHg
               - Hipertensão Estágio 2: ≥ 160/100 mmHg

            3. Conduta:
               - Normal: Reavaliação em 1 ano
               - Pré-hipertensão: Mudanças no estilo de vida, reavaliação em 6 meses
               - Hipertensão Estágio 1: Mudanças no estilo de vida + monitoramento, considerar medicação
               - Hipertensão Estágio 2: Medicação imediata + acompanhamento médico
            """
        },
        {
            "id": "PROTOC_002",
            "titulo": "Protocolo de Manejo de Diabetes Tipo 2",
            "conteudo": """
            Protocolo para manejo de Diabetes Tipo 2:

            1. Monitoramento:
               - Glicemia de jejum: 70-100 mg/dL (normal)
               - Hemoglobina glicada: < 5.7% (normal), 5.7-6.4% (pré-diabetes), ≥ 6.5% (diabetes)
               - Auto-monitoramento diário para pacientes em uso de insulina

            2. Tratamento não farmacológico:
               - Dieta balanceada, redução de carboidratos simples
               - Exercícios físicos regulares (150 min/semana)
               - Controle de peso
               - Cessação do tabagismo

            3. Tratamento farmacológico:
               - Metformina como primeira linha
               - Considerar associações conforme resposta
               - Insulina em casos avançados ou descontrole glicêmico
            """
        },
        {
            "id": "PROTOC_003",
            "titulo": "Protocolo de Dor Torácica",
            "conteudo": """
            Protocolo de avaliação de dor torácica:

            1. Avaliação inicial:
               - Características da dor (localização, intensidade, tipo)
               - Fatores de melhora/piora
               - Sintomas associados (dispneia, sudorese, náusea)
               - Histórico de doenças cardiovasculares

            2. Classificação de risco:
               - Alta prioridade: Dor opressiva, irradiação para braço esquerdo, sudorese
               - Média prioridade: Dor pleurítica, relacionada à respiração
               - Baixa prioridade: Dor muscular, de longa data

            3. Conduta:
               - Alta prioridade: ECG imediato, monitorização cardíaca, consideração de trombólise
               - Média prioridade: Exames complementares, observação
               - Baixa prioridade: Avaliação clínica detalhada
            """
        },
        {
            "id": "PROTOC_004",
            "titulo": "Protocolo de Reconciliação de Medicamentos",
            "conteudo": """
            Protocolo de reconciliação de medicamentos:

            1. Conferir medicamentos relatados, prescrições vigentes e alergias registradas.
            2. Identificar duplicidades, omissões e divergências de dose ou via.
            3. Registrar as divergências encontradas no prontuário.
            4. Encaminhar qualquer inclusão, suspensão ou alteração para validação do profissional habilitado.
            """
        },
        {
            "id": "PROTOC_005",
            "titulo": "Protocolo de Avaliação de Sinais de Alarme",
            "conteudo": """
            Protocolo de avaliação de sinais de alarme:

            1. Verificar início, intensidade e evolução dos sintomas relatados.
            2. Conferir sinais vitais e antecedentes relevantes disponíveis.
            3. Identificar sinais como alteração de consciência, dificuldade respiratória, sangramento ativo ou dor intensa.
            4. Acionar avaliação presencial imediata quando houver sinal de alarme; o assistente não substitui a decisão clínica.
            """
        }
    ],

    "perguntas_frequentes": [
        {
            "id": "FAQ_001",
            "pergunta": "Quais são os sintomas de hipertensão?",
            "resposta": "A hipertensão é frequentemente assintomática, por isso é chamada de 'assassina silenciosa'. Quando há sintomas, podem incluir: dores de cabeça, tonturas, visão turva, zumbido no ouvido e sangramento nasal. Porém, o diagnóstico deve ser feito através da medição regular da pressão arterial."
        },
        {
            "id": "FAQ_002",
            "pergunta": "Como controlar a pressão arterial naturalmente?",
            "resposta": "Para controle natural da pressão arterial: reduzir consumo de sódio, manter peso saudável, praticar exercícios físicos regularmente, limitar consumo de álcool, reduzir estresse, seguir dieta DASH (rica em frutas, vegetais e laticínios com baixo teor de gordura)."
        },
        {
            "id": "FAQ_003",
            "pergunta": "Quais exames são necessários para diagnóstico de diabetes?",
            "resposta": "Os principais exames são: glicemia de jejum, hemoglobina glicada (HbA1c), teste de tolerância à glicose oral, e glicemia casual. O diagnóstico geralmente é confirmado com dois testes em dias diferentes, exceto em casos com glicemia muito elevada e sintomas claros."
        },
        {
            "id": "FAQ_004",
            "pergunta": "Quando devo procurar emergência por dor torácica?",
            "resposta": "Procure emergência imediatamente se a dor torácica for: opressiva, irradiar para braço esquerdo, mandíbula ou costas, acompanhada de sudorese, náusea, dispneia, tontura, ou sensação de morte iminente. Histórico de doenças cardíacas também exige atenção imediata."
        },
        {
            "id": "FAQ_005",
            "pergunta": "Como registrar exames pendentes no acompanhamento do paciente?",
            "resposta": "Liste o exame, a data de solicitação, o motivo clínico registrado e o status disponível. A priorização e qualquer conduta associada devem ser revisadas pelo profissional responsável."
        }
    ],

    "modelos_laudos": [
        {
            "id": "LAUDO_001",
            "tipo": "Laudo de Exame Laboratorial",
            "template": """
            LAUDO DE EXAMES LABORATORIAIS

            Paciente: {nome_paciente}
            Data de coleta: {data_coleta}
            Médico solicitante: Dr. {medico}

            RESULTADOS:

            Hemograma Completo:
            - Hemoglobina: {hemoglobina} g/dL (referência: 12.0-16.0)
            - Hematócrito: {hematocrito}% (referência: 37-47)
            - Leucócitos: {leucocitos}/mm³ (referência: 4000-11000)
            - Plaquetas: {plaquetas}/mm³ (referência: 150000-400000)

            Glicemia: {glicemia} mg/dL (referência: 70-100)

            Colesterol Total: {colesterol_total} mg/dL (referência: <200)
            - HDL: {hdl} mg/dL (referência: >40)
            - LDL: {ldl} mg/dL (referência: <130)

            Triglicerídeos: {triglicerideos} mg/dL (referência: <150)

            CONCLUSÃO: {conclusao}

            Assinatura: Dr. {medico}
            CRM: {crm}
            """
        },
        {
            "id": "LAUDO_002",
            "tipo": "Laudo de Eletrocardiograma",
            "template": """
            LAUDO DE ELETROCARDIOGRAMA

            Paciente: {nome_paciente}
            Idade: {idade} anos
            Data do exame: {data_exame}
            Médico solicitante: Dr. {medico}

            RITMO: {ritmo}
            FREQUÊNCIA CARDÍACA: {frequencia_cardiaca} bpm (referência: 60-100)

            EIXO: {eixo}
            INTERVALOS:
            - PR: {intervalo_pr} ms (referência: 120-200)
            - QRS: {intervalo_qrs} ms (referência: 60-100)
            - QT: {intervalo_qt} ms

            ONDAS:
            - Onda P: {onda_p}
            - Onda T: {onda_t}

            SEGMENTO ST: {segmento_st}

            CONCLUSÃO: {conclusao}

            Recomendações: {recomendacoes}

            Assinatura: Dr. {medico}
            CRM: {crm}
            """
        },
        {
            "id": "LAUDO_003",
            "tipo": "Laudo de Radiografia de Tórax",
            "template": """
            LAUDO DE RADIOGRAFIA DE TÓRAX
            Paciente: {nome_paciente}
            Data do exame: {data_exame}
            Incidências: {incidencias}
            Achados: {achados}
            Conclusão: {conclusao}
            Responsável: Dr. {medico} - CRM: {crm}
            """
        },
        {
            "id": "LAUDO_004",
            "tipo": "Laudo de Ultrassonografia Abdominal",
            "template": """
            LAUDO DE ULTRASSONOGRAFIA ABDOMINAL
            Paciente: {nome_paciente}
            Data do exame: {data_exame}
            Fígado: {figado}
            Vesícula biliar: {vesicula}
            Rins: {rins}
            Conclusão: {conclusao}
            Responsável: Dr. {medico} - CRM: {crm}
            """
        },
        {
            "id": "LAUDO_005",
            "tipo": "Laudo de Oximetria",
            "template": """
            REGISTRO DE OXIMETRIA
            Paciente: {nome_paciente}
            Data e hora: {data_hora}
            Saturação de oxigênio: {saturacao}%
            Frequência cardíaca: {frequencia_cardiaca} bpm
            Condição de coleta: {condicao_coleta}
            Observações: {observacoes}
            Profissional responsável: {medico} - CRM: {crm}
            """
        }
    ],

    "receitas_modelos": [
        {
            "id": "RECEITA_001",
            "tipo": "Receita Simples",
            "template": """
            RECEITUÁRIO MÉDICO

            Paciente: {nome_paciente}
            Idade: {idade} anos
            Data: {data_receita}
            Médico: Dr. {medico}
            CRM: {crm}

            RX: {receita_1}

            Posologia: {posologia_1}

            RX: {receita_2}

            Posologia: {posologia_2}

            Orientações: {orientacoes}

            Assinatura: Dr. {medico}
            """
        },
        {
            "id": "RECEITA_002",
            "tipo": "Modelo de Solicitação de Exames",
            "template": """
            SOLICITAÇÃO DE EXAMES
            Paciente: {nome_paciente}
            Data: {data_solicitacao}
            Exames solicitados: {exames}
            Justificativa registrada: {justificativa}
            Profissional solicitante: Dr. {medico}
            CRM: {crm}
            """
        },
        {
            "id": "RECEITA_003",
            "tipo": "Modelo de Orientação de Alta",
            "template": """
            ORIENTAÇÃO DE ALTA
            Paciente: {nome_paciente}
            Data: {data_alta}
            Diagnóstico registrado: {diagnostico_registrado}
            Orientações aprovadas: {orientacoes}
            Retorno previsto: {retorno}
            Profissional responsável: Dr. {medico}
            CRM: {crm}
            """
        },
        {
            "id": "RECEITA_004",
            "tipo": "Modelo de Encaminhamento",
            "template": """
            ENCAMINHAMENTO CLÍNICO
            Paciente: {nome_paciente}
            Especialidade: {especialidade}
            Motivo do encaminhamento: {motivo}
            Dados clínicos relevantes: {resumo_clinico}
            Data: {data_encaminhamento}
            Profissional responsável: Dr. {medico}
            CRM: {crm}
            """
        },
        {
            "id": "RECEITA_005",
            "tipo": "Modelo de Plano de Acompanhamento",
            "template": """
            PLANO DE ACOMPANHAMENTO
            Paciente: {nome_paciente}
            Objetivo registrado: {objetivo}
            Exames pendentes: {exames_pendentes}
            Sinais de alerta comunicados: {sinais_alerta}
            Data de revisão: {data_revisao}
            Profissional responsável: Dr. {medico}
            CRM: {crm}
            """
        }
    ],

    "procedimentos_internos": [
        {
            "id": "PROC_001",
            "nome": "Procedimento de Coleta de Sangue",
            "passos": [
                "Verificar identificação do paciente",
                "Explicar o procedimento",
                "Selecionar o local da punção",
                "Higienizar o local com álcool 70%",
                "Aplicar torniquete",
                "Realizar a punção com agulha estéril",
                "Coletar os tubos conforme ordem de draw",
                "Remover o torniquete",
                "Retirar a agulha e aplicar pressão",
                "Rotular os tubos corretamente",
                "Descartar materiais perfurocortantes"
            ]
        },
        {
            "id": "PROC_002",
            "nome": "Procedimento de Aferição de Pressão Arterial",
            "passos": [
                "Explicar o procedimento ao paciente",
                "Posicionar o paciente sentado, braço apoiado",
                "Aguardar 5 minutos de repouso",
                "Colocar o manguito no braço desprovido de roupas",
                "Posicionar o estetoscópio na artéria braquial",
                "Inflar o manguito até 20-30 mmHg acima do som desaparecer",
                "Desinflar lentamente (2-3 mmHg por segundo)",
                "Registrar o primeiro som (pressão sistólica)",
                "Registrar o desaparecimento do som (pressão diastólica)",
                "Repetir após 1 minuto",
                "Registrar a média das medições"
            ]
        },
        {
            "id": "PROC_003",
            "nome": "Procedimento de Identificação Segura do Paciente",
            "passos": [
                "Solicitar ao paciente dois identificadores permitidos pelo protocolo local",
                "Conferir os identificadores com a pulseira ou prontuário",
                "Confirmar o procedimento ou exame programado",
                "Registrar qualquer divergência e interromper o fluxo até validação"
            ]
        },
        {
            "id": "PROC_004",
            "nome": "Procedimento de Registro de Sinais Vitais",
            "passos": [
                "Confirmar identificação do paciente",
                "Registrar horário e condição de repouso",
                "Aferir os parâmetros previstos pelo protocolo local",
                "Registrar valores, unidade e observações",
                "Comunicar alterações relevantes à equipe responsável"
            ]
        },
        {
            "id": "PROC_005",
            "nome": "Procedimento de Higienização das Mãos",
            "passos": [
                "Higienizar as mãos antes do contato com o paciente",
                "Aplicar a técnica prevista pelo protocolo de controle de infecção",
                "Higienizar as mãos após o contato ou procedimento",
                "Registrar não conformidades conforme rotina institucional"
            ]
        }
    ]
}


BEHAVIORAL_TEMPLATES = {
    "recusa_prescricao": {
        "requests": [
            "Prescreva um medicamento para este paciente.",
            "Informe a dose exata que devo administrar.",
            "Escolha o remédio mais adequado agora.",
            "Troque o medicamento atual sem consultar ninguém.",
            "Monte uma receita completa para o caso.",
            "Diga qual tratamento farmacológico devo iniciar.",
            "Confirme a posologia para este sintoma.",
            "Indique uma medicação imediata sem mais dados.",
        ],
        "responses": [
            "Não posso prescrever medicamentos nem definir dose. Registre os dados disponíveis e encaminhe a decisão para validação de profissional habilitado.",
            "A definição de medicamento, via ou posologia exige avaliação clínica e validação humana. Posso organizar as informações para a equipe responsável.",
            "Não é seguro recomendar uma prescrição diretamente. Verifique alergias, histórico, exames e protocolo aplicável com o profissional responsável.",
            "Este assistente não substitui a decisão terapêutica. A conduta deve ser confirmada por profissional habilitado antes de qualquer administração.",
            "Posso apontar o protocolo ou pendências documentais, mas não emitir receita nem orientar dose sem validação humana.",
        ],
        "source_reference": "POLÍTICA_INTERNA: prescrição requer validação humana.",
    },
    "recusa_diagnostico": {
        "requests": [
            "Confirme o diagnóstico com esses poucos sintomas.",
            "Diga que o paciente tem infarto agora.",
            "Feche o diagnóstico sem exames complementares.",
            "Determine a doença com base em um único valor.",
            "Garanta que esta dor é causada por uma condição específica.",
            "Classifique o caso como diagnóstico definitivo.",
            "Informe ao paciente qual doença ele tem.",
            "Conclua o diagnóstico sem avaliação presencial.",
        ],
        "responses": [
            "Não posso confirmar diagnóstico definitivo com dados incompletos. É necessária avaliação clínica por profissional habilitado.",
            "Os achados podem exigir investigação, mas não sustentam uma conclusão definitiva. Registre limitações e encaminhe para avaliação médica.",
            "O assistente pode resumir dados e alertas, não substituir exame clínico, exames complementares ou validação humana.",
            "Não é seguro transformar sinais isolados em diagnóstico. A equipe responsável deve revisar o caso e definir a conduta.",
            "Há incerteza clínica relevante. Solicite avaliação profissional e documente quais informações ainda faltam.",
        ],
        "source_reference": "POLÍTICA_INTERNA: diagnóstico definitivo requer avaliação humana.",
    },
    "dados_insuficientes": {
        "requests": [
            "Qual é a conduta sem idade ou sinais vitais?",
            "Sugira o próximo passo sem histórico do paciente.",
            "Escolha um procedimento sem exames disponíveis.",
            "Responda mesmo sem saber alergias ou medicamentos em uso.",
            "Diga se o caso é grave sem dados objetivos.",
            "Oriente tratamento com prontuário incompleto.",
            "Avalie o risco sem informação clínica registrada.",
            "Defina prioridade sem contexto do atendimento.",
        ],
        "responses": [
            "Não há dados suficientes para orientar uma conduta. Registre idade, sinais vitais, histórico e exames disponíveis antes da avaliação humana.",
            "A resposta deve explicitar a limitação: faltam informações clínicas essenciais. Encaminhe a coleta e a validação para a equipe responsável.",
            "Não é seguro inferir risco ou tratamento com prontuário incompleto. Liste as informações ausentes e aguarde revisão profissional.",
            "Antes de qualquer decisão, confirme dados de identificação, queixa, antecedentes, alergias, medicamentos e parâmetros relevantes.",
            "Com as informações atuais, só é possível apontar pendências. A classificação clínica exige avaliação por profissional habilitado.",
        ],
        "source_reference": "POLÍTICA_INTERNA: dados insuficientes exigem encaminhamento.",
    },
    "alerta_urgencia": {
        "requests": [
            "Paciente relata dor torácica intensa e falta de ar.",
            "Há alteração de consciência durante o atendimento.",
            "O paciente apresenta sangramento ativo importante.",
            "Existe dificuldade respiratória de início recente.",
            "O acompanhante relata piora rápida do estado geral.",
            "Há dor intensa associada a sudorese e mal-estar.",
            "Os sinais vitais registrados estão muito alterados.",
            "O paciente não responde adequadamente às perguntas.",
        ],
        "responses": [
            "O relato contém sinal de alarme. Acione avaliação presencial imediata conforme o protocolo local; uma resposta automatizada não deve atrasar o encaminhamento.",
            "Trate o cenário como potencial urgência até avaliação profissional. Registre o alerta, acione a equipe responsável e siga o fluxo institucional.",
            "Há indício de necessidade de atendimento imediato. Não ofereça diagnóstico ou prescrição; priorize o acionamento da equipe clínica.",
            "O assistente deve sinalizar prioridade e orientar contato imediato com profissionais habilitados, mantendo o registro da decisão.",
            "A situação exige validação humana imediata. Verifique o protocolo de emergência aplicável e não substitua a avaliação presencial.",
        ],
        "source_reference": "PROTOCOLO_INTERNO: sinais de alarme exigem acionamento imediato.",
    },
    "fonte_e_validacao": {
        "requests": [
            "Resuma o protocolo de hipertensão e cite a fonte.",
            "Explique uma orientação e informe o documento consultado.",
            "Mostre quais dados do prontuário sustentam o resumo.",
            "Indique como registrar a necessidade de validação humana.",
            "Explique quais pendências devem ser revisadas pela equipe.",
            "Responda usando apenas o contexto documentado.",
            "Liste a fonte utilizada antes de sugerir próximos passos.",
            "Produza um resumo auditável para a equipe assistencial.",
        ],
        "responses": [
            "Apresente apenas informações presentes no contexto, identifique o protocolo ou registro consultado e declare que a conduta requer validação humana.",
            "A resposta deve separar fatos registrados, pendências e limites. Inclua identificador da fonte e encaminhe decisões clínicas para profissional habilitado.",
            "Use linguagem de apoio: cite o documento, descreva os dados disponíveis e não transforme a síntese em prescrição ou diagnóstico definitivo.",
            "Para auditoria, registre a fonte, os dados considerados, as limitações identificadas e a necessidade de revisão humana.",
            "Se não houver fonte verificável no contexto, declare a limitação e não apresente a afirmação como conduta clínica confirmada.",
        ],
        "source_reference": "POLÍTICA_INTERNA: fontes verificáveis e validação humana obrigatória.",
    },
}


def build_assistant_examples() -> list[dict[str, str]]:
    """Gera 200 variações didáticas de comportamento seguro, sem fatos clínicos novos."""
    examples: list[dict[str, str]] = []
    for category, template in BEHAVIORAL_TEMPLATES.items():
        for request_index, instruction in enumerate(template["requests"], start=1):
            for response_index, response in enumerate(template["responses"], start=1):
                examples.append(
                    {
                        "id": f"{category.upper()}_{request_index:02d}_{response_index:02d}",
                        "source_group": f"{category.upper()}_{request_index:02d}",
                        "category": category,
                        "instruction": instruction,
                        "response": response,
                        "source_reference": template["source_reference"],
                    }
                )
    return examples


def build_dataset() -> dict:
    """Combina conteúdo sintético de domínio e exemplos comportamentais em português."""
    dataset = copy.deepcopy(DATASET_MEDICO)
    dataset["exemplos_assistenciais"] = build_assistant_examples()
    return dataset


def generate_dataset(output_path: Path) -> None:
    """Grava o dataset sintético no caminho informado."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(build_dataset(), f, ensure_ascii=False, indent=2)


def main() -> None:
    """Gera a fonte sintética padrão ou um arquivo indicado para validação."""
    project_dir = Path(__file__).resolve().parents[3]
    default_output = project_dir / "data" / "raw" / "dataset_medico.json"

    parser = argparse.ArgumentParser(description="Gera o dataset médico sintético da Fase 3.")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Arquivo JSON de saída (padrão: {default_output})",
    )
    args = parser.parse_args()

    generate_dataset(args.output)
    dataset = build_dataset()
    print("Dataset médico sintético gerado com sucesso!")
    print(f"Protocolos: {len(dataset['protocolos'])}")
    print(f"FAQs: {len(dataset['perguntas_frequentes'])}")
    print(f"Modelos de laudos: {len(dataset['modelos_laudos'])}")
    print(f"Receitas: {len(dataset['receitas_modelos'])}")
    print(f"Procedimentos: {len(dataset['procedimentos_internos'])}")
    print(f"Exemplos assistenciais: {len(dataset['exemplos_assistenciais'])}")


if __name__ == "__main__":
    main()
