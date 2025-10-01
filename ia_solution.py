# 1. Ferramentas que vamos usar (Bibliotecas)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- FUNÇÃO PRINCIPAL: Treina o "Robô" de IA ---
def treinar_e_testar_avaliador_de_mensagens():
    """
    Função que faz o processo completo de IA: prepara dados, treina o modelo, avalia e testa.
    """
    print("--- Início do Processo de IA: Treinamento do Avaliador de Mensagens ---")
    
    # 2. DADOS DE TREINAMENTO (O que o robô vai aprender)
    # Frases e seus rótulos (Positivo/Negativo)
    dados = {
        'frase': [
            "Eu amo aprender IA",
            "Este desafio é muito difícil",
            "Excelente oportunidade de vaga",
            "Estou estressado com o prazo",
            "Gostei muito da proposta",
            "Não entendi nada do que foi pedido"
        ],
        'sentimento': [
            'positivo', 
            'negativo', 
            'positivo', 
            'negativo', 
            'positivo',
            'negativo'
        ]
    }
    
    df = pd.DataFrame(dados)
    
    # Dividir os dados em treino (para aprender) e teste (para verificar)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        df['frase'], df['sentimento'], test_size=0.3, random_state=42
    )

    # 3. TRANSFORMAÇÃO DE TEXTO (Converter palavras em números)
    # O computador não entende palavras. O vetorizador transforma texto em números.
    vetorizador = CountVectorizer()
    X_treino_vetorizado = vetorizador.fit_transform(X_treino)

    # 4. TREINAMENTO DO MODELO (O "aprendizado" da IA)
    # Usamos o Naive Bayes, um modelo simples e rápido para classificação de texto.
    modelo = MultinomialNB()
    modelo.fit(X_treino_vetorizado, y_treino)
    
    print("-> Treinamento Concluído.")
    
    # 5. AVALIAÇÃO (Ver se o robô aprendeu direito)
    X_teste_vetorizado = vetorizador.transform(X_teste)
    previsoes = modelo.predict(X_teste_vetorizado)
    acuracia = accuracy_score(y_teste, previsoes)
    
    print(f"** Acurácia (Desempenho) do Modelo: {acuracia*100:.0f}% **")

    # 6. TESTE FINAL (Colocar o robô para trabalhar)
    print("\n--- Teste Prático com Novas Mensagens ---")
    
    # Mensagem de teste 1 (Deveria ser Positivo)
    texto_para_teste_1 = "Estou muito animado com esta chance!"
    texto_1_vetorizado = vetorizador.transform([texto_para_teste_1])
    previsao_1 = modelo.predict(texto_1_vetorizado)
    print(f"Mensagem: '{texto_para_teste_1}' -> Sentimento Previsto: {previsao_1[0].upper()}")

    # Mensagem de teste 2 (Deveria ser Negativo)
    texto_para_teste_2 = "Este código está cheio de problemas e erros"
    texto_2_vetorizado = vetorizador.transform([texto_para_teste_2])
    previsao_2 = modelo.predict(texto_2_vetorizado)
    print(f"Mensagem: '{texto_para_teste_2}' -> Sentimento Previsto: {previsao_2[0].upper()}")
    
    print("\n--- Fim do Processo de IA ---")

# Bloco de execução: inicia o processo
if __name__ == "__main__":
    treinar_e_testar_avaliador_de_mensagens()