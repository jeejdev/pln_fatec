
import numpy as np
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt

# Tokenizer simples
tokenizer = RegexpTokenizer(r'\w+')

# Função utilitária: converte frase em vetor médio das palavras
def sentence_vector(model, sentence):
    words = tokenizer.tokenize(sentence.lower())
    words_in_vocab = [word for word in words if word in model.wv]
    if not words_in_vocab:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in words_in_vocab], axis=0)

# 1. Criar 9 entradas (3 incidente, 3 duvida, 3 elogio)

incident_entries = [
    "O sistema travou completamente",
    "Erro ao salvar o arquivo",
    "A aplicação não inicia após atualização"
]

duvida_entries = [
    "Como posso redefinir minha senha?",
    "Onde encontro o manual do usuário?",
    "Qual é o procedimento para enviar um relatório?"
]

elogio_entries = [
    "Ótimo atendimento, estou satisfeito",
    "Parabéns pela agilidade na resposta",
    "Equipe muito competente, gostei do suporte"
]

sentences = incident_entries + duvida_entries + elogio_entries
labels = ['incidente'] * 3 + ['duvida'] * 3 + ['elogio'] * 3

# Tokenizar frases
tokenized_sentences = [tokenizer.tokenize(sentence.lower()) for sentence in sentences]

# Treinar Word2Vec
w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, workers=1, epochs=100)

# Gerar vetores
X = np.array([sentence_vector(w2v_model, sentence) for sentence in sentences])

# Codificar labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Treinar MLP
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
mlp.fit(X, y)

# 2. Criar 6 reviews com erro do operador (duas por categoria)

operator_errors = [
    ("A tela fica em branco ao tentar abrir o sistema", "duvida"),    # deveria ser incidente
    ("Erro inesperado durante o processamento do pedido", "duvida"),  # deveria ser incidente

    ("Gostei muito da nova interface", "incidente"),                  # deveria ser elogio
    ("Atendimento foi excelente, recomendo", "incidente"),            # deveria ser elogio

    ("Como configurar permissões avançadas no sistema?", "elogio"),   # deveria ser duvida
    ("Onde posso encontrar as configurações de segurança?", "elogio") # deveria ser duvida
]

# 3. Executar modelo sobre as frases de erro do operador

print("\n===== Resultados com frases simuladas (PP.3.8) =====\n")

for sentence, operator_label in operator_errors:
    vec = sentence_vector(w2v_model, sentence).reshape(1, -1)
    pred_idx = mlp.predict(vec)[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    print(f"Solicitação: '{sentence}'")
    print(f" -> Classificação do operador: {operator_label}")
    print(f" -> Classificação pelo modelo: {pred_label}")

    if pred_label != operator_label:
        print(" -> ⚠️ ERRO DO OPERADOR DETECTADO!")
    else:
        print(" -> ✅ Modelo concorda com o operador.")

    print("-" * 50)
