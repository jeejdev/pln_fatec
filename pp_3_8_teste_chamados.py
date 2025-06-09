
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from collections import Counter

# Tokenizer simples
tokenizer = RegexpTokenizer(r'\w+')

# Função utilitária: converte frase em vetor médio das palavras
def sentence_vector(model, sentence):
    words = tokenizer.tokenize(sentence.lower())
    words_in_vocab = [word for word in words if word in model.wv]
    if not words_in_vocab:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in words_in_vocab], axis=0)

# 1. Carregar CSV
df = pd.read_csv("Chamados.csv", sep=';', encoding='utf-8', engine='python')

# 2. Preparar dataset
df = df[~df['Tipo'].isnull() & ~df['Título'].isnull()]

# Mapeia categorias
def map_tipo(tipo):
    if tipo.strip().lower() == 'incidente':
        return 'incidente'
    elif tipo.strip().lower() == 'requisição':
        return 'duvida'
    else:
        return None

df['categoria'] = df['Tipo'].map(map_tipo)
df = df[~df['categoria'].isnull()]

# 3. Adicionar frases de elogio artificiais
elogios_artificiais = [
    "Excelente atendimento, muito obrigado",
    "Parabéns pelo suporte rápido",
    "Muito satisfeito com a equipe",
    "Atendimento impecável, nota 10",
    "Ajudaram com muita competência, obrigado",
    "Serviço de TI está de parabéns",
    "Ótimo atendimento, como sempre",
    "Equipe muito eficiente e prestativa",
    "Recomendo o suporte técnico",
    "Suporte de alta qualidade, muito bom"
]

sentences = list(df['Título'].values) + elogios_artificiais
labels = list(df['categoria'].values) + ['elogio'] * len(elogios_artificiais)

# 4. Tokenizar frases
tokenized_sentences = [tokenizer.tokenize(sentence.lower()) for sentence in sentences]

# 5. Treinar Word2Vec
w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, workers=1, epochs=100)

# 6. Gerar vetores
X = np.array([sentence_vector(w2v_model, sentence) for sentence in sentences])

# 7. Codificar labels
le = LabelEncoder()
y = le.fit_transform(labels)

# 8. Treinar MLP
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
mlp.fit(X, y)

# 9. Split train/test para avaliação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

mlp_split = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
mlp_split.fit(X_train, y_train)

y_pred = mlp_split.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n===== Acurácia no test set: {accuracy:.2%} =====\n")

print("\n===== Relatório de classificação =====\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 13. Gráfico de precisão por categoria
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0,1,2])

plt.figure(figsize=(10,6))
bar_width = 0.25
classes = le.classes_

# Contagem de exemplos por categoria no train e test
train_counts = Counter(y_train)
test_counts = Counter(y_test)

idx_to_class = dict(zip(range(len(le.classes_)), le.classes_))

# Monta lista de rótulos no eixo X com info de treino/teste
xtick_labels = []
for i in range(len(classes)):
    train_count = train_counts.get(i, 0)
    test_count = test_counts.get(i, 0)
    label = f"{idx_to_class[i]}\nTrain: {train_count} / Test: {test_count}"
    xtick_labels.append(label)

# Posicionamento das barras
r1 = np.arange(len(classes))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Criar as barras
plt.bar(r1, precision, color='skyblue', width=bar_width, edgecolor='black', label='Precision')
plt.bar(r2, recall, color='lightgreen', width=bar_width, edgecolor='black', label='Recall')
plt.bar(r3, f1, color='salmon', width=bar_width, edgecolor='black', label='F1-Score')

# Labels
plt.xlabel('Categoria (Train/Test)', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(classes))], xtick_labels)
plt.ylabel('Score')
plt.ylim(0,1.05)
plt.title('Precision, Recall e F1-Score por categoria')
plt.legend()

plt.tight_layout()
plt.savefig("precision_recall_f1_por_categoria.png")
print("Gráfico de precisão/recall/F1 por categoria salvo como 'precision_recall_f1_por_categoria.png'")

# 10. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

fig_cm, ax_cm = plt.subplots(figsize=(6,6))
disp.plot(ax=ax_cm, cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix (Acurácia: {accuracy:.2%})')
plt.tight_layout()
plt.savefig("confusion_matrix_real.png")
print("Confusion matrix salva como 'confusion_matrix_real.png'")

# 11. Gráfico de frases por categoria
label_counts = Counter(labels)
categories = list(label_counts.keys())
counts = list(label_counts.values())

plt.figure(figsize=(6,4))
plt.bar(categories, counts, color=['red', 'blue', 'green'])
plt.title('Número de frases por categoria (real + elogios)')
plt.xlabel('Categoria')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.savefig("frases_por_categoria_real.png")
print("Gráfico de frases por categoria salvo como 'frases_por_categoria_real.png'")

# 12. Teste com frases simuladas (como o professor pediu)
operator_errors = [
    ("O sistema apresenta falha na conexão", "duvida"),
    ("Como posso atualizar minha senha?", "elogio"),
    ("Atendimento foi impecável, recomendo", "duvida")
]

print("\n===== Resultados com frases simuladas =====\n")

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
