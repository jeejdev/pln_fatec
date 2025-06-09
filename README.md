
# PP.3.8 - Classificação de solicitações de Service Desk com Word2Vec + MLP

Este repositório contém duas implementações para o exercício **PP.3.8** da disciplina:

## Arquivos

### `pp_3_8_exercicio_professor.py`

- Implementa exatamente o enunciado do exercício PP.3.8:
  - Treina um modelo com **9 entradas** (3 incidente, 3 dúvida, 3 elogio).
  - Testa o modelo em **6 reviews com erro de operador**.
  - Verifica se o modelo consegue detectar os erros de classificação.

### `pp_3_8_teste_chamados.py`

- Implementa um pipeline mais completo com **dados reais** de chamados (arquivo `Chamados.csv`):
  - Carrega e processa os chamados reais.
  - Adiciona frases de elogio artificiais.
  - Treina um Word2Vec + MLP com todos os dados.
  - Avalia o modelo com train/test split e gera:
    - Gráfico de **precision/recall/F1** com **quantidade de treino/teste** por categoria.
    - **Confusion matrix**.
    - **Distribuição de exemplos** por categoria.
  - Testa também com frases de **erro de operador** (simuladas).

## Como executar

### Requisitos

Instalar as dependências com:

```bash
pip install -r requirements.txt
```

### Rodar o código do exercício (modelo com 9 frases + 6 erros)

```bash
python pp_3_8_exercicio_professor.py
```

### Rodar o código com os chamados reais

- Certifique-se de ter o arquivo `Chamados.csv` na mesma pasta.

```bash
python pp_3_8_teste_chamados.py
```

## Interpretação dos resultados

- **Incidente:** O modelo apresenta desempenho **excelente** nesta classe (alta precisão e recall), pois há muitos exemplos reais no dataset.
- **Duvida:** O modelo tem desempenho **razoável/aceitável**. Algumas requisições são redigidas de forma similar a incidentes, o que gera confusão.
- **Elogio:** O desempenho é **baixo**, pois o dataset não contém elogios reais — foram adicionadas apenas frases artificiais (muito poucas).
- A **quantidade de treino/teste por categoria** está mostrada no gráfico de **Precision/Recall/F1**, o que explica o desempenho observado.

## Observações

- O dataset real é **altamente desbalanceado** (predomínio da classe "incidente").
- A classe "elogio" foi criada artificialmente para cumprir o enunciado do exercício.
- O pipeline foi projetado para seguir **estritamente o que o professor pediu**.
- O modelo foi capaz de detectar corretamente os **erros de operador simulados** no exercício.
