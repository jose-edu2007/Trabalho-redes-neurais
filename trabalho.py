# Rede neural de 2 camadas — Lecture 4 (CS231n, Fei-Fei Li)
# Implementação baseada nos slides 30-34.
# Alterações em relação ao código original dos slides:
#   1. O print(t, loss) foi removido de dentro do loop.
#      Motivo: imprimia 2000 linhas, excedendo o scrollback do
#      terminal e impossibilitando a leitura completa da saída.
#   2. Adicionado um print condicional (a cada 100 iterações).
#      Motivo: exibir a evolução da loss de forma legível,
#      totalizando ~20 linhas e evidenciando a convergência.
#   3. Adicionado um print final com a loss após o treinamento.
#      Motivo: registrar explicitamente o resultado final.
# Observação: a lógica matemática (forward, backward e
# atualização dos pesos) permanece idêntica à dos slides.
import numpy as np
from numpy.random import randn
N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)
for t in range(2000):
# ---------- Forward pass ----------
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
  # Print condicional (substitui o print(t, loss) original)
    if t % 100 == 0:
        print(f"iter {t:4d} | loss = {loss:.6f}")
    # ---------- Backward pass ----------
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    # ---------- Atualização dos pesos ----------
    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2
print(f"loss final = {loss:.6f}")  