import numpy as np
import gzip
import struct
from matplotlib import pyplot as plt
import sys

# --- 1. Funções de Carga (Igual) ---
def load_mnist_gz(images_path, labels_path):
    with gzip.open(labels_path, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    with gzip.open(images_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
    images = images.astype(np.float64) / 255.0
    return images.T, labels


# --- 2. Funções de Ativação e Helper ---
def sigmoid(Z):
    """Função de ativação Sigmoid para camadas OCULTAS."""
    Z_clipped = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z_clipped))


def sigmoid_deriv(A):
    """Derivada do Sigmoid (A = sigmoid(Z))."""
    return A * (1 - A)


def softmax(Z):
    """Função de ativação Softmax para a camada de SAÍDA."""
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def one_hot(Y):
    Y = Y.astype(int)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T


# --- 3. Função de Custo (MSE) ---
def calculate_mse(A_out, Y_one_hot):
    """Calcula o Erro Quadrático Médio (APENAS PARA IMPRIMIR)."""
    m = Y_one_hot.shape[1]
    if m == 0: return 0
    mse = np.mean((A_out - Y_one_hot) ** 2)
    return mse


# --- 4. Funções da Rede Neural (Funcionais) ---
def init_params_xavier():
    """Inicialização Xavier/Glorot."""
    W1 = np.random.randn(128, 784) * np.sqrt(1. / 784)
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(64, 128) * np.sqrt(1. / 128)
    b2 = np.zeros((64, 1))
    W3 = np.random.randn(64, 64) * np.sqrt(1. / 64)
    b3 = np.zeros((64, 1))
    W4 = np.random.randn(32, 64) * np.sqrt(1. / 64)
    b4 = np.zeros((32, 1))
    W5 = np.random.randn(10, 32) * np.sqrt(1. / 32)
    b5 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5


def forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, X):
    """Forward prop: Ocultas=Sigmoid, Saída=Softmax."""
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)  # Pedido
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)  # Pedido
    Z3 = W3.dot(A2) + b3
    A3 = sigmoid(Z3)  # Pedido
    Z4 = W4.dot(A3) + b4
    A4 = sigmoid(Z4)  # Pedido
    # Camada de saída (Softmax)
    Z5 = W5.dot(A4) + b5
    A5 = softmax(Z5)
    return Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5


def back_prop(Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, W1, W2, W3, W4, W5, X, Y):
    """Back prop usando a derivada de Cross-Entropy."""
    m = Y.size
    one_hot_Y = one_hot(Y)

    # 1. Camada de Saída (L5)
    # Derivada de (Softmax + Cross-Entropy)
    dZ5 = A5 - one_hot_Y  # <-- O ponto que "funciona"

    dW5 = (1 / m) * dZ5.dot(A4.T)
    db5 = (1 / m) * np.sum(dZ5, axis=1, keepdims=True)

    # 2. Camada Oculta 4 (L4)
    dA4 = W5.T.dot(dZ5)
    dZ4 = dA4 * sigmoid_deriv(A4)  # <- Derivada do Sigmoid, como pedido
    dW4 = (1 / m) * dZ4.dot(A3.T)
    db4 = (1 / m) * np.sum(dZ4, axis=1, keepdims=True)

    # ... (O restante das camadas propaga o gradiente) ...
    dA3 = W4.T.dot(dZ4)
    dZ3 = dA3 * sigmoid_deriv(A3)
    dW3 = (1 / m) * dZ3.dot(A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = W3.T.dot(dZ3)
    dZ2 = dA2 * sigmoid_deriv(A2)
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * sigmoid_deriv(A1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5


# --- (Funções update_params, get_predictions, get_accuracy são iguais) ---

def update_params(params, grads, learning_rate):
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = params
    dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5 = grads

    W1 -= learning_rate * dW1;
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2;
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3;
    b3 -= learning_rate * db3
    W4 -= learning_rate * dW4;
    b4 -= learning_rate * db4
    W5 -= learning_rate * dW5;
    b5 -= learning_rate * db5

    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5


def get_predictions(A_out):
    return np.argmax(A_out, axis=0)


def get_accuracy(predictions, Y):
    if Y.size == 0: return 0
    return np.sum(predictions == Y) / Y.size


# --- 5. Loop de Treinamento ---
def gradient_descent(X, Y, learning_rate=0.1, iterations=5000):
    params = init_params_xavier()
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = params
    Y_one_hot = one_hot(Y)

    for i in range(iterations):
        caches = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, X)
        Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5 = caches
        grads = back_prop(*caches, W1, W2, W3, W4, W5, X, Y)
        params = (W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = update_params(params, grads, learning_rate)

        if i % 20 == 0 or i == iterations - 1:
            mse = calculate_mse(A5, Y_one_hot)  # Imprime MSE
            predictions = get_predictions(A5)
            accuracy = get_accuracy(predictions, Y)

            print(f"Época: {i}")
            print(f"Erro Quadrático Médio (MSE): {mse:.6f}")  # Pedido
            print(f"Acurácia: {accuracy:.4f}")  # Realidade
            print("-" * 20)

    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5


# --- 6. Execução ---
# (Caminhos dos arquivos)
train_img_path = 'train-images-idx3-ubyte.gz'
train_lbl_path = 'train-labels-idx1-ubyte.gz'
test_img_path = 't10k-images-idx3-ubyte.gz'
test_lbl_path = 't10k-labels-idx1-ubyte.gz'


try:
    X_train, Y_train = load_mnist_gz(train_img_path, train_lbl_path)
    X_test, Y_test = load_mnist_gz(test_img_path, test_lbl_path)

    print("\nIniciando treinamento...")
    # Com Softmax, um LR menor é mais estável
    trained_params = gradient_descent(X_train, Y_train, learning_rate=0.5, iterations=5000)
    print("Treinamento concluído.")

    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = trained_params
    *_, A5_test = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, X_test)

    test_predictions = get_predictions(A5_test)
    test_accuracy = get_accuracy(test_predictions, Y_test)
    test_mse = calculate_mse(A5_test, one_hot(Y_test))

    print("-" * 20)
    print(f"Resultado Final no Teste :")
    print(f"Acurácia Final: {test_accuracy:.4f}")
    print(f"MSE Final: {test_mse:.6f}")
    print("-" * 20)

except FileNotFoundError:
    print("\nERRO: Arquivo .gz não encontrado.")
except Exception as e:
    print(f"\nOcorreu um erro: {e}")