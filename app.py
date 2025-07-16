#!/usr/bin/env python3
"""
Sistema de Reconhecimento de Dígitos
==========================================================

Este script implementa um sistema completo de reconhecimento de dígitos
usando redes neurais e interface web. Funcionalidades:

1. Treinamento de modelo neural com dataset MNIST
2. Servidor web para interface de desenho
3. Predição em tempo real
4. Visualização das ativações da rede

Uso:
    python app.py          # Executar servidor (modelo já treinado)
"""


import os
import sys
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
import pandas as pd

# Configuração de logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten
from flask import Flask, request, jsonify, render_template

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

ARQUIVO_MODELO = 'mnist_model.h5'
ARQUIVO_TREINO = 'dataset/mnist_train.csv'
ARQUIVO_TESTE = 'dataset/mnist_test.csv'

# ============================================================================
# CLASSE PARA TREINAMENTO DO MODELO
# ============================================================================

class TreinadorModelo:
    """Classe responsável pelo treinamento da rede neural"""
    
    def __init__(self):
        self.modelo = None
        self.historico = None
    
    def carregar_dados(self):
        """Carrego e preparo os dados do MNIST"""
        print("Carregando dataset MNIST...")
        
        # Verifico se arquivos existem
        if not os.path.exists(ARQUIVO_TREINO) or not os.path.exists(ARQUIVO_TESTE):
            print(f"Erro: Arquivos de dataset não encontrados!")
            print(f"   Certifique-se que {ARQUIVO_TREINO} e {ARQUIVO_TESTE} existem.")
            sys.exit(1)
        
        # Carrego os dados
        dados_treino = pd.read_csv(ARQUIVO_TREINO)
        dados_teste = pd.read_csv(ARQUIVO_TESTE)
        
        # Separo labels e features
        self.y_treino = dados_treino['label'].values
        self.X_treino = dados_treino.drop('label', axis=1).values
        
        self.y_teste = dados_teste['label'].values
        self.X_teste = dados_teste.drop('label', axis=1).values
        
        # Reshape para formato de imagem (28x28)
        self.X_treino = self.X_treino.reshape(-1, 28, 28)
        self.X_teste = self.X_teste.reshape(-1, 28, 28)
        
        # Normalização - valores entre 0 e 1
        self.X_treino = self.X_treino / 255.0
        self.X_teste = self.X_teste / 255.0
        
        print(f"Dados carregados: {len(self.X_treino)} treino, {len(self.X_teste)} teste")
    
    def criar_modelo(self):
        """Crio a arquitetura da rede neural"""
        print("Criando arquitetura da rede neural...")
        
        self.modelo = Sequential([
            tf.keras.Input(shape=(28, 28), name='entrada'),
            Flatten(name='achatamento'),
            Dense(128, activation='relu', name='oculta_1'),
            Dense(64, activation='relu', name='oculta_2'),
            Dense(10, activation='softmax', name='saida')
        ])
        
        # Compilo o modelo
        self.modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Resumo da arquitetura:")
        self.modelo.summary()

# ============================================================================
# CLASSE PARA PREDIÇÕES
# ============================================================================

class Preditor:
    """Classe para fazer predições com o modelo"""
    
    def __init__(self):
        self.modelo = None
        self.modelo_ativacoes = None
    
    def carregar_modelo(self):
        """Carrego o modelo treinado"""
        if not os.path.exists(ARQUIVO_MODELO):
            print(f"Erro: Modelo {ARQUIVO_MODELO} não encontrado!")
            print("   O modelo deve estar presente no projeto.")
            sys.exit(1)
        
        print(f"Carregando modelo {ARQUIVO_MODELO}...")
        self.modelo = load_model(ARQUIVO_MODELO)
        
        # Crio modelo para extrair ativações
        saidas_camadas = [camada.output for camada in self.modelo.layers[1:]]
        self.modelo_ativacoes = Model(
            inputs=self.modelo.inputs, 
            outputs=saidas_camadas
        )
        
        print("Modelo carregado!")
    
    def predizer(self, imagem_processada):
        """
        Faço predição e extraio ativações
        
        Args:
            imagem_processada: Imagem processada pelo ProcessadorImagem
            
        Returns:
            dict: Resultados da predição
        """
        # Obtenho ativações de todas as camadas
        ativacoes = self.modelo_ativacoes.predict(imagem_processada, verbose=0)
        
        # Última ativação são as probabilidades finais
        probabilidades = ativacoes[-1][0]
        digito_predito = int(np.argmax(probabilidades))
        
        # Preparo ativações para JSON
        ativacoes_json = {
            'hidden_layer_1': ativacoes[0][0].tolist(),
            'hidden_layer_2': ativacoes[1][0].tolist(),
            'output_layer': ativacoes[2][0].tolist()
        }
        
        # Obtenho pesos da camada de saída
        pesos_saida = self.modelo.layers[-1].get_weights()[0].tolist()
        
        return {
            'prediction': digito_predito,
            'activations': ativacoes_json,
            'weights': pesos_saida
        }

# ============================================================================
# CLASSE PARA PROCESSAMENTO DE IMAGENS
# ============================================================================

class ProcessadorImagem:
    """Classe para processamento de imagens do canvas"""
    
    @staticmethod
    def processar_imagem(dados_imagem):
        """
        Processo imagem do canvas para formato adequado ao modelo
        
        Args:
            dados_imagem (str): Imagem em base64 do canvas
            
        Returns:
            numpy.ndarray: Imagem processada pronta para predição
        """
        try:
            # Decodifico base64
            dados_decodificados = base64.b64decode(dados_imagem.split(',')[1])
            imagem = Image.open(BytesIO(dados_decodificados))
            
            # Converto para escala de cinza
            imagem = imagem.convert('L')
            
            # Inverto cores (MNIST espera dígitos brancos em fundo preto)
            imagem = ImageOps.invert(imagem)
            
            # Redimensiono para 28x28
            imagem = imagem.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Converto para array e normalizo
            array_imagem = np.array(imagem) / 255.0
            
            # Adiciono dimensão de batch
            array_imagem = np.expand_dims(array_imagem, axis=0)
            
            return array_imagem
            
        except Exception as e:
            print(f"Erro ao processar imagem: {e}")
            return None

# ============================================================================
# APLICAÇÃO FLASK
# ============================================================================

# Instâncias globais
preditor = Preditor()
processador = ProcessadorImagem()
app = Flask(__name__)

@app.route('/')
def pagina_principal():
    """Rota principal - interface de desenho"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def endpoint_predicao():
    """
    Endpoint para predição de dígitos
    
    Recebe: JSON com imagem em base64
    Retorna: JSON com predição e ativações
    """
    try:
        # Verifico se modelo está carregado
        if preditor.modelo is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Obtenho dados da requisição
        dados = request.get_json()
        if 'image' not in dados:
            return jsonify({'error': 'No image provided'}), 400
        
        # Processo imagem
        imagem_processada = processador.processar_imagem(dados['image'])
        if imagem_processada is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Faço predição
        resultado = preditor.predizer(imagem_processada)
        
        return jsonify(resultado)
        
    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def iniciar_servidor():
    """Inicio o servidor Flask"""
    preditor.carregar_modelo()
    print("Iniciando servidor em http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal - inicia o servidor"""
    print("Sistema de Reconhecimento de Dígitos")
    print("=" * 50)
    print("O modelo já está treinado e pronto para uso!")
    print("Acesse: http://127.0.0.1:5000")
    print("=" * 50)
    
    iniciar_servidor()

if __name__ == '__main__':
    main() 