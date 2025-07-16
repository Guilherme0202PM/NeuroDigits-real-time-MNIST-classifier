# 🤖 Sistema de Reconhecimento de Dígitos com Redes Neurais

> **Projeto concluído** - Backend completo com API REST para reconhecimento de dígitos escritos à mão usando TensorFlow/Keras e Flask.

<div align="center">
  <img src="Demonstração.gif" alt="Demonstração do Sistema" width="600">
</div>

## Visão Geral

Este projeto implementa um sistema completo de **Inteligência Artificial** para reconhecimento de dígitos manuscritos (0-9) utilizando **Deep Learning** e **APIs REST**. O sistema demonstra uma arquitetura moderna com separação clara de responsabilidades, seguindo princípios de **Clean Code** e **SOLID**.

### 🎯 Objetivos do Projeto

- Demonstrar conhecimento em **Machine Learning** e **Deep Learning**
- Implementar arquitetura de **microserviços** com APIs REST
- Aplicar boas práticas de **desenvolvimento de software**
- Criar sistema escalável e manutenível

## Arquitetura do Sistema

### Backend (Implementado)
```
┌─────────────────────────────────────────────────────────────┐
│                    ARQUITETURA BACKEND                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Flask     │    │ TensorFlow  │    │   Pandas    │    │
│  │   API REST  │    │   / Keras   │    │   NumPy     │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│           │                   │                   │        │
│           └───────────────────┼───────────────────┘        │
│                               │                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              CLASSES PRINCIPAIS                    │    │
│  │  • TreinadorModelo - Preparação de dados          │    │
│  │  • Preditor - Sistema de predição                  │    │
│  │  • ProcessadorImagem - Pré-processamento          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Frontend (Implementado)
```
┌─────────────────────────────────────────────────────────────┐
│                   ARQUITETURA FRONTEND                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   HTML5     │    │   CSS3      │    │ JavaScript  │    │
│  │   Canvas    │    │ Responsive  │    │   ES6+      │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│           │                   │                   │        │
│           └───────────────────┼───────────────────┘        │
│                               │                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              FUNCIONALIDADES                       │    │
│  │  • Interface de desenho interativa                │    │
│  │  • Visualização da rede neural em tempo real      │    │
│  │  • Design responsivo e moderno                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Tecnologias Utilizadas

### Backend
- **Python 3.8+** - Linguagem principal
- **TensorFlow 2.x** - Framework de Deep Learning
- **Keras** - API de alto nível para redes neurais
- **Flask** - Framework web para APIs REST
- **Pandas** - Manipulação e análise de dados
- **NumPy** - Computação numérica
- **PIL/Pillow** - Processamento de imagens

### Frontend (Implementado)
- **HTML5 Canvas** - Interface de desenho
- **CSS3** - Estilização moderna e responsiva
- **JavaScript ES6+** - Lógica frontend
- **Fetch API** - Comunicação com backend

## Dataset e Modelo

### MNIST Dataset
- **imagens** de dígitos manuscritos
- **Formato**: 28x28 pixels em escala de cinza
- **Classes**: 10 dígitos (0-9)

### Arquitetura da Rede Neural
```
Input Layer (784) → Hidden Layer 1 (128) → Hidden Layer 2 (64) → Output Layer (10)
     ↓                    ↓                        ↓                    ↓
  28x28 pixels      ReLU Activation         ReLU Activation      Softmax Activation
```

## Funcionalidades Implementadas

### ✅ Backend Completo
- [x] **Carregamento e preparação de dados MNIST**
- [x] **Arquitetura de rede neural otimizada**
- [x] **Sistema de predição com extração de ativações**
- [x] **Pré-processamento de imagens (base64 → tensor)**
- [x] **API REST com endpoints / e /predict**
- [x] **Tratamento robusto de erros**
- [x] **Validação de dados e modelos**

### ✅ Frontend Completo
- [x] **Interface de desenho interativa**
- [x] **Visualização da rede neural em tempo real**
- [x] **Design responsivo e moderno**
- [x] **Animações e feedback visual**

## Estrutura do Projeto

```
projeto/
├── app.py                 # Aplicação principal Flask
├── mnist_model.h5        # Modelo treinado
├── templates/            # Templates HTML
└── README.md            # Documentação
```

## Competências Demonstradas

### Hard Skills
- **Deep Learning** com TensorFlow/Keras
- **APIs REST** com Flask
- **Processamento de Imagens** com PIL
- **Manipulação de Dados** com Pandas/NumPy
- **Arquitetura de Software** orientada a objetos
- **Versionamento** com Git

### Soft Skills
- **Pensamento estruturado** e lógico
- **Resolução de problemas** complexos
- **Documentação** técnica clara
- **Organização** e planejamento de projeto

## Próximos Passos

### Melhorias Futuras
1. **Otimizações de Performance**
   - Cache de predições
   - Compressão de imagens
   - Lazy loading de componentes

2. **Funcionalidades Avançadas**
   - Histórico de predições
   - Exportação de resultados
   - Múltiplos modelos de IA

3. **Expansão do Sistema**
   - Suporte a outros datasets
   - API para treinamento online
   - Integração com cloud services

## 📞 Contato
 
**LinkedIn**: https://www.linkedin.com/in/guilherme-peres-mundim/  
**GitHub**: https://github.com/Guilherme0202PM
