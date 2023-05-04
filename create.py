import numpy as np
import pandas as pd

class PCA:
  
  def __init__(self, dados):
    self.dados = dados

  def pca(self,n_pc):
    # Centralizar os dados
    dados_centralizados = self.dados - self.dados.mean()
    #Deixa os dados na mesma escala
    lista = []
    for i in list(dados_centralizados):
      max = dados_centralizados[i].max()
      min = dados_centralizados[i].min()
      if max > min*-1:
        a = list(dados_centralizados[i]/max)
      else:
        a = list(dados_centralizados[i]/min)
      lista.append(a)
    nova_base_dados = pd.DataFrame(lista)
    nova_base_dados.T
    # Calcular a matriz de covariância
    matriz_covariancia = np.cov(nova_base_dados)
    # Calcular os autovalores e autovetores da matriz de covariância
    autovalores, autovetores = np.linalg.eig(matriz_covariancia)
    # Ordenar os autovalores em ordem decrescente (talvez a parte anterior já faça isso direto, mas prefiro garantir)
    idx = autovalores.argsort()[::-1]
    self.autovalores = autovalores[idx]
    self.autovetores = autovetores[:, idx]
    #calcula variância de cada PC em ordem decrescente
    soma_cumulativa = np.cumsum(autovalores)
    self.variancia = soma_cumulativa/soma_cumulativa[-1]
    # Projetar os dados nos componentes principais 
    self.componentes_principais = autovetores.T[:n_pc].dot(nova_base_dados)
    # Criar dataframe dos novos dados
    self.dados_pca = pd.DataFrame(self.componentes_principais.T)
    # Nomeia as colunas como PC1, PC2 e PC3
    self.dados_pca.columns =  ['PC1','PC2','PC3']
    