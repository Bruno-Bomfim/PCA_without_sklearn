import create
import numpy as np
import pandas as pd
import plotly.express as px

def main():
#importa o arquivo do excel
    arquivo = str(input('Digite o local do arquivo: '))
    dados = create.PCA(pd.read_excel(arquivo))
#aplica pca na base de dados
    dados.pca(3)
#dados com pca ja aplicado
    dados.dados_pca
#plota os dados em um grafico 3d
    fig = px.scatter_3d(dados.dados_pca, x=0, y=1, z=2)
    fig.show()

if __name__ == '__main__':
  main()