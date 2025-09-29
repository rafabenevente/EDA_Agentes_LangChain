"""
Geração de dados de teste para o sistema EDA
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_sales_data():
    """Cria um dataset de vendas de exemplo"""
    np.random.seed(42)
    
    # Configurações
    n_samples = 1000
    start_date = datetime(2023, 1, 1)
    
    # Gerar dados
    dates = [start_date + timedelta(days=x) for x in range(n_samples)]
    
    # Produtos
    products = ['Produto_A', 'Produto_B', 'Produto_C', 'Produto_D', 'Produto_E']
    product_names = np.random.choice(products, n_samples)
    
    # Regiões
    regions = ['Norte', 'Sul', 'Leste', 'Oeste', 'Centro']
    region_names = np.random.choice(regions, n_samples)
    
    # Vendedores
    salespeople = [f'Vendedor_{i:02d}' for i in range(1, 21)]
    salesperson_names = np.random.choice(salespeople, n_samples)
    
    # Vendas (com correlações e padrões)
    base_price = {'Produto_A': 100, 'Produto_B': 150, 'Produto_C': 80, 'Produto_D': 200, 'Produto_E': 120}
    prices = [base_price[p] * np.random.uniform(0.8, 1.2) for p in product_names]
    
    quantities = np.random.poisson(5, n_samples) + 1
    
    # Adicionar sazonalidade
    seasonal_factor = [1 + 0.3 * np.sin(2 * np.pi * i / 365) for i in range(n_samples)]
    quantities = [int(q * s) for q, s in zip(quantities, seasonal_factor)]
    
    revenues = [p * q for p, q in zip(prices, quantities)]
    
    # Custos (correlacionado com preço)
    costs = [p * np.random.uniform(0.4, 0.7) for p in prices]
    profits = [r - c * q for r, c, q in zip(revenues, costs, quantities)]
    
    # Adicionar alguns outliers
    outlier_indices = np.random.choice(n_samples, 20, replace=False)
    for idx in outlier_indices:
        revenues[idx] *= np.random.uniform(3, 5)  # Receitas anômalas
    
    # Criar DataFrame
    df = pd.DataFrame({
        'data': dates,
        'produto': product_names,
        'regiao': region_names,
        'vendedor': salesperson_names,
        'preco_unitario': prices,
        'quantidade': quantities,
        'receita': revenues,
        'custo_unitario': costs,
        'lucro': profits,
        'margem_lucro': [l/r*100 if r > 0 else 0 for l, r in zip(profits, revenues)]
    })
    
    # Adicionar algumas linhas com valores nulos
    null_indices = np.random.choice(n_samples, 30, replace=False)
    for idx in null_indices[:10]:
        df.loc[idx, 'margem_lucro'] = np.nan
    for idx in null_indices[10:20]:
        df.loc[idx, 'vendedor'] = np.nan
    for idx in null_indices[20:]:
        df.loc[idx, 'custo_unitario'] = np.nan
    
    return df


def create_sample_customer_data():
    """Cria um dataset de clientes de exemplo"""
    np.random.seed(123)
    
    n_customers = 500
    
    # Dados básicos
    customer_ids = [f'CUST_{i:04d}' for i in range(1, n_customers + 1)]
    
    ages = np.random.normal(40, 15, n_customers).astype(int)
    ages = np.clip(ages, 18, 80)
    
    genders = np.random.choice(['M', 'F', 'Outro'], n_customers, p=[0.45, 0.45, 0.1])
    
    cities = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Salvador', 'Brasília', 
             'Fortaleza', 'Manaus', 'Curitiba', 'Recife', 'Porto Alegre']
    customer_cities = np.random.choice(cities, n_customers)
    
    # Renda (correlacionada com idade)
    base_income = 3000 + ages * 100 + np.random.normal(0, 1000, n_customers)
    incomes = np.clip(base_income, 1500, 15000)
    
    # Score de crédito (correlacionado com renda)
    credit_scores = 300 + (incomes / 15000) * 500 + np.random.normal(0, 50, n_customers)
    credit_scores = np.clip(credit_scores, 300, 850).astype(int)
    
    # Número de compras (correlacionado com renda e score)
    purchase_counts = np.random.poisson(
        2 + (incomes / 5000) + (credit_scores / 400), n_customers
    )
    
    # Valor total gasto
    total_spent = purchase_counts * incomes * 0.1 * np.random.uniform(0.5, 2, n_customers)
    
    # Segmento de cliente
    segments = []
    for i in range(n_customers):
        if total_spent[i] > 10000 and credit_scores[i] > 700:
            segments.append('Premium')
        elif total_spent[i] > 5000:
            segments.append('Gold')
        elif total_spent[i] > 2000:
            segments.append('Silver')
        else:
            segments.append('Bronze')
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'idade': ages,
        'genero': genders,
        'cidade': customer_cities,
        'renda_mensal': incomes,
        'score_credito': credit_scores,
        'num_compras': purchase_counts,
        'valor_total_gasto': total_spent,
        'segmento': segments,
        'ativo': np.random.choice([True, False], n_customers, p=[0.8, 0.2])
    })
    
    # Adicionar valores nulos
    null_indices = np.random.choice(n_customers, 25, replace=False)
    for idx in null_indices[:10]:
        df.loc[idx, 'renda_mensal'] = np.nan
    for idx in null_indices[10:]:
        df.loc[idx, 'score_credito'] = np.nan
    
    return df


def save_test_datasets():
    """Salva os datasets de teste"""
    os.makedirs('tests/test_data', exist_ok=True)
    
    # Dataset de vendas
    sales_df = create_sample_sales_data()
    sales_df.to_csv('tests/test_data/vendas_exemplo.csv', index=False)
    print(f"Dataset de vendas criado: {sales_df.shape}")
    
    # Dataset de clientes
    customers_df = create_sample_customer_data()
    customers_df.to_csv('tests/test_data/clientes_exemplo.csv', index=False)
    print(f"Dataset de clientes criado: {customers_df.shape}")
    
    # Criar um dataset pequeno para testes rápidos
    small_df = sales_df.head(100)
    small_df.to_csv('tests/test_data/vendas_pequeno.csv', index=False)
    print(f"Dataset pequeno criado: {small_df.shape}")
    
    return sales_df, customers_df


if __name__ == "__main__":
    print("Gerando datasets de teste...")
    sales_df, customers_df = save_test_datasets()
    
    print("\n=== Resumo dos Datasets ===")
    print(f"\n📊 Vendas ({sales_df.shape}):")
    print(sales_df.info())
    
    print(f"\n👥 Clientes ({customers_df.shape}):")
    print(customers_df.info())
    
    print("\n✅ Datasets de teste criados com sucesso!")
    print("Arquivos salvos em: tests/test_data/")