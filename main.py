import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Загрузка и базовая предобработка
df = pd.read_csv('amazon_ecommerce_1M.csv')

# Выбираем нужные столбцы (числовые и категориальные)
cols = ['user_id', 'product_id', 'category',
        'subcategory', 'brand', 'price', 'discount',
        'final_price', 'rating', 'review_count']
df = df[cols].copy()

# Очистка цен: удаляем символ ₹ и запятые, преобразуем во float
def clean_price(series):
    return series.astype(str).str.replace('[₹,]', '', regex=True).astype(float)

df['price'] = clean_price(df['price'])
df['final_price'] = clean_price(df['final_price'])

# discount: если есть знак %, убираем, иначе просто во float
if df['discount'].dtype == object:
    df['discount'] = df['discount'].str.replace('%', '', regex=False).astype(float)

# Рейтинг и количество отзывов к числовому типу
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce')

# Обработка пропусков
num_cols = ['price', 'discount', 'final_price', 'rating', 'review_count']
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = ['category', 'subcategory', 'brand']
df[cat_cols] = df[cat_cols].fillna('Unknown')

# Удаляем дубликаты
df = df.drop_duplicates()

# Для скорости визуализации берём случайные 5000 строк
df_sample = df.sample(n=5000, random_state=42)

print(f"Размер рабочей выборки: {df_sample.shape}")
print(df_sample.head())
print(df_sample.dtypes)

# ---------- 2. Визуализации Seaborn ----------
sns.set_style("whitegrid")

#Распределение рейтинга
plt.figure(figsize=(8,5))
sns.histplot(df_sample['rating'], kde=True, bins=30, color='skyblue')
plt.title('Распределение рейтинга товаров')
plt.xlabel('Рейтинг')
plt.ylabel('Количество')
plt.show()

#Scatterplot: финальная цена vs исходная цена, окраска по категории
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_sample, x='price', y='final_price',
                hue='category', alpha=0.6, palette='tab10')
plt.title('Исходная цена vs Финальная цена (со скидкой)')
plt.xlabel('Исходная цена')
plt.ylabel('Финальная цена')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.show()

#Boxplot финальной цены по топ-10 категориям
top_categories = df_sample['category'].value_counts().nlargest(10).index
data_top = df_sample[df_sample['category'].isin(top_categories)]

plt.figure(figsize=(12,6))
sns.boxplot(data=data_top, x='category', y='final_price', palette='Set3')
plt.xticks(rotation=45)
plt.title('Боксплоты финальной цены для топ-10 категорий')
plt.xlabel('Категория')
plt.ylabel('Финальная цена')
plt.tight_layout()
plt.show()

# Heatmap корреляций числовых признаков
plt.figure(figsize=(8,6))
corr = df_sample[['price', 'discount', 'final_price', 'rating', 'review_count']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Тепловая карта корреляций')
plt.show()

# 3. Визуализации Plotly
# Интерактивный Scatterplot
fig = px.scatter(df_sample, x='price', y='final_price',
                 color='category', opacity=0.7,
                 title='Plotly Scatter: Исходная цена vs Финальная цена',
                 labels={'price':'Исходная цена', 'final_price':'Финальная цена'})
fig.show()

# Line plot: средняя финальная цена по категориям
avg_price = df_sample.groupby('category')['final_price'].mean().sort_values()
fig = px.line(x=avg_price.index, y=avg_price.values,
              title='Средняя финальная цена по категориям',
              labels={'x':'Категория', 'y':'Средняя финальная цена'})
fig.update_traces(mode='lines+markers')
fig.update_xaxes(tickangle=45)
fig.show()

# Bar chart: количество товаров в топ-10 категориях
cat_counts = df_sample['category'].value_counts().nlargest(10)
fig = px.bar(x=cat_counts.index, y=cat_counts.values,
             title='Топ-10 категорий по количеству товаров',
             labels={'x':'Категория', 'y':'Количество товаров'},
             color=cat_counts.values, color_continuous_scale='Viridis')
fig.show()

# Heatmap корреляций (Plotly)
corr_matrix = df_sample[['price', 'discount', 'final_price', 'rating', 'review_count']].corr()
fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                title='Тепловая карта корреляций (Plotly)',
                color_continuous_scale='RdBu_r')
fig.show()

# Интерактивный dropdown для выбора категории на scatter plot
categories = df_sample['category'].unique()
fig = go.Figure()

for cat in categories:
    data_cat = df_sample[df_sample['category'] == cat]
    fig.add_trace(go.Scatter(
        x=data_cat['price'],
        y=data_cat['final_price'],
        mode='markers',
        name=cat,
        marker=dict(opacity=0.6),
        visible=False
    ))

# Включаем видимость первого трейса по умолчанию
if fig.data:
    fig.data[0].visible = True

# Строим выпадающий список
buttons = []
for i, cat in enumerate(categories):
    visibility = [False] * len(categories)
    visibility[i] = True
    buttons.append(dict(
        label=cat,
        method='update',
        args=[{'visible': visibility},
              {'title': f'Исходная vs Финальная цена — {cat}'}]
    ))

fig.update_layout(
    updatemenus=[dict(active=0, buttons=buttons,
                      x=1.15, y=1, xanchor='left', yanchor='top')],
    title='Интерактивный scatter с выбором категории',
    xaxis_title='Исходная цена',
    yaxis_title='Финальная цена'
)
fig.show()