#!/usr/bin/env python
# coding: utf-8

# # **Visualización de datos**

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


cp = pd.read_csv("../datasets/campus-recruitment/Placement_Data_Full_Class.csv")
price = pd.read_csv("../datasets/houses-prices/train.csv")
age = pd.read_csv("../datasets/titanic-disaster/train.csv")
sales = pd.read_csv("../datasets/predict-futures-sales/items.csv")
word = pd.read_csv("../datasets/NLP-disaster-tweets/train.csv")
cases = pd.read_csv("../datasets/coronavirus-2019/covid_19_data.csv")
iris = pd.read_csv("../datasets/iris/iris.csv")
cp = cp.fillna(0)
stocks = pd.read_csv("../datasets/tesla-stock/TSLA.csv")


# ## **$1-$ Bar Plot**

# Un **diagrama de barras** *(o bar plot)* es uno de los tipos de diagrama más comunes. Muestra la relación entre una variable numérica y una variable categórica. Por ejemplo, puede mostrar la altura de varias personas mediante un gráfico de barras. Los gráficos de barras a menudo se confunden con histogramas, que es muy diferente *(solo tiene una variable numérica como entrada y muestra su distribución)*. 
# 
# Un error común es usar gráficos de barras para representar el valor promedio de cada grupo. Si tiene varios valores por grupo, mostrar solo el promedio disimula una parte de la información. En este caso, considere hacer un diagrama de caja *(boxplot)* o un diagrama de violín *(violinplot)*. Al menos, debe mostrar el número de observaciones por grupo y el intervalo de confianza de cada grupo.

# ### **Bar**

# In[6]:


import plotly.express as px

grs = cp.groupby(["gender"])[["salary"]].mean().reset_index()
fig = px.bar(grs[['gender', 'salary']].sort_values('salary', ascending=False), 
             y="salary", x="gender", color='gender', 
             log_y=True, template='ggplot2')
fig.show()


# ### **Horizontal Bar**

# In[7]:


grs = cp.groupby(["gender"])[["salary"]].mean().reset_index()
fig = px.bar(grs[['gender', 'salary']].sort_values('salary', ascending=False), 
             y="gender", x="salary", color='gender', orientation = 'h')
fig.show()


# ### **Stacked Bar**

# In[8]:


grgs = cp.groupby(["gender","specialisation"])[["salary"]].mean().reset_index()
fig = px.bar(grgs, x="gender", y="salary", color='specialisation', barmode='stack',
             height=400)
fig.show()


# ### **Group Bar**

# In[9]:


import plotly.express as px

fig = px.bar(grgs, x="gender", y="salary", color='specialisation', barmode='group',
             height=400)
fig.show()


# ## **$2-$ Count Plot**

# In[10]:


import seaborn as sns

sns.set(style="darkgrid")
ax = sns.countplot(x="gender", data=cp)


# ## **$3-$ Histogram**

# Un histograma es una visualización gráfica de datos utilizando barras de diferentes alturas. En un histograma, cada barra agrupa los números en rangos. Las barras más altas muestran que hay más datos en ese rango. Un histograma muestra la forma y la extensión de los datos de muestra continua.

# In[11]:


fig = px.histogram(cp, x="degree_p", y="salary", color="gender")
fig.show()


# ### **$2$-D Histogram**

# In[12]:


fig = px.density_heatmap(cp, x="degree_p", y="salary")
fig.show()


# ### **Marginal Histogram**

# In[13]:


fig = px.density_heatmap(cp, x="degree_p", y="salary",marginal_x="histogram", marginal_y="histogram")
fig.show()


# ### **Facet Histogram**

# In[14]:


fig = px.density_heatmap(cp, x="degree_p", y="salary",facet_row="ssc_b", facet_col="hsc_b")
fig.show()


# ### **With Box Margin**

# In[15]:


fig = px.histogram(cp, x="degree_p", y="salary", color="gender",
                   marginal="box")
fig.show()


# ### **With Violin Margin**

# In[16]:


fig = px.histogram(cp, x="degree_p", y="salary", color="gender",
                   marginal="violin")
fig.show()


# ## **$4-$ Rel Plot**

# In[17]:


sns.relplot(x="mba_p", y="salary", hue="gender", palette="muted",
            height=6, data=cp)


# ## **$5-$ Dist Plot**

# In[18]:


sns.distplot(cp['salary'], bins=10, kde=True)


# ## **$6-$ Pie Plot**

# Un **gráfico circular** *(o Pie Plot)* es un gráfico estadístico circular, que se divide en sectores para ilustrar la proporción numérica. En un gráfico circular, la longitud del arco de cada corte (y, en consecuencia, su ángulo central y área), es proporcional a la cantidad que representa. Si bien se llama así por su parecido con un pastel que se ha cortado, hay variaciones en la forma en que se puede presentar.

# ### **Plotting Pie Chart of Degree and percentage is a feature**

# In[20]:


grdsp = cp.groupby(["degree_t"])[["degree_p"]].mean().reset_index()

fig = px.pie(grdsp,
             values="degree_p",
             names="degree_t",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[21]:


import plotly.graph_objs as go

fig = go.Figure(data=[go.Pie( values=cp.hsc_s, hole=.6)])
fig.show()


# ## **$7-$ Tree Plot**

# Visualiza datos jerárquicos que se extienden radialmente hacia afuera desde las raíces hasta la partida. Las raíces comienzan desde el centro y los niños solamente.

# In[22]:


grss = cp.groupby(["hsc_b","hsc_s"])[["hsc_p"]].mean().reset_index()

fig = px.treemap(grss, path=['hsc_b','hsc_s'], values='hsc_p',
                  color='hsc_p', hover_data=['hsc_s'],
                  color_continuous_scale='rainbow')
fig.show()


# ## **$8-$ Sunburst Plot**

# Las gráficas *Sunburst* visualizan datos jerárquicos que se extienden radialmente hacia afuera desde la raíz hasta las hojas. La jerarquía del sector del *Sunburst* está determinada por las entradas en las etiquetas *(nombres en px. Sunburst)* y en los padres. La raíz comienza desde el centro y los hijos se agregan a los anillos exteriores.

# In[23]:


sales = sales.tail(20)
fig = px.sunburst(sales, path=["item_category_id",'item_id'],
                  color='item_category_id', hover_data=['item_id'],
                  color_continuous_scale='rainbow')
fig.show()


# ## **$8-$ Scatter Plot**

# Un diagrama de dispersión *(también llamado scatterplot, scatter graph, scatter chart, scattergram o scatter diagram)* es un tipo de diagrama de diagrama o diagrama matemático que utiliza coordenadas cartesianas para mostrar valores para típicamente dos variables para un conjunto de datos.

# In[24]:


plt.scatter(price.LotArea,price.SalePrice)


# ## **$9-$ Trend Line**

# In[ ]:





# ## **$10-$ Ternary Plot**

# In[ ]:





# ## **$11-$ Line Chart**

# In[ ]:





# ## **$12-$ Density Plot**

# In[ ]:





# ## **$13-$ Bubble Plot**

# In[ ]:





# ## **$14-$ Calender Plot**

# In[ ]:





# ## **$15-$ Box Plot**

# In[ ]:





# ## **$16-$ Violin Plot**

# In[ ]:





# ## **$17-$ Joint Plot**

# In[ ]:





# ## **$18-$ Funnel Plot**

# In[ ]:





# ## **$19-$ Correlation Plot**

# In[ ]:





# ## **$20-$ Pair Plot**

# In[ ]:





# ## **$21-$ Cluster Plot**

# In[ ]:





# In[ ]:




