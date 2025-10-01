# Allisson Lourdes Guevara Palma
# Rene Oswaldo Orellana de la O


a. Describe brevemente de que trata el dataset utilizado.
 El dataset contiene una lista de los 100 videos musicales más populares en YouTube para el año 2025. Para cada video, incluye información como el título, el número de visualizaciones (view_count), la categoría, el nombre del canal y la cantidad de seguidores del canal (channel_follower_count).


b. ¿Que informacion permite ver el resumen estadistico?
Un resumen estadístico (usando una función como describe() en pandas) nos daría una visión general de las columnas numéricas como view_count, duration y channel_follower_count. Mostraría lo siguiente para cada una:

count: El número total de registros.
mean: El valor promedio (por ejemplo, el promedio de visualizaciones).
std: La desviación estándar, que indica qué tan dispersos están los datos respecto al promedio.
min: El valor mínimo (la canción con menos vistas).
25%, 50%, 75%: Los percentiles, que ayudan a entender la distribución de los datos. El 50% es la mediana.
max: El valor máximo (la canción con más vistas).


c. ¿Que cambios o tendencias se detectan en la informacion del dataset?
Con solo este conjunto de datos, que es una "foto" de un momento específico, no se pueden detectar tendencias a lo largo del tiempo (como qué artistas han ganado o perdido popularidad). Sin embargo, se pueden observar tendencias dentro del propio dataset, como la relación entre el número de seguidores de un canal y las visualizaciones de sus videos. Es probable que exista una tendencia a que los artistas con más seguidores (channel_follower_count) tengan videos con más visualizaciones (view_count).


d. ¿Que categorias sobresalen en la comparacion y por que crees que sera?
La categoría que más sobresale es, sin duda, "Music". Esto es completamente esperado, ya que el dataset trata sobre las "100 mejores canciones". La gran mayoría de los videos son videos musicales oficiales, visualizadores o videos con letra. También aparece la categoría "People & Blogs", pero en una proporción mucho menor.


e. ¿Que diferencias se observan entre los primeros y ultimos registros?
El dataset parece estar ordenado por popularidad (probablemente por view_count de mayor a menor). Por lo tanto, las diferencias clave son:

Primeros registros: Corresponden a las canciones más exitosas, con un número de visualizaciones extremadamente alto (cientos o incluso miles de millones).
Últimos registros: Aunque siguen siendo canciones muy populares para estar en el top 100, tendrán un número de visualizaciones significativamente menor en comparación con las primeras posiciones.


 f. ¿Que aportan las medidas del dataset?
 Las medidas estadísticas son fundamentales para cuantificar y resumir el dataset, permitiendo un análisis más objetivo:

Promedio y Mediana: Nos dan una idea del valor "típico" de visualizaciones o seguidores para una canción del top 100.
Desviación Estándar: Nos dice si la popularidad está concentrada en unas pocas canciones (una desviación alta) o si está más repartida.
Máximos y Mínimos: Muestran el rango de popularidad, desde la canción más exitosa hasta la "menos" exitosa dentro del top 100.
Percentiles: Ayudan a entender cómo se agrupan los datos. Por ejemplo, podríamos ver si el 25% de las canciones más populares acapara el 75% de las visualizaciones totales.