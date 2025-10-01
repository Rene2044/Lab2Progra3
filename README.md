# Allisson Lourdes Guevara Palma
# Rene Oswaldo Orellana de la O

a. Describe brevemente de que trata el dataset utilizado.
Contiene mediciones de flores, especificamente la longitud y el ancho de los sepalos y petalos. Cada fila representa una flor y la columna tarjet indica la categoria o especie a la que pertenece (clasificadas como 0, 1 o 2).

b. ¿Que informacion permite ver el resemne estadistico?
Un resumen estadístico (como el que genera la función describe() de Pandas) te daría para cada columna numérica:

El conteo total de registros (count).
El valor promedio (mean).
La dispersión de los datos respecto al promedio (std o desviación estándar).
El valor mínimo (min) y máximo (max).
Los percentiles (25%, 50%, 75%), que te ayudan a entender cómo se distribuyen los datos.
Esto es útil para tener una visión general rápida de las características de las flores.

c. ¿Que cambios o tendencias se detectan en la informacion del dataset?
Se observa una tendencia clara: a medida que los valores de longitud y ancho de los pétalos aumentan, la categoría (target) también cambia. Las flores de la categoría 0 tienen pétalos notablemente más pequeños que las de las categorías 1 y 2.

d. ¿Que categorias sobresalen en la comparacion y por que crees que sera?
En la muestra de datos, la categoría 1 tiene más registros que las demás. Sin embargo, en un análisis completo, si una categoría tuviera muchas más muestras que otras, podría deberse a la forma en que se recolectaron los datos (por ejemplo, una especie de flor era más común en la zona de recolección).

e. Que diferencias se observan entre los primeros y ultimos registros?
Primeros registros (target 0): Muestran flores con sépalos y pétalos más pequeños. Por ejemplo, la longitud del pétalo es de alrededor de 1.3-1.4 cm.
Últimos registros (target 1 y 2): Tienen medidas significativamente mayores. El último registro (target 2) tiene la mayor longitud (5.1 cm) y ancho (1.8 cm) de pétalo en la muestra.
Esto refuerza la idea de que el tamaño, especialmente del pétalo, es un factor distintivo clave entre las especies.


f. ¿Que aportan las medidas estadisticas al analisis del dataset?
Las medidas estadísticas son cruciales porque permiten:

Resumir y simplificar grandes volúmenes de datos en cifras manejables (como promedios o medianas).
Entender la variabilidad de los datos (con la desviación estándar) es decir, si las medidas para cada especie son muy parecidas o muy diferentes entre si.
Comparar grupos de forma objetiva. Por ejemplo, puedes comparar la longitud media del pétalo de la categoría 0 con la de la categoría 1 para confirmar que son diferentes.
Identificar valores atípicos o posibles errores en los datos.