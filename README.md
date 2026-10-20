# Alejandro Alejo Santana

## Práctica 3. Detección y reconocimiento de formas



## Contenidos

[Tarea 1](#31-tarea-1)  
[Tarea 2](#32-tarea-2)

## 3.1. Tarea 1

### Detección de monedas en una imagen fija

Lo primero que hacemos es definir los diámetros de las monedas en mm

``` python
DIAMETROS_EURO_MM = {
    2.00: 25.75,
    0.50: 24.25,
    1.00: 23.25,
    0.20: 22.25,
    0.05: 21.25,
    0.10: 19.75,
    0.02: 18.75,
    0.01: 16.25
}
```

convertimos la imagen a escala de grises, con medianBlur() reducimos el ruido antes de aplicar HoughCircles(), que busca formas circulares en la imagen


``` python
img_monedas = cv2.imread('Monedas.jpg')     # si se pone Monedas_mias.jpg, se prueba con otra imagen, pero los datos que proporciona no son correctos.
img_rgb = cv2.cvtColor(img_monedas, cv2.COLOR_BGR2RGB)
img_gris = cv2.cvtColor(img_monedas, cv2.COLOR_BGR2GRAY)

img_gris = cv2.medianBlur(img_gris, 5)

circulos = cv2.HoughCircles(img_gris, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=100, param2=40, minRadius=30, maxRadius=130)
```

HoughCircles, controla la sensibilidad del detector, separación mínima y rango de radios

el siguiente fragmento de código recorre los círculos detectados, para cada uno calcula la distancia entre el centro del círculo  y el punto donde se hizo clic. Si la distancia es menos que el radio r, el clic está dentro de ese círculo, para localizar un círculo donde se hizo clic, se utilizó la ayuda de la inteligencia artificial

``` python
for c in circulos[0, :]:
    x = int(c[0])
    y = int(c[1])
    r = int(c[2])
    distancia = np.sqrt((x - punto_clic[0]) ** 2 + (y - punto_clic[1]) ** 2)


    if distancia < r:
        radio_ref = r
        break
``` 
una vez encontrado el radio, calculamos la escala de píxeles por mm.

``` python
escala = (2 * radio_ref) / DIAMETROS_EURO_MM[1.00]
```
2 * radio, para obtener el diámetro , y al dividirlo por el de la moneda de 1 euros obtenemos los píxeles por mm

en la siguiente parte, haremos la suma, y compararemos las monedas

``` python
total = 0

for c in circulos[0, :]:
    x = int(c[0])
    y = int(c[1])
    r = int(c[2])
    diametro_mm = (2 * r) / escala

    diferencia_minima = float('inf')
    valor = 0

    for moneda in DIAMETROS_EURO_MM:
        
        diferencia = abs(diametro_mm - DIAMETROS_EURO_MM[moneda])
        if diferencia < diferencia_minima:
            diferencia_minima = diferencia
            valor = moneda
        
    
    total += valor
``` 
obtenemos el valor del diámetro de la moneda, y comprobamos con todo el diccionario, y nos quedamos con el que se quede más cercano en valor absoluto, y al final sumamos el valor, y dibujamos los círculos de nuevo, alrededor de las monedas

RESULTADO:Funciona con la imagen ideal, es cierto que, en la otra imagen que se puede usar Monedas_mias.jpg, no funciona del todo correcto y es que habría que usar HoughCircles y cambiar los parámetros, para poder hacer un mejor análisis con la otra imagen

<img src="Monedas_contadas.png" width="500">

### Detección de monedas con cámara en tiempo real

lo primero que hacemos es abrir la cámara ( esto es importante ya que no se usará del portátil, para mayor comodidad ), y la escala, esto influye mucho, ya que dependiendo si va estar muy cerca o muy lejos, hará que no se detecten las monedas, o hará que se detecten muchas más

``` python
vid = cv2.VideoCapture(1)
escala = 3.5
``` 

se realiza lo mismo que antes, solo que en vez de ser una imagen, es en un video, se convierten las imágenes a escala de grises y se reduce el ruido con un filtro, y volvemos a utilizar a Hough para los círculos.

``` python
while(True):      
   
    ret, frame = vid.read()

    if ret:
        
        framem=cv2.flip(frame, 1)
        gris = cv2.cvtColor(framem, cv2.COLOR_BGR2GRAY)
        gris = cv2.medianBlur(gris, 5)

        circulos = cv2.HoughCircles(gris, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=30, minRadius=10, maxRadius=80)

``` 

en esta última sección de código se hace algo muy parecido a lo de antes, se calcula el diámetro de la moneda y lo dividimos por la escala, comprobamos el diámetro, con el resto de las monedas, le asignamos si coincide y tiene unos valores muy parecido, en ese momento le ponemos el valor y dibujamos un círculo con el valor, si no esta dentro de los margenes, dibujamos un círculos, poniendo que es un objeto desconocido

```python
if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            for c in circulos[0, :]:
                x = int(c[0])
                y = int(c[1])
                r = int(c[2])
                diametro_mm = (2 * r) / escala
                diferencia_minima = float('inf')
                valor = 0
                for moneda in DIAMETROS_EURO_MM:
                    diferencia = abs(diametro_mm - DIAMETROS_EURO_MM[moneda])
                    if diferencia < diferencia_minima:
                        diferencia_minima = diferencia
                        valor = moneda
                if diferencia_minima < 2.5:
                    total += valor
                    cv2.circle(resultado, (x, y), r, (0, 255, 0), 2)
                    cv2.putText(resultado, f"{valor}€", (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    cv2.circle(resultado, (x, y), r, (0, 0, 255), 2)
                    cv2.putText(resultado, "Desconocida", (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
``` 

luego simplemente, mostramos por pantalla, el valor total de las monedas que se muestran, y pintamos, los objetos circulares que tengan medidas aceptables, sean monedas o no

RESULTADO: no funciona del todo bien y es que lo que pasa es que al estar en una escala fija 3,5 px/mm, va a depender mucho de la distancia de la cámara, y es que hay varias monedas con tamaños similares, tampoco se probaron solapes


## 3.2. Tarea 2

la función contornos, convierte la imagen a escala de grises, suaviza el ruido y aplica un umbral de Otsu, y findContours extrae los contornos externos

``` python
def contornos(img):
    gris = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5,5), 0)
    _, th = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
``` 

para extraer las características, se utilizó la IA, ya que, esta fue una de las partes más complicadas, y es que al principio se intentó usar varias características, como medidas y colores y aunque al final el resultado es bajo (0,383 de accuracy), creo que uno de los errores se debe a que las imágenes de entrenamiento y de test tienen escalas diferentes

``` python
def caracteristicas(img, c):
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    mascara = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mascara, [c], -1, 255, -1)
    perimetro = cv2.arcLength(c, True)

    return [
    np.log1p(area),
    max(w/h, h/w),
    area/(w*h),
    4*np.pi*area/(perimetro**2 + 1e-6)
]
``` 

para la siguiente parte también se utilizó IA, y es que en esta parte lo que se realiza es el procesamiento de las imágenes originales; StandardScaler, normaliza las características para que tengan escalas comparables, SVC(kernel="rbf") crea un clasificador SVM,y class_weight="balanced" compensa parcialmente las diferencias en el número de ejemplos 

``` python
for archivo, clase in archivos:
    img = cv2.imread(archivo)
    for c in contornos(img):
        if cv2.contourArea(c) > 20:
            X.append(caracteristicas(img, c))
            y.append(clase)

modelo = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=3, class_weight="balanced")
)
```

ahora se intentará extraer las características de la imagen

``` python
img = cv2.imread("MPs_test.jpg")
vista = img.copy()
verdaderas, predichas = [], []
colores = {"FRA": (0,0,255), "PEL": (0,255,0), "TAR": (255,0,0)}

``` 

accedemos a cada columna por su nombre, y extraemos de la imagen la región indicada por el csv

``` python
with open("MPs_test_bbs.csv", newline="") as f:
    for fila in csv.DictReader(f):
        x1, y1, x2, y2 = [
            int(float(fila[k])) for k in ["x_min", "y_min", "x_max", "y_max"]
        ]
        recorte = img[y1:y2+1, x1:x2+1]
        if recorte.size == 0:
            continue
``` 



aplicamos la función contornos, y se descartan los contornos con un área muy pequeña, si hay varios elige la mayor área 
``` python


  candidatos = [c for c in contornos(recorte)
                      if cv2.contourArea(c) > 5]

        if candidatos:
            c = max(candidatos, key=cv2.contourArea)
        else:
            h, w = recorte.shape[:2]
            c = np.array([[[0,0]], [[w-1,0]],
                          [[w-1,h-1]], [[0,h-1]]], dtype=np.int32)
                          
``` 

ahora por ultimo, intentamos predecir con modelo.predict y real = fila["label"] es para comprobar, y dibujamos en la imagen, los elementos encontrados, según el tipo encontrado

``` python
 pred = modelo.predict([caracteristicas(recorte, c)])[0]
        real = fila["label"]
        verdaderas.append(real)
        predichas.append(pred)

        cv2.rectangle(vista, (x1,y1), (x2,y2), colores[pred], 2)
        cv2.putText(vista, pred, (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colores[pred], 2)

clases = ["FRA", "PEL", "TAR"]
print("Exactitud:", round(accuracy_score(verdaderas, predichas), 3))
print("Predicciones:", {c: predichas.count(c) for c in clases})
``` 
y por ultimo, sería mostrar la matriz de confusión, y como de preciso es el modelo

RESULTADO:
aquí se observa que la exactitud no es muy buena, como se señaló antes y es que, aunque selecciona mal la mayoría, las partículas se localizan con las cajas del CSV, pero en lo que se falla es en la clasificación
<img src = "confusion.png" width="500">




LIMITACIONES: los malos resultados de esta última tarea, se pueden deber a varios motivos y uno de ellos es que, las imágenes usadas tienen escala diferente y al hacer log1p(área), depende directamente de la escala, y se perjudica el resultado y otro motivo, es que no se utilizan ni color ni textura, otro motivo del fallo es que en el entrenamiento , el filtro de área > 20 es demasiado pequeño