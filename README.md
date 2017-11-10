# Argument mining with BiLSTM

## Objetivo

## Instrucciones para su uso
### Entorno virtual python

```
$ virtualenv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
```

### Archivos de entrenamiento
TODO: Agregar los {dev,train,test}.txt a data

dentro de ```/data``` se encontrarán los archivos de entrenamiento dev, train y test para ser directamente aplicados al modelo.

Para obtener archivos para evaluar el modelo, mediante el siguiente comando se descargarán un set de textos con sus dos formatos: etiquetado y no para una posterior evaluacion de performance.  

TODO: completar con la direccion donde se descargan los archivos essay_0XX.{txt, ann}
```
$ wget ...
```

### Obtener modelo entrenado.
TODO: Aca hay que configurar para que por parametro seleccione el embedding [levy, glove, word2vec]
```
$ python Train_AM.py
```

TODO: marcar como se guardan los modelos entrenados y cuando corta el entrenamiento(>5 epochs without changes).

### Etiquetar texto con modelo entrenado

TODO: completar comando para etiquetar texto
```
$ python RunModel /models/AM_
```

### Evaluar eficiencia
TODO: generar eval.py que evalue los output etiquetados comparandolos con los reales. True vs predicted y reporte los errores


## Análisis de resultados
