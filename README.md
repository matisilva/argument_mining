# Argument mining with BiLSTM

## Objetivos
Basandonos en los repositorios de [UKPLab](https://github.com/UKPLab):

- https://github.com/UKPLab/acl2017-neural_end2end_am
- https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf

Adaptar el código para..
- Desarrollar un entorno para la identificación y etiquetado de argumentos en textos del idioma inglés.
- Proponer alternativas de embeddings distintos para evaluación.
- Flexibilizar el input/output de texto para permitir diversas entradas y salidas según el caso.

## Instrucciones para su uso
### Entorno virtual python

```
$ virtualenv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
```

### Archivos de entrenamiento

dentro de ```/data/am_/``` se encontrarán los archivos de entrenamiento dev, train y test para ser directamente aplicados al modelo.

Para obtener archivos para evaluar el modelo, mediante el siguiente comando se descargarán un set de textos con sus dos formatos: etiquetado y no para una posterior evaluacion de performance.  

TODO: Hacer script bash
```
$ wget https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/data/argument-recognition/ArgumentAnnotatedEssays-1.0.zip
$ unzip ArgumentAnnotatedEssays-1.0.zip
$ cd ArgumentAnnotatedEssays-1.0
$ unzip brat-project.zip
$ cd ..
$ mv ArgumentAnnotatedEssays-1.0/brat-project example_essays
```
Tendremos ahora una carpeta "example_essays" con archivos de tipo .txt los cuáles serán el input de nuestra red, mientras que los .ann será nuestra verdad a comparar para la evaluación del sistema.

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
