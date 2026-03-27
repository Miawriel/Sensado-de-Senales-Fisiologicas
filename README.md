# Práctica XAI — Detección de enojo con señales fisiológicas

Esta práctica compara dos enfoques de clasificación:

- **V1 (Raw):** usa features más generales del dataset original.
- **V2 (Anger-detection):** usa solo features fisiológicas seleccionadas y se analiza con SHAP.

## Objetivo
Entrenar un modelo, interpretar las features más importantes con SHAP y comparar el comportamiento de un modelo general vs uno basado solo en señales fisiológicas.

## Features usadas en V2
- rmssd_post
- gsr_change_pct
- scl_delta
- mean_eeg_af8_post

## Resultados principales
- V1 tuvo mejor desempeño global.
- V2 fue más fácil de interpretar, pero predijo peor.
- En V2, las features más importantes fueron:
  - scl_delta
  - gsr_change_pct
  - rmssd_post
  - mean_eeg_af8_post

## Archivos importantes
- `Practica_XAI_Blockchain.ipynb`: notebook principal
- `dataset_B_engineered.csv`: dataset usado en V2
- `xai_chain_logger.py`: soporte para registro
- `output_v2_anger/`: gráficas y resultados generados

## Ejemplos de visualización
Incluye:
- matriz de confusión
- SHAP summary plot
- SHAP bar plot
- waterfall plots
