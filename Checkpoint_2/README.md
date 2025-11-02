# Годовой проект HSE ИИ25 — Команда 33  
## Детектирование аномалий в данных распределённых облаков

## Краткое описание выполненного второго этапа (Checkpoint 2)

**Описание и ссылки:**
1. Анализ андроид логов, применение drain для шаблонизации логов, подсчет статистик по шаблонам (Павел Ванюшин). ([Тетрадка](https://github.com/Vanarty/anomaly-detection-distributed-clouds/Checkpoint_2/Anroid_v1_log_analysis.ipynb)). Анализ логов HDFS v1, применение drain и spell библиотек для шаблонизации логов, подсчет статистик по шаблонам (Павел Ванюшин). ([Тетрадка](https://github.com/Vanarty/anomaly-detection-distributed-clouds/Checkpoint_2/HDFS_v1_templating.ipynb)), ([Архив с получившимися шаблонами](https://github.com/Vanarty/anomaly-detection-distributed-clouds/Checkpoint_2/HDFS_templates.rar))
2. Подробный анализ с выводом графиков логов HDFS v1 c выводами (Андрей Богданов). ([Тетрадка](https://github.com/Vanarty/anomaly-detection-distributed-clouds/Checkpoint_2/EDA_HDFS.ipynb));
3. Сгенерированные шаблоны и структурированные лог-файлы для 2k - HDFS 2k (Виталий Кузнецов). ([Тетрадка](https://github.com/Vanarty/anomaly-detection-distributed-clouds/Checkpoint_2/log_parsers.ipynb)), ([Структурированные логи](https://github.com/Vanarty/anomaly-detection-distributed-clouds/Checkpoint_2/structured.zip));
4. Парсинг сырых логов HDFS full (>11 млн. строк), препроцессинг дат и времени, вывод статистик по датам, получение BERT эмбеддингов + PCA + HDBSCAN кластеризация на части данных. (Артём Иванов). ([Тетрадка](https://github.com/Vanarty/anomaly-detection-distributed-clouds/Checkpoint_2/semantic_PCA_HDBSCAN.ipynb)).

###  Участники:
- <Участник 1, Иванов Артём> — @Vanarti, Vanarty
- <Участник 2, Богданов Андрей> — @wanna_sleeeep, andrewb-codes
- <Участник 3, Кузнецов Виталий> — @pismith, Vitaly
- <Участник 4, Ванюшин Павел> — @LepoMepo, LepoMepo
