# Критерии и что нужно сделать отдельно

| Критерий | В проекте уже есть | Что тебе нужно сделать отдельно |
|---|---|---|
| Корректная структура GitHub-репозитория | Папки `data`, `notebooks`, `src`, `app`, `img`, `report` | Создать GitHub-репозиторий и загрузить проект |
| Первичный анализ данных / EDA | `notebooks/01_eda.ipynb` | Открыть notebook и при необходимости выполнить ячейки |
| Очистка данных и обоснование | `src/data_cleaning.py`, описание в отчёте | На защите объяснить: медиана для числовых, мода для категорий, IQR для выбросов |
| Качество визуализаций | PNG-графики в `img/`, интерактивные Plotly-графики | Добавить скриншот запущенного дашборда в `img/` |
| Интерактивный дашборд и фильтры | `app/streamlit_app.py` | Запустить через `streamlit run app/streamlit_app.py` |
| Математико-статистический анализ | `data/processed/math_statistics.csv` и notebook | Показать таблицу статистики на защите |
| ML-задача | KMeans-кластеризация | Объяснить, что регрессия невозможна без `SalePrice` |
| Визуализация ML | `07_ml_clusters_pca.png`, `08_elbow_method.png` | Показать графики и объяснить кластеры |
| Итоговый отчёт | `report/final_report.md` | Вставить свою ссылку GitHub |
| README и воспроизводимость | `README.md`, `requirements.txt` | Проверить запуск на своём ПК |

## Обязательные действия перед сдачей
1. Создай публичный репозиторий на GitHub.
2. Загрузи туда все файлы проекта.
3. Сделай минимум 5 коммитов, например:
   - `Initial project structure`
   - `Add EDA notebook`
   - `Add data cleaning and visualizations`
   - `Add Streamlit dashboard`
   - `Add ML model and final report`
4. Запусти проект:
   ```bash
   pip install -r requirements.txt
   streamlit run app/streamlit_app.py
   ```
5. Сделай скриншот дашборда и положи его в `img/dashboard_screen.png`.
6. Вставь ссылку GitHub в `README.md` и `report/final_report.md`.
