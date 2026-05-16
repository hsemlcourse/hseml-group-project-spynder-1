# ML Project — Предсказание спроса на городской велопрокат

**Студент:** Басов Андрей Игоревич

**Группа:** БИВ232

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)

## Описание задачи

**Задача:** Регрессия - предсказать количество поездок на велосипедах на основе погодных условий и временных факторов

**Датасет:** [London Bike Sharing Dataset](https://www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset)

**Целевая метрика:** RMSE

RMSE является основной метрикой, так как сильно штрафует модель за большие ошибки, что критично для предсказания спроса на велопрокат. Лучше ошибаться много, но чуть-чуть, чем ошибаться редко, но сильно.
Вторичными метриками являются MAE и R2, выбранные за их простоту интерпретации.

## Структура репозитория

```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── app.py                  # Приложение на FastAPI
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── .flake8                     # Настройки flake8
├── .dockerignore               # Исключения для Docker-контекста
├── Dockerfile                  # Сборка Docker-образа с API и моделью
├── docker-compose.yml          # Запуск проекта через Docker Compose
├── Makefile                    # Команды для lint, fix, local run и Docker
├── pyproject.toml              # Конфигурация ruff
├── requirements.txt            # Зависимости проекта
└── README.md                   # Этот файл
```

## Запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/hsemlcourse/hseml-group-project-spynder-1
cd hseml-group-project-spynder-1
```

### 2. Создать виртуальное окружение

- Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

- Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Проверка стиля кода

Проверка Python-кода:

```bash
python -m flake8 src tests
```

Линтинг ноутбуков:

```bash
python -m nbqa flake8 notebooks
```

Автоисправление замечаний в коде и ноутбуках:

```bash
make fix
```

Полная проверка проекта:

```bash
make lint
```

### 5. Локальный запуск

Обучение и сохранение финальной модели:

```bash
python src/modeling.py
```

То же самое через `Makefile`:

```bash
make train
```

Запуск `FastAPI` локально:

```bash
uvicorn src.app:app --reload
```

или

```bash
make run-api
```

После запуска локально будут доступны:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`
- `http://localhost:8000/predict`

### 6. Запуск в Docker

Docker-образ собирает окружение проекта, обучает финальную модель `CatBoost` и сохраняет ее в `models/`. После запуска контейнера автоматически поднимается `FastAPI`-приложение.

Сборка и запуск API:

```bash
docker compose up --build
```

То же самое через `Makefile`:

```bash
make docker-up
```

После запуска будут доступны:

- `http://localhost:8000` — корень приложения
- `http://localhost:8000/docs` — Swagger UI с документацией
- `http://localhost:8000/health` — healthcheck
- `http://localhost:8000/predict` — POST-ручка для предсказания спроса

Если образ уже собран, приложение можно поднять без пересборки:

```bash
docker compose up
```

Отдельная сборка образа:

```bash
docker compose build
```

или

```bash
make docker-build
```

Остановка контейнеров:

```bash
make docker-down
```

Запуск линтера внутри контейнера:

```bash
docker compose run --rm ml-project make lint
```

Пример тела запроса для `POST /predict`:

```json
{
  "weather_code": 1,
  "season": 1,
  "is_holiday": 0,
  "is_weekend": 0,
  "hour": 8,
  "day_of_week": 2,
  "month": 7,
  "t1": 18.5,
  "t2": 17.0,
  "hum": 65.0,
  "wind_speed": 12.0
}
```

Пример ответа:

```json
{
  "count": 1234.56
}
```

## Данные

- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные

## Результаты

Итоговые результаты экспериментов на `validation` из `03_experiments.ipynb`:

| Модель | RMSE_val | R2_val | Комментарий |
|--------|----------|--------|-------------|
| CatBoost | 266.4703 | 0.9529 | Лучшая модель на validation |
| LightGBM | 281.9683 | 0.9473 | Один из лучших результатов |
| XGBoost | 284.6780 | 0.9462 | Очень близок к LightGBM |
| Random Forest | 293.7703 | 0.9428 | Сильный ансамбль, но слабее boosting |
| Decision Tree | 334.2247 | 0.9259 | Лучше baseline, но хуже ансамблей |
| SVR | 971.5045 | 0.3739 | Существенно уступает деревьям |
| Ridge | 1073.3268 | 0.2358 | Линейная модель хуже справляется с нелинейной структурой спроса |
| KNN | 1113.8493 | 0.1770 | Худший результат среди протестированных моделей |
| Linear Regression | 998.83 | 0.22 | Baseline |

Финальная модель `CatBoost` после переобучения на `train + val` получила результаты:
- **RMSE:** 313.0022
- **R2:** 0.9231

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
