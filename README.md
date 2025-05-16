# CycleGAN Art Style Transfer

## Описание

Проект позволяет переносить стиль известных художников (например, Ван Гога) на обычные фотографии с помощью генеративных нейросетей (CycleGAN).
Включает:
- обучение и дообучение модели в Google Colab или локально,
- оценку качества генераций (FID),
- улучшение качества изображений (супер-разрешение),
- простой веб-интерфейс для генерации изображений на локальном компьютере.

---

## Как использовать

### 1. Обучение и дообучение модели

- Используйте Colab-блокнот или main.py для обучения/дообучения модели.
- Сохраняйте веса генератора в файл `generator_B2A.pth`.

### 2. Перенос весов в веб-интерфейс

- Скачайте файл `generator_B2A.pth` из Colab или получите после локального обучения.
- Поместите его в папку `van_gogh_cyclegan` вашего локального проекта.

### 3. Запуск веб-интерфейса

1. Установите зависимости:
   ```
   pip install -r requirements.txt
   pip install flask
   ```

2. Запустите приложение:
   ```
   python app.py
   ```

3. Перейдите в браузере по адресу [http://localhost:5055](http://localhost:5055).

4. Загрузите свою фотографию — получите результат в стиле Ван Гога!

---

## Оценка качества и улучшение изображений

- Для оценки качества используйте FID (см. инструкции в Colab или main.py).
- Для увеличения разрешения используйте Real-ESRGAN (см. инструкции ниже).

### Супер-разрешение (Real-ESRGAN)

1. Установите:
   ```
   pip install realesrgan
   ```
2. Пример использования:
   ```python
   from realesrgan import RealESRGAN
   from PIL import Image
   import torch
   model = RealESRGAN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale=4)
   model.load_weights('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/RealESRGAN_x4.pth')
   img = Image.open('results/your_image.png').convert('RGB')
   sr_img = model.predict(img)
   sr_img.save('results/your_image_sr.png')
   ```

---

## Структура проекта

```
van_gogh_cyclegan/
├── app.py                # Flask-приложение
├── main.py               # Скрипт для обучения/дообучения
├── generator_B2A.pth     # Веса генератора (фото -> стиль)
├── requirements.txt      # Зависимости
├── README.md             # Инструкция
├── templates/
│   └── index.html        # HTML-шаблон для веб-интерфейса
├── uploads/              # Загруженные пользователем фото
└── results/              # Сгенерированные изображения
```

---

## Контакты

Если возникли вопросы — пишите! 