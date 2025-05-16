# CycleGAN Van Gogh

## Описание

Проект позволяет переносить стиль Ван Гога на обычные фотографии с помощью генеративных нейросетей CycleGAN.

main.py содержит в себе функцию для локального обучения, ознакомится с результатом и поцессом можно в блокноте:
https://colab.research.google.com/drive/1Yvv9B4gtvnsYXRVGmF4B0h2xpoQ864Ie?usp=sharing
Дообучение модели и суперразрешение:
https://colab.research.google.com/drive/1PL6a8ZMLBYWXVAhYLrxPE6JOonXNhQk1?usp=sharing

### Запуск веб-интерфейса

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

4. Загрузите свою фотографию — программа выдаст результат в стиле Ван Гога!


Подробный отчет в текстовом файле
