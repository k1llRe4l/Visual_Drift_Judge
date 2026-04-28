# VISUAL DRIFT JUDGE

Проект Печерицы Кирилла Павловича, студента группы ПИ22-1 факультета ИТиАБД Финансового Университета при правительстве РФ.

## Описание проекта

Данный проект представляет собой веб-сервис на основе моделей машинного зрения для автоматического анализа записи дрифт-заезда и последующего выставления оценок по заданным критериям: траектория (Line) и угол (Angle). Проект позволяет создавать и сохранять конфиги оцениваемых участков, задавать целевые углы в клиппинг-зонах и параметры начисления баллов, загружать видео заезда и получать обработанный ролик с визуальной разметкой, прогрессом анализа и итоговой оценкой по траектории, углу и общему результату.

## Демонстрация функционала

Необработанное изображение:
![Необработанный результат](/static/misc/non_evaluated_screen.jpg)
Обработанное изображение:
![Обработанный результат](/static/misc/evaluated_screen.jpg)

## Системные требования

Для корректной работы установите все библиотеки перечисленные в ***requirements.txt***
Установите компоненты CUDA Toolkit и cuDNN требуемых версий:
- CUDA 12.0: https://developer.nvidia.com/cuda-12-0-0-download-archive
- cuDNN 9.21: https://developer.nvidia.com/cudnn-downloads
> **Внимание:** Работает только с видеокартами NVIDIA. При отсутствии обсчет будет производится на мощностях процессора и займет значительно больше времени.

## Запуск

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

После запуска открыть в браузере:

```text
http://127.0.0.1:5000/
```

## Ограничения
<details>
  <summary>Развернуть</summary>

На данный момент система обучена только на данных полученных в симуляторе Assetto Corsa. Для того чтобы система корректно обработала вашу запись, нужно сделать следующее:
- Assetto Corsa должна быть куплена в [**Steam**](https://store.steampowered.com/app/244210/Assetto_Corsa/)
- Установите [**Content Manager**](https://assettocorsa.club/content-manager.html) и **Custom Shaders Patch**
- В настройках по пути **Settings ➔ Assetto Corsa ➔ Chase Camera** установите следующие значения для любой из камер:
![Настройки камеры](/static/misc/chase_cam.png)
-  В Content Manager в настройках по пути **Settings ➔ Custom Shaders ➔ Camera ➔ Chase** установите флажок "Active" и Script "Basic".
![Скрипт камеры](/static/misc/csp_chase.png)
- Отключите дым в **Settings ➔ Assetto Corsa ➔ Video ➔ Smoke Generation** и отключите партиклы в **Settings ➔ Custom Shaders Patch ➔ Graphics ➔ Particles FX** убрать флажок "New Smoke and Dust" 
![Выключение дыма](/static/misc/smoke.png)
![Выключение дыма CSP](/static/misc/csp_particles.png)

### Осуществление записи

Записывайте заезд любым удобным для вас способом через Replay или в режиме Live.
</details>

# **Важно**: для корректной работы системы используйте прямоугольные белые клиппинг-зоны типа **"Touch-and-go"** на треке.
![Зона контроля](/static/misc/clip.jpg)

## Скриншоты интерфейса
<details>
  <summary>Развернуть</summary>
  
Выбор конфигурации:
![Выбор конфигурации](/static/misc/config_select.png)
Настройки конфигурации:
![Редактор конфигурации](/static/misc/config_edit.png)
Окно загрузки видео и консоль логирования обработчика:
![Окно загрузки видео](/static/misc/video_evaluation.png)
Окно вывода результата и загрузки ролика:
![Окно вывода результата](/static/misc/result_screen.png)
</details>