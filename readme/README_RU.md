# 📚 ebook2audiobook

Конвертация электронных книг в аудиокниги с сохранением глав и метаданных, используются механизмы Calibre и XTTS. Поддерживаются опциональное клонирование голоса и множественные языки!
> [!IMPORTANT]
**Этот инструмент предназначен для использования только с электронными книгами, не защищёнными DRM, приобретёнными законным путём.**  
Авторы не несут ответственности за неправильное использование этого программного обеспечения или любые юридические последствия, связанные с его использованием.  
Используйте этот инструмент ответственно и в соответствии с действующим законодательством.


#### 🖥️ Web-интерфейс
![demo_web_gui](https://github.com/user-attachments/assets/85af88a7-05dd-4a29-91de-76a14cf5ef06)

<details>
  <summary>Больше картинок Web-интерфейса</summary>
<img width="1728" alt="image" src="https://github.com/user-attachments/assets/b36c71cf-8e06-484c-a252-934e6b1d0c2f">
<img width="1728" alt="image" src="https://github.com/user-attachments/assets/c0dab57a-d2d4-4658-bff9-3842ec90cb40">
<img width="1728" alt="image" src="https://github.com/user-attachments/assets/0a99eeac-c521-4b21-8656-e064c1adc528">
</details>

## README.md
- en [English](README.md)
- zh_CN [简体中文](readme/README_CN.md)
- ru [Русский](readme/README_RU.md)


## 🌟 Возможности

- 📖 Преобразование электронных книг в текстовой формат при помощи Calibre.
- 📚 Разбитие электронных книг по главам для аудиоформата.
- 🎙️ Высококачественное преобразование текста в голос при помощи Coqui XTTS.
- 🗣️ Опциональное клонирование голоса на основе вашего голосового файла.
- 🌍 Многоязыковая поддержка (Английский по умолчанию).
- 🖥️ Для работы достаточно всего 4 Гб ОЗУ.

## 🤗 [Демонстрация на Huggingface](https://huggingface.co/spaces/drewThomasson/ebook2audiobookXTTS)
- Пространство на Huggingface работает на бесплатном процессорном уровне, посему не стоит ожидать от него высокой скорости обработки или отсутствие сообщений о таймаутах. Даже и не пытайтесь обработать большие файлы.
- Лучше всего скопировать пространство или запустить приложение локально.

## Бесплатный Google Colab [![Бесплатный Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrewThomasson/ebook2audiobookXTTS/blob/main/Notebooks/colab_ebook2audiobookxtts.ipynb)


## 🛠️ Требования

- Python 3.10
- `coqui-tts` Python package
- Calibre (для конвертации электронных книг)
- FFmpeg (для создания аудиокниг)
- Опционально: собственный файл с голосом для начитки


### 🔧 Установка

1. **Установить Python 3.x** из [Python.org](https://www.python.org/downloads/).

2. **Установить Calibre**:
   - **Ubuntu**: `sudo apt-get install -y calibre`
   - **macOS**: `brew install calibre`
   - **Windows** (Admin Powershell): `choco install calibre`

3. **Установить FFmpeg**:
   - **Ubuntu**: `sudo apt-get install -y ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows** (Admin Powershell): `choco install ffmpeg`

4. **Опционально: установить Mecab** (для не латинских языков):
   - **Ubuntu**: `sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8`
   - **macOS**: `brew install mecab`, `brew install mecab-ipadic`
   - **Windows**: [mecab-website-to-install-manually](https://taku910.github.io/mecab/#download) (Замечание: Японский язык поддерживается ограничено)

5. **Установить пакеты Python**:
   ```bash
   pip install coqui-tts==0.24.2 pydub nltk beautifulsoup4 ebooklib tqdm gradio==4.44.0
   
   python -m nltk.downloader punkt
   python -m nltk.downloader punkt_tab
   ```

   **Для не латинских языков**:
   ```bash
   pip install mecab mecab-python3 unidic
   
   python -m unidic download
   ```

## 🌐 Поддерживаемые языки

- **English (en)**
- **Spanish (es)**
- **French (fr)**
- **German (de)**
- **Italian (it)**
- **Portuguese (pt)**
- **Polish (pl)**
- **Turkish (tr)**
- **Russian (ru)**
- **Dutch (nl)**
- **Czech (cs)**
- **Arabic (ar)**
- **Chinese (zh-cn)**
- **Japanese (ja)**
- **Hungarian (hu)**
- **Korean (ko)**

Указывайте код нужного языка при запуске в безинтерфейсном режиме (в коммандной строке).
## 🚀 Использование

### 🖥️ Запуск Gradio Web-интерфейса

1. **Запустите скрипт**:
   ```bash
   python app.py
   ```

2. **Откройте Web-приложение**: нажмите на ссылку появившуся в окне терминала для доступа к Web-приложению и конвертированию электронных книг.
3. **Для доступа из сети**: добавьте `--share True` в конец команды, наподобие: `python app.py --share True`
- **[Для большего количества параметров]**: используйте `-h` ключ, наподобие: `python app.py -h`

### 📝 Типовое использование в безинтерфейсном режиме

```bash
python app.py --headless True --ebook <path_to_ebook_file> --voice [path_to_voice_file] --language [language_code]
```

- **<path_to_ebook_file>**: путь к файлу электронной книги.
- **[path_to_voice_file]**: путь к примеру голоса, для опционального клонирования голоса для начитки.
- **[language_code]**: по желанию, выбрать язык.
- **[Для большего количества парамтеров]**: используйте `-h` ключ, наподобие `python app.py -h`

### 🧩 Безинтерфейсное использование с индивиуальной моделью XTTS

```bash
python app.py --headless True --use_custom_model True --ebook <ebook_file_path> --voice <target_voice_file_path> --language <language> --custom_model <custom_model_path> --custom_config <custom_config_path> --custom_vocab <custom_vocab_path>
```

- **<ebook_file_path>**: путь к файлу электронной книги.
- **<target_voice_file_path>**: путь к примеру голоса, для опционального клонирования.
- **\<language>**: по желанию, выбрать язык.
- **<custom_model_path>**: путь к `model.pth`.
- **<custom_config_path>**: путь к `config.json`.
- **<custom_vocab_path>**: путь к `vocab.json`.
- **[Для большего количества парамтеров]**: используйте `-h` ключ, наподобие `python app.py -h`


### 🧩 Безинтерфейсое использование с индивидуальной моделью XTTS со ссылкой на Zip-архив содержащий модель тонкой настройки XTTS 🌐

```bash
python app.py --headless True --use_custom_model True --ebook <ebook_file_path> --voice <target_voice_file_path> --language <language> --custom_model_url <custom_model_URL_ZIP_path>
```

- **<ebook_file_path>**: путь к файлу eBook.
- **<target_voice_file_path>**: путь к примеру голоса, для опционального клонирования.
- **\<language>**: по желанию, выбрать язык.
- **<custom_model_URL_ZIP_path>**: путь в виде URL к архиву формата zip с папкой модели. Например, [xtts_David_Attenborough_fine_tune](https://huggingface.co/drewThomasson/xtts_David_Attenborough_fine_tune/tree/main) `https://huggingface.co/drewThomasson/xtts_David_Attenborough_fine_tune/resolve/main/Finished_model_files.zip?download=true`
- Для индивидуальной модели все равно потребуется референсный аудиофайл с голосом:
[референсный аудиофайл с голосом David Attenborough](https://huggingface.co/drewThomasson/xtts_David_Attenborough_fine_tune/blob/main/ref.wav)
- **[Для большего количества парамтеров]**: используйте `-h` ключ, наподобие `python app.py -h`

### 🔍 Для подробного списка всех параметров используйте
```bash
python app.py -h
```
- Будет выведен примерно следующий сприсок ключей:
```bash
использование: app.py [-h] [--share SHARE] [--headless HEADLESS] [--ebook EBOOK] [--voice VOICE]
              [--language LANGUAGE] [--use_custom_model USE_CUSTOM_MODEL]
              [--custom_model CUSTOM_MODEL] [--custom_config CUSTOM_CONFIG]
              [--custom_vocab CUSTOM_VOCAB] [--custom_model_url CUSTOM_MODEL_URL]
              [--temperature TEMPERATURE] [--length_penalty LENGTH_PENALTY]
              [--repetition_penalty REPETITION_PENALTY] [--top_k TOP_K] [--top_p TOP_P]
              [--speed SPEED] [--enable_text_splitting ENABLE_TEXT_SPLITTING]

Преобразование электронных книг в аудиокниги с использованием модели Text-to-Speech (TTS). Вы можете либо использовать
интерфейс Gradio, либо запустить скрипт в безинтерфейсном режиме (командная строка) для прямого конвертирования.

опции:
  -h, --help            Отобразить этот список и выйти
  --share SHARE         Установить в True для включения публичного доступа к Web-интерфейсу Gradio. По умолчанию False.
  --headless HEADLESS   Установить в True для использования безинтерфейсного режима. По умолчанию False.
  --ebook EBOOK         Путь к электронной книге для конвертации. Необходимо для безинтерфейсного режима.
  --voice VOICE         Путь к целевому голосовому файлу для TTS (текст-в-голос). Опционально, используется голос по умолчанию, если путь не указан.
  --language LANGUAGE   Язык для конвертации в аудиокнигу. Варианты: en, es, fr, de,
                        it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko. По умолчанию English (en).
  --use_custom_model USE_CUSTOM_MODEL
                        Установить в True для использования индивидуальной модели TTS. По умолчанию False. Необходимо переключить в
                        True для использования индивидулаьной модели, в противном случае возникнет ошибка.
  --custom_model CUSTOM_MODEL
                        Путь к файлу индивидуальной модели (.pth). Требуется, если используется индивидуальная модель.
  --custom_config CUSTOM_CONFIG
                        Путь к конфигурационнмоу файлу индивидуальной модели (config.json). Требуется, если используется индивидуальная модель.
  --custom_vocab CUSTOM_VOCAB
                        Путь к словарю индивидуальной модели (vocab.json). Требуется, если используется индивидуальная модель.
  --custom_model_url CUSTOM_MODEL_URL
                        URL для скачивания индивидуальной модели в виде zip-архива. Опционально, но если указано, то будет использовано.
                        Примеры включающие модель David Attenborough: 'https://huggingface.co/drewThomasson/xtts_David_Attenborough_fine_tune/resolve/main/Finished_model_files.zip?download=true'. Больше точно-настроенных моделей XTTS можно найти на Hugging Face 'https://huggingface.co/drewThomasson'.
  --temperature TEMPERATURE
                        Температура для модели. По умолчанию 0.65. Чем выше температура, тем более креативным будет синтез голоса, с большим наваждением. Чем меньше, тем более монотонным и спокойным.
  --length_penalty LENGTH_PENALTY
                        Ограничение длинны авторегрессионного декодреа. По умолчанию 1.0. Не применяется к индивидуальным моделям.
  --repetition_penalty REPETITION_PENALTY
                        Ограниечение предотвращающее повторение авторегрессивным декодером за собой. По умолчанию 2.0
  --top_k TOP_K         Сэмплирование Top-k. Меньшее значние приводит к более вероятностному выводу и ускоряют генерацию аудио. По умолчанию 50.
  --top_p TOP_P         Сэмплирование Top-p. Меньшее значние приводит к более вероятностному выводу и ускоряют генерацию аудио. По умолчанию 0.8.
  --speed SPEED         Фактор скорости начитки. Чем больше значение, тем быстрее диктор будет читать текст. По умолчанию 1.0.
  --enable_text_splitting ENABLE_TEXT_SPLITTING
                        Включает разбиение текста на предложения. По умолчаниею True.

Пример: python script.py --headless --ebook path_to_ebook --voice path_to_voice --language en --use_custom_model True --custom_model model.pth --custom_config config.json --custom_vocab vocab.json
```



### 🐳 Использование Docker

Помимо всего прочего, можно использовать Docker для использования конвертера электронных книг в аудиокниги. Этот метод обеспечивает согласованность в различных средах и упрощает настройку.

#### 🚀 Запуск контейнера Docker

Для запуска контейнера Docker и интерфейса Gradio используйте следующую команду:

 -Запуск с использованием только CPU (процессора)
```powershell
docker run -it --rm -p 7860:7860 --platform=linux/amd64 athomasson2/ebook2audiobookxtts:huggingface python app.py
```
 -Запуск с использованием ускорения на GPU (графической карты), поддерживаются только видеокарты NVIDIA
```powershell
docker run -it --rm --gpus all -p 7860:7860 --platform=linux/amd64 athomasson2/ebook2audiobookxtts:huggingface python app.py
```

Эта команда запускает интерфейс Gradio на порту 7860. (localhost:7860)
- Для получения большей информации о доступных командах в безинтерфейсном режиме или предоставление доступа к Gradio в сети, используйте ключ `-h` после имени команды `app.py` в терминале Docker
<details>
  <summary><strong>Пример использования Docker в безинтерфейсном режиме или модификаций параметров + полный гид</strong></summary>
   
## Пример использования Docker в безинтерфейсном режиме

- Сперва необходимо получить свежий контейнер с приложением
```bash 
docker pull athomasson2/ebook2audiobookxtts:huggingface
```

- Прежде чем запустить команду на исполнение, необходимо создать директрую с именем "input-folder" в текущей папке, которая будет подтянута к использованию. В эту папку необходимо помещать файлы, которые будут видны образу Docker
```bash
mkdir input-folder && mkdir Audiobooks
```

- В команде ниже замените **YOUR_INPUT_FILE.TXT** именем файла, который необходимо начитать 

```bash
docker run -it --rm \
    -v $(pwd)/input-folder:/home/user/app/input_folder \
    -v $(pwd)/Audiobooks:/home/user/app/Audiobooks \
    --platform linux/amd64 \
    athomasson2/ebook2audiobookxtts:huggingface \
    python app.py --headless True --ebook /home/user/app/input_folder/YOUR_INPUT_FILE.TXT
```

- И на этом это все! 

- Начитанная аудиокнига будет сформирована в папке Audiobooks, которая будет создана в вашей локальной директории, в которой был осуществлен запуск Docker


## Для получения помощи по параметрам, необходимо запустить следующую команду 

```bash
docker run -it --rm \
    --platform linux/amd64 \
    athomasson2/ebook2audiobookxtts:huggingface \
    python app.py -h

```


и вывод будет следующим

```bash
user/app/ebook2audiobookXTTS/input-folder -v $(pwd)/Audiobooks:/home/user/app/ebook2audiobookXTTS/Audiobooks --memory="4g" --network none --platform linux/amd64 athomasson2/ebook2audiobookxtts:huggingface python app.py -h
starting...
Преобразование электронных книг в аудиокниги с использованием модели Text-to-Speech (TTS). Вы можете либо использовать
интерфейс Gradio, либо запустить скрипт в безинтерфейсном режиме (командная строка) для прямого конвертирования.

опции:
  -h, --help            Отобразить этот список и выйти
  --share SHARE         Установить в True для включения публичного доступа к Web-интерфейсу Gradio. По умолчанию False.
  --headless HEADLESS   Установить в True для использования безинтерфейсного режима. По умолчанию False.
  --ebook EBOOK         Путь к электронной книге для конвертации. Необходимо для безинтерфейсного режима.
  --voice VOICE         Путь к целевому голосовому файлу для TTS (текст-в-голос). Опционально, используется голос по умолчанию, если путь не указан.
  --language LANGUAGE   Язык для конвертации в аудиокнигу. Варианты: en, es, fr, de,
                        it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko. По умолчанию English (en).
  --use_custom_model USE_CUSTOM_MODEL
                        Установить в True для использования индивидуальной модели TTS. По умолчанию False. Необходимо переключить в
                        True для использования индивидулаьной модели, в противном случае возникнет ошибка.
  --custom_model CUSTOM_MODEL
                        Путь к файлу индивидуальной модели (.pth). Требуется, если используется индивидуальная модель.
  --custom_config CUSTOM_CONFIG
                        Путь к конфигурационнмоу файлу индивидуальной модели (config.json). Требуется, если используется индивидуальная модель.
  --custom_vocab CUSTOM_VOCAB
                        Путь к словарю индивидуальной модели (vocab.json). Требуется, если используется индивидуальная модель.
  --custom_model_url CUSTOM_MODEL_URL
                        URL для скачивания индивидуальной модели в виде zip-архива. Опционально, но если указано, то будет использовано.
                        Примеры включающие модель David Attenborough: 'https://huggingface.co/drewThomasson/xtts_David_Attenborough_fine_tune/resolve/main/Finished_model_files.zip?download=true'. Больше точно-настроенных моделей XTTS можно найти на Hugging Face 'https://huggingface.co/drewThomasson'.
  --temperature TEMPERATURE
                        Температура для модели. По умолчанию 0.65. Чем выше температура, тем более креативным будет синтез голоса, с большим наваждением. Чем меньше, тем более монотонным и спокойным.
  --length_penalty LENGTH_PENALTY
                        Ограничение длинны авторегрессионного декодреа. По умолчанию 1.0. Не применяется к индивидуальным моделям.
  --repetition_penalty REPETITION_PENALTY
                        Ограниечение предотвращающее повторение авторегрессивным декодером за собой. По умолчанию 2.0
  --top_k TOP_K         Сэмплирование Top-k. Меньшее значние приводит к более вероятностному выводу и ускоряют генерацию аудио. По умолчанию 50.
  --top_p TOP_P         Сэмплирование Top-p. Меньшее значние приводит к более вероятностному выводу и ускоряют генерацию аудио. По умолчанию 0.8.
  --speed SPEED         Фактор скорости начитки. Чем больше значение, тем быстрее диктор будет читать текст. По умолчанию 1.0.
  --enable_text_splitting ENABLE_TEXT_SPLITTING
                        Включает разбиение текста на предложения. По умолчаниею True.

Пример: python script.py --headless --ebook path_to_ebook --voice path_to_voice --language en --use_custom_model True --custom_model model.pth --custom_config config.json --custom_vocab vocab.json
```
</details>

#### 🖥️ Docker Web-интерфейс 
![demo_web_gui](https://github.com/user-attachments/assets/85af88a7-05dd-4a29-91de-76a14cf5ef06)

<details>
  <summary>Нажмите для просмотра изображений Web-интерфейса</summary>
<img width="1728" alt="image" src="https://github.com/user-attachments/assets/b36c71cf-8e06-484c-a252-934e6b1d0c2f">
<img width="1728" alt="image" src="https://github.com/user-attachments/assets/c0dab57a-d2d4-4658-bff9-3842ec90cb40">
<img width="1728" alt="image" src="https://github.com/user-attachments/assets/0a99eeac-c521-4b21-8656-e064c1adc528">
</details>

### 🛠️ Для индивидуальных Xtts моделей

Модели создаются для лучшего использования с конкретным голосом. Проверьте различные модели на страничке Hugging Face [тут](https://huggingface.co/drewThomasson).

Для использования индивидуальных моделей, используйте ссылку на архив с моделью `Finished_model_files.zip`, например:
[David Attenborough точно настроенный голос Finished_model_files.zip](https://huggingface.co/drewThomasson/xtts_David_Attenborough_fine_tune/resolve/main/Finished_model_files.zip?download=true)

Для индивидуальной модели также необходим файл с голосом:
[файл с голосом David Attenborough](https://huggingface.co/drewThomasson/xtts_David_Attenborough_fine_tune/blob/main/ref.wav)



Больше информации можно найти на [странице Dockerfile Hub]([https://github.com/DrewThomasson/ebook2audiobookXTTS](https://hub.docker.com/repository/docker/athomasson2/ebook2audiobookxtts/general)).

## 🌐 Точно отстроенные модели Xtts models

Для поиска уже подготовленных точно настроенных моделей XTTS обратитесь к [этой страничке на Hugging Face](https://huggingface.co/drewThomasson) 🌐. Ищите модели которые имеют в наименовании "xtts fine tune".

## 🎥 Демонстрация

Голос ненастного дня

https://github.com/user-attachments/assets/8486603c-38b1-43ce-9639-73757dfb1031

Голос David Attenborough

https://github.com/user-attachments/assets/47c846a7-9e51-4eb9-844a-7460402a20a8


## 🤗 [Демонстрация в пространстве Huggingface](https://huggingface.co/spaces/drewThomasson/ebook2audiobookXTTS)
- Пространства на Huggingface работают на бесплатном уровне процессоров, поэтому выполнение очень медленное и частво возникают ошибки связанные с истечением времени. Не пытайтесь преобразовывать большие файлы.
- Лучше всего клонировать пространство или запускать его локально.

## Бесплатный Google Colab [![Бесплатный Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrewThomasson/ebook2audiobookXTTS/blob/main/Notebooks/colab_ebook2audiobookxtts.ipynb)



## 📚 Поддерживаемые форматы электронных книг

- **Можно**: `.epub`, `.pdf`, `.mobi`, `.txt`, `.html`, `.rtf`, `.chm`, `.lit`, `.pdb`, `.fb2`, `.odt`, `.cbr`, `.cbz`, `.prc`, `.lrf`, `.pml`, `.snb`, `.cbc`, `.rb`, `.tcr`
- **Лучше**: `.epub` или `.mobi` для автоматического определения глав.

## 📂 Вывод

- Создается файл с расширением `.m4b`, содержащий метаданные и главы.
- **Пример вывода**: ![Пример](https://github.com/DrewThomasson/VoxNovel/blob/dc5197dff97252fa44c391dc0596902d71278a88/readme_files/example_in_app.jpeg)

## 🛠️ Частые проблемы:
- "Очень медленно!" - При конвертации только на CPU она происходит медленно, единственный способ ускорения - использовать GPU от NVIDIA: [Обсуждение](https://github.com/DrewThomasson/ebook2audiobookXTTS/discussions/19#discussioncomment-10879846). Для быстрой многоязыковой генерации аудио, рекомендуется использовать другой проект [использующий piper-tts](https://github.com/DrewThomasson/ebook2audiobookpiper-tts). (Тем не менее, в нем нет функции клонирования голоса без лишней суеты и он воспроизводит голоса в качестве siri, но он намного быстрее работает на CPU.)
- "У меня проблема с зависимостями" - Просто используейте Docker. Образы в Docker самодостаточны, имеют, в том числе режим работы с конмандной строкой, ключ для вывода помощи.
- "У меня проблема с обрезаным аудио!" - создайте запись о проблеме, автор не говорит на каждом из поддерживаемых языков и ему требуется помощь по автоматическому разбиению текста на предложения в поддерживаемых языках.😊
- "Процесс застопорился на 30% в Web-интерфейсе!" - Отображение прогресса в Web-интерфейсе выполнено на базовом уровне и содержит всего 3 шага, для контроллирования процесса посматривайте в терминальный вывод, где и отображается обработка текущего предложения.

## С чем требуется помощь! 🙌 
## [Полный список тут](https://github.com/DrewThomasson/ebook2audiobookXTTS/issues/32)
- Любая помощь от людей говорящий на поддерживаемых языках для более корретного разбиения текста на предложения.
- Потенциальная помощь в создании инструкций для разных языков (автор знает только английский 😔).

## 🙏 Отдельные спасибо

- **Coqui TTS**: [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
- **Calibre**: [Calibre Website](https://calibre-ebook.com)

- [@shakenbake15 за лучший способ сохранения глав](https://github.com/DrewThomasson/ebook2audiobookXTTS/issues/8) 

