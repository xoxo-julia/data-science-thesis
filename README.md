# Выпускная квалификационная работа по курсу «Инженер данных (Data engineer Pro)»

## Тема

Классификация госконтрактов по объектам закупки.

## Описание

В соответствии с Федеральным законом «О контрактной системе» (44-ФЗ) государственный контракт (госконтракт) — это соглашение между поставщиком и органами власти федерального, регионального или муниципального уровней. Карточки госконтрактов хранятся в специальном реестре. Карточка госконтракта содержит информацию о контракте: описание, сроки, исполнителей и т.д. В том числе хранится общероссийский классификатор продукции по видам экономической деятельности (ОКПД-2). Зачастую ОКПД-2 заявляется ошибочный. В этом заключается проблема – выделить ошибочно обозначенные госконтракты и переназначить их ОКПД-2.

Входными данными в работе являются подготовленные гос. заказчиком данные карточек госконтрактов с ftp.zakupki.gov.ru. В результате выполнения задания получен классификатор госконтрактов по ОКПД-2 в соответствии с технических заданием, разработано приложение, позволяющее на основе входных данных предоставлять пользователю прогноз.

## Актуальность

Создание классификатора госконтрактов по ОКПД-2 в соответствии с объектом закупки, позволит эффективно перераспределять контракты по соответствующим им группам.

## Установка приложения

Выполните ниже приведенные команды чтобы создать виртуальное окружение и установить приложение:

```bash
python -m venv venv
source ./venv/bin/activate
# .\venv\Scripts\activate - Windows
pip install -r webapp/requirements.txt
```

Для запуска используйте комманду:

```bash
python webapp/server.py
```

Проект будет доступен по адресу http://localhost:8080

## Используемые библиотеки

flask для создания веб-приложения Flask, получения запросов от внешнего интерфейса и отправки ему ответов.
waitress для запуска веб-сервера и обслуживания на нем веб-приложения Flask.
nltk для обработки текста.
catboost для работы модели и предсказаний.
pandas для работы с датафреймами.
pyarrow для загрузки .parquet файлов.
