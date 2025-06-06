import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from io import StringIO

# Настройка бокового меню
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите страницу:",
    ["Информация о разработчике", "Информация о наборе данных", "Визуализации зависимостей",
     "Прогнозирование стоимости"],
    index=0
)

# Основной контент приложения
if page == "Информация о разработчике":
    st.title("Информация о разработчике")
    st.write("### ФИО: Кронштадтов Максим Сергеевич")
    st.write("### Номер учебной группы: МО-231")
    st.write("### Тема РГР: Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных")

elif page == "Информация о наборе данных":
    st.title("Информация о наборе данных")
    st.write("### Описание предметной области")
    st.write(
        "Датасет содержит информацию о подержанных автомобилях, включая их технические характеристики, пробег и другие параметры.")

    st.write("### Описание признаков")
    st.write("""
    1. **Year** — год выпуска автомобиля.  
    2. **Style** — тип кузова (например, Sedan, SUV, Hatchback).  
    3. **Distance** — пробег автомобиля (в км).  
    4. **Engine capacity** — объём двигателя (вкубических сантиметрах).  
    5. **Fuel type** — тип топлива (бензин, дизель, электро, гибрид).  
    6. **Transmission** — тип трансмиссии (автоматическая, механическая, роботизированная).  
    7. **Price(euro)** — цена автомобиля в евро (целевая переменная).  
    """)

    st.write("### Особенности предобработки данных")
    st.write("""
    1. **Удаление пропущенных значений** — строки с отсутствующими данными были удалены.  
    2. **Нормализация данных** — числовые признаки (пробег, объём двигателя) приведены к единой шкале.  
    3. **Кодирование категориальных признаков** — текстовые значения (тип топлива, трансмиссия) преобразованы в числовые метки.  
    """)

elif page == "Визуализации зависимостей":
    st.title("Визуализации зависимостей в наборе данных")

    df = pd.read_csv("updated_dataset1.csv")
    st.write("### Гистограммы всех числовых признаков")
    fig, ax = plt.subplots(figsize=(15, 10))
    df.hist(ax=ax)
    st.pyplot(fig)

    st.write("### Средняя цена по годам выпуска")
    avg_price_by_year = df.groupby('Year')['Price(euro)'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=avg_price_by_year, x='Year', y='Price(euro)', marker='o', ax=ax)
    st.pyplot(fig)

    st.write("### Округлённая матрица корреляции")
    corr_matrix = df.corr()
    corr_matrix_rounded = corr_matrix.round(2)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix_rounded,annot=True,cmap='coolwarm',fmt='.2f',ax=ax)
    ax.set_title("Округлённая матрица корреляции")
    st.pyplot(fig)

elif page == "Прогнозирование стоимости":
    st.title("Прогнозирование стоимости автомобиля")
    st.markdown('<p class="big-font">Прогнозирование рыночной стоимости автомобиля в евро на основе характеристик</p>',
                unsafe_allow_html=True)

    # Загрузка моделей
    @st.cache_resource
    def load_models():
        models = {'Bagging Regression': pickle.load(open('bagging_reg.pkl', 'rb')),'CatBoost Regression': pickle.load(open('cat_reg.pkl', 'rb')),'Elastic Net': pickle.load(open('elastic_net.pkl', 'rb')),'Gradient Boost Regression': pickle.load(open('gb_reg.pkl', 'rb')),'Stacking Regression': pickle.load(open('stacking_reg.pkl', 'rb'))}
        return models
    try:
        models = load_models()
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        st.stop()

    tab1, tab2 = st.tabs(["Загрузить CSV файл", "Ручной ввод"])

    with tab1:
        st.header("Пакетное прогнозирование")
        st.write("Загрузите CSV файл с данными об автомобилях для массового расчета стоимости")

        uploaded_file = st.file_uploader("Выберите CSV файл с данными",type=["csv"],help="Файл должен содержать колонки: Year, Distance, Engine_capacity(cm3), Style_ID, Transmission_ID, Fuel_type_ID")

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['Year', 'Distance', 'Engine_capacity(cm3)', 'Style_ID', 'Transmission_ID', 'Fuel_type_ID']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    st.error(f"Отсутствуют обязательные колонки: {', '.join(missing_cols)}")
                else:
                    st.success("Файл успешно загружен!")
                    st.dataframe(df.head())

                    model_choice = st.selectbox("Выберите модель для прогнозирования",options=list(models.keys()),index=0,key="model_csv")

                    if st.button("Рассчитать стоимость", key="predict_csv"):
                        with st.spinner("Выполняется прогнозирование..."):
                            model = models[model_choice]
                            predictions = model.predict(df)
                            result_df = df.copy()
                            result_df["Predicted Price(euro)"] = predictions.round(2)
                            st.success("Прогнозирование завершено!")
                            st.dataframe(result_df)

                            csv = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(label="Скачать результаты",data=csv,file_name="car_price_predictions.csv",mime="text/csv")
            except Exception as e:
                st.error(f"Ошибка обработки файла: {str(e)}")

    with tab2:
        st.header("Индивидуальное прогнозирование")
        st.write("Введите параметры автомобиля для расчета стоимости")

        with st.form("car_form"):
            col1, col2 = st.columns(2)

            with col1:
                Year = st.number_input(
                    "Год выпуска (Year)*",min_value=1900,max_value=2025,value=2020)

                style_options = {'Hatchback': 1,'Universal': 2,'Microvan': 3,'SUV': 4,'Sedan': 5,'Coupe': 6,'Crossover': 7,'Minivan': 8,'Pickup': 9,'Combi': 10,'Cabriolet': 11,'Roadster': 12}
                selected_style = st.selectbox("Тип кузова (Style)*",options=list(style_options.keys()),index=0,help="Выберите тип кузова автомобиля")
                Style_ID = style_options[selected_style]
                Distance = st.number_input("Пробег (км)*",min_value=0,max_value=100000000,value=100000)

            with col2:

                Engine_capacity_cm3 = st.number_input("Объем двигателя (л)*",min_value=0.7,max_value=2.7,value=1.5,step=0.1) * 1000  # Конвертируем литры в cm3

                fuel_options = {'Hybrid': 1,'Diesel': 2,'Petrol': 3,'Metan/Propan': 4,'Electric': 5,'Plug-in Hybrid': 6}
                selected_fuel = st.selectbox("Тип топлива (Fuel type)*",options=list(fuel_options.keys()),index=0,help="Выберите тип топлива автомобиля")
                Fuel_type_ID = fuel_options[selected_fuel]

                # Выбор трансмиссии
                transmission_options = {'Manual': 1,'Automatic': 2}
                selected_transmission = st.selectbox("Трансмиссия (Transmission)*",options=list(transmission_options.keys()),index=0,help="Выберите тип трансмиссии автомобиля")
                Transmission_ID = transmission_options[selected_transmission]

            model_choice = st.selectbox("Модель прогнозирования*",options=list(models.keys()),index=0,key="model_manual")

            submitted = st.form_submit_button("Рассчитать стоимость")
            input_data = pd.DataFrame({
                'Year': [Year],
                'Distance': [Distance],
                'Engine_capacity(cm3)': [Engine_capacity_cm3],
                'Style_ID': [Style_ID],
                'Transmission_ID': [Transmission_ID],
                'Fuel_type_ID': [Fuel_type_ID]})

            try:
                with st.spinner("Выполняется расчет..."):
                    prediction = models[model_choice].predict(input_data)[0]

                st.success("### Результат прогнозирования")
                st.markdown(f"""
                <div style="
                    background: #f0f2f6;
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;
                    margin: 20px 0;
                ">
                    <h3 style="color: #333;">Предсказанная стоимость</h3>
                    <h1 style="color: #2e7d32;">{prediction:,.2f} €</h1>
                    <p style="color: #666;">Использована модель: {model_choice}</p>
                </div>
                """, unsafe_allow_html=True)

                st.write("**Введенные параметры:**")
                st.dataframe(input_data)

            except Exception as e:
                st.error(f"Ошибка при прогнозировании: {str(e)}")