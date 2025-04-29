def main():
    st.title("🔥 Burnout Early Warning System")
    user_input, submitted = create_questionnaire()

    if submitted:
        input_df = pd.DataFrame([user_input])
        model, metrics = backend.load_model("burnout_model.pkl")
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][int(prediction)]

        display_burnout_prediction(prediction, proba, input_df)

    st.sidebar.header("Model Info")
    model_path = "burnout_model.pkl"
    _, metrics = backend.load_model(model_path)
    display_model_accuracy(metrics)


if __name__ == "__main__":
    main()
