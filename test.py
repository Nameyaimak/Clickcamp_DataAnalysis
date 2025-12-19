import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# หน้าเว็บ
st.set_page_config(page_title="AI Waste Prediction", page_icon="🤖", layout="wide")

st.title(" AI ทำนายปริมาณขยะล่วงหน้า")

st.divider()

# อัปโหลด CSV
uploaded = st.file_uploader("📂 อัปโหลดไฟล์ CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("โหลดไฟล์สำเร็จ ✔")

    st.subheader("📄 ตัวอย่างข้อมูล")
    st.dataframe(df.head(), use_container_width=True)

    # แสดงชื่อคอลัมน์
    st.divider()
    st.subheader("🧩 เลือกฟีเจอร์เพื่อเทรนโมเดล")

    all_columns = df.columns.tolist()

    # เลือก features และ target
    features = st.multiselect(
        "เลือก Features",
        all_columns,
        default=[
            "population", "recyclable_kg", "organic_kg",
            "collection_capacity_kg", "temp_c", "rain_mm",
            "is_weekend", "is_holiday", "recycling_campaign"
        ] if set([
            "population", "recyclable_kg", "organic_kg",
            "collection_capacity_kg", "temp_c", "rain_mm",
            "is_weekend", "is_holiday", "recycling_campaign"
        ]).issubset(all_columns) else []
    )

    target = st.selectbox(
        "เลือกค่า Target (ค่าที่ต้องการทำนาย)",
        all_columns,
        index=all_columns.index("waste_kg") if "waste_kg" in all_columns else 0
    )

    if len(features) > 0 and target:
        X = df[features]
        y = df[target]

        # Train-test split
        test_size = st.slider("ขนาด Test Set (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )

        # Train button
        if st.button("🚀 Train Model"):
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            st.success("🎉 เทรนโมเดลสำเร็จ!")

            col1, col2 = st.columns(2)
            col1.metric("📉 MSE", f"{mse:,.2f}")
            col2.metric("📈 R²", f"{r2:.4f}")

            st.divider()

            # =============================
            #    กราฟ Predicted vs Actual
            # =============================
            st.subheader("📌 Predicted vs Actual Plot")

            feature_list_str = ", ".join(features)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, preds, alpha=0.6)

            min_val = min(y_test.min(), preds.min())
            max_val = max(y_test.max(), preds.max())

            ax.plot([min_val, max_val], [min_val, max_val], '--', color='red', lw=2, label='Perfect Prediction Line')

            ax.set_xlabel(f'Actual {target}')
            ax.set_ylabel(f'Predicted {target}')
            ax.set_title(f'Predicted vs Actual: {target}\nUsing Features: {feature_list_str}')
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

            st.divider()

            # =============================
            #   Download Predictions + Show Table
            # =============================
            st.subheader("📥 ผลการทำนายทั้งหมด")

            pred_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": preds,
            })

            pred_df["Error (Pred - Actual)"] = pred_df["Predicted"] - pred_df["Actual"]

            st.dataframe(pred_df, use_container_width=True)

            csv_out = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 ดาวน์โหลด CSV ผลทำนาย + ค่าคลาดเคลื่อน",
                csv_out,
                "predictions_with_error.csv",
                "text/csv"
            )

else:
    st.info("⬆️ อัปโหลดไฟล์เพื่อเริ่มต้นใช้งาน")
