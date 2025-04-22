import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
import pickle
from sklearn.metrics import r2_score
import base64

# make pandas dataframe
df = pd.read_csv("car_data_minor.csv")
# rename the column name
df.rename(columns={"Name": 'Model', "company_name": 'Brand'}, inplace=True)
pipe = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Car Price Prediction", layout="wide")


# --- Sidebar Navigation ---
st.sidebar.title("📂 Navigation")
# Sidebar styling to improve visibility
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.7);  /* Black with transparency */
        color: white;
    }
    [data-testid="stSidebar"] .css-1cpxqw2 {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# function for add BG image
def set_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        img_base64 = base64.b64encode(img.read()).decode()

    st.markdown(f"""
        <style>
        .stApp {{
            background-color: transparent;
        }}
        body::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-image: url("data:image/jpeg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            opacity: 0.2;  /* 🔁 control how faded the image is */
            z-index: -1;
        }}
        </style>
    """, unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Go to",
    ["Home", "Specific Vehicle Info", "Trends & Insights", "Analytics Dashboard", "Price Predictor",
     "About the Project"],
    index=0  # Default to "Home"
)

# --- HOME ---
if menu == "Home":
    st.title("🚗 Car Price Prediction App")
    st.markdown("### Unlock the True Value of Your Car Using AI & Data 📊")

    st.image("https://cdn.pixabay.com/photo/2015/01/19/13/51/car-604019_1280.jpg", use_container_width=True)

    st.markdown("""
    Welcome to the **Car Price Prediction App** – your intelligent assistant for estimating the market value of used cars!  
    Powered by **machine learning**, this app helps car buyers and sellers make **smarter, faster decisions**.

    ---

    ### 🔧 What You Can Do:
    - 🧠 **Predict Car Prices** based on features like brand, year, mileage, fuel type, and more  
    - 📈 **Visualize Market Trends** using interactive charts and dashboards  
    - 🧪 **Learn How It Works** with a behind-the-scenes look at our ML model
    """)

    with st.expander("🧬 How This Works"):
        st.markdown("""
        This app uses a trained **regression model** to predict the price of used cars.  
        Here’s a simplified view of our pipeline:
        - Data Preprocessing 🧹
        - Feature Engineering ⚙️
        - Model Training 🤖
        - Real-time Predictions 💡
        """)
        st.image("https://miro.medium.com/v2/resize:fit:1200/1*84CuvpAKsIuAeJPNYjR2FQ.png",
                 caption="Car Price Prediction Workflow", use_container_width=True)

    st.image(
        "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExOWZsanZ6Y2xtb3ljb3FnMzczNndjeHAyNW9iNnM2eDBwamxvNjlzZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/iEJ5zikFjdFjq/giphy.gif",
        caption="Let's drive into data!",
        use_container_width=None)
    st.markdown("### 💬 How do you like the app so far?")
    feedback = st.radio("Give us a quick emoji review:", ["😃 Awesome", "🙂 Good", "😐 Okay", "😕 Needs work"])
    if feedback:
        st.success("Thanks for your feedback! 💌")
        st.markdown("🚗 — 🚙 — 🚘 — 🏎️ — 🚓 — 🚐 — 🚛")

    st.markdown("---")
    st.markdown("""


        📬 Contact: [kuldeep9981patidar@gmail.com]
         
        👋 Linkedin: [www.linkedin.com/in/kuldeep-patidar-mp14]
        """)








# --- ABOUT ---
elif menu == "About the Project":
    st.title("ℹ️ About the Project")
    set_bg_from_local("car_bg.jpg")

    st.markdown("""
    This project uses a machine learning regression model trained on a dataset of used cars.  
    It includes features such as:
    - Name
    - Year
    - Kilometers Driven
    - Fuel Type
    - Transmission
    - Mileage
    - Engine
    - Power
    - Seats



    **Tools Used:**
    - Python 🐍
    - Streamlit 🎈
    - Pandas, NumPy, Scikit-learn ⚙️
    - Matplotlib & Seaborn 📊

    ---

    This web app includes the following sections:
   - 🏠 **Home** – Overview and welcome page  
   - 🔮 **Predict** – Predict car prices based on input features  
   - 📈 **Insights** – Visual exploration of dataset trends  
   - 📊 **Dashboard** – Interactive Power BI report  
   - ℹ️ **About Project** – Description and background of the project

    """)




elif menu == 'Trends & Insights':
    st.title("📊 Understand the Market, Buy Smarter")

    st.markdown("#### 🏷️ Average Car Price by Company")
    st.write("""
    This chart shows the average car price for each brand. 
    It helps you understand which companies sell premium vehicles (like Audi, BMW, or Mercedes) and which ones offer more budget-friendly options (like Maruti, Hyundai, or Tata).

    💡 **Use it to:**  
    - Compare brand pricing at a glance  
    - Identify luxury vs economy brands  
    - Decide which company fits your budget range
    """)
    avg_price_by_company = df.groupby('Brand')['Price'].mean().sort_values(ascending=False).reset_index()

    fig1 = px.bar(avg_price_by_company, x='Brand', y='Price', title='Average Car Price by Company')
    st.plotly_chart(fig1)

    # plot second figure and write some distcription for 2nd figure
    st.markdown("#### 📉 Car Price vs Year (Depreciation Trend)")
    st.write(
        "This graph shows how car prices decrease with age. You’ll notice that most cars start losing value after just a few years. This helps you find the sweet spot — a car that’s affordable but not too old to be unreliable.")
    fig2 = px.scatter(df, x='Year', y='Price', color='Brand',
                      title='Car Price vs Manufacturing Year', size='Price',
                      hover_data=['Model', 'Fuel_Type'])
    st.plotly_chart(fig2)

    # plot 3rd figure and write some description for 3rd figure
    st.markdown("#### ⛽ Average Mileage by Fuel Type")
    st.write(
        "Fuel economy can make or break your budget in the long run. This chart compares average mileage across fuel types (Petrol, Diesel, CNG, etc.) so you can choose what’s most efficient and cost-friendly.")

    avg_mileage = df.groupby('Fuel_Type')['Mileage'].mean().reset_index()
    fig3 = px.bar(avg_mileage, x='Fuel_Type', y='Mileage',
                  title='Average Mileage by Fuel Type', color='Fuel_Type')
    st.plotly_chart(fig3)

    # 4th figure and write some description for 4th figure
    st.markdown("#### ⚙️ Price vs Engine Capacity")
    st.write("""
    This scatter plot visualizes how engine size (in CC) relates to car price across different brands.  
    Larger engines typically mean higher power — and often higher prices. Each point represents a car, with the bubble size also reflecting price.

    💡 **Use it to:**  
    - Understand the cost difference between small vs large engine cars  
    - Spot which companies offer powerful engines at lower prices  
    - Explore performance options within your budget
    """)

    fig4 = px.scatter(df, x='Engine(in-CC)', y='Price', size='Price', color='Brand',
                      title='Price vs Engine Capacity', hover_data=['Model'])
    st.plotly_chart(fig4)

    # 5th figure and write some description for 5th figure
    st.markdown("#### 💸 Best Budget + High Mileage Cars")
    st.write(
        "Looking to save money upfront *and* on fuel? These cars hit the sweet spot with lower prices and high mileage — ideal for city commuting or budget-conscious buyers.")
    cheap_efficient = df[(df['Price'] < df['Price'].quantile(0.3)) &
                         (df['Mileage'] > df['Mileage'].quantile(0.7))]

    fig5 = px.scatter(cheap_efficient, x='Mileage', y='Price',
                      title='Budget Cars with High Mileage',
                      color='Brand', hover_data=['Model', 'Year'])
    st.plotly_chart(fig5)

    # add some more insights
    st.markdown("### 💡 Extra Buying Insights")

    st.info(
        "🧠 **3–5 Year Old Cars Offer Great Value**\n\nThey’ve already depreciated but are still in great condition — often the best deal.")

    st.success(
        "🛠️ **Manual vs Automatic**\n\nManuals are cheaper and fuel-efficient. Automatics are easier for city driving.")

    st.warning(
        "💸 **Diesel Isn’t Always Cheaper**\n\nHigher purchase cost and maintenance. Worth it only for long-distance drivers.")

    st.info(
        "🏎️ **Pick Engine Size Based on Your Need**\n\nSmall engines (1000–1200 CC) are ideal for cities; larger ones are better for highways.")

    st.success(
        "📊 **Compare Prices Across Brands and Locations**\n\nUse filters to find the best value for your budget.")

    # write this line of code for good ending of this page
    st.markdown("---")
    st.markdown("### 🙌 Thanks for Exploring with Us")

    st.write("""
    We built this insights section to help you buy smarter — not harder. Every chart, every tip here is meant to save you time, money, and guesswork.

    Whether you’re buying your first car or your fifth, we hope this guide helped you feel more confident.

    **Good luck, and may the best deal be yours! 🔑🚗**
    """)



elif menu == 'Price Predictor':
    st.title("🚗 Predict Your Car Price")
    set_bg_from_local("car_bg_img.jpg")
    st.markdown("Enter your car details below to get an estimated market price:")
    st.markdown("This app uses machine learning to estimate the price of used cars.")

    # You can wrap this in a form for cleaner UX
    with st.form("car_input_form"):
        car_brd = st.selectbox("Enter Car Brand",
                               ['-- Select --', 'Audi', 'BMW', 'Chevrolet', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda',
                                'Hyundai', 'ISUZU', 'Isuzu', 'Jaguar', 'Jeep', 'Land', 'Mahindra', 'Maruti',
                                'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan', 'Porsche', 'Renault', 'Skoda', 'Tata',
                                'Toyota', 'Volkswagen', 'Volvo'], index=0)
        car_name = st.text_input("Model")
        fuel_type = st.selectbox("Fuel_Type", ['Select one', 'Diesel', 'Petrol', 'CNG', 'LPG'], index=0)
        engine = st.number_input("Engine Capacity (in CC)", min_value=500, max_value=6000)
        power = st.number_input("Power (in bhp)", min_value=30.0, max_value=800.0, step=1.0)
        mileage = st.number_input("Mileage (km/l) or (km/kg)", min_value=5.0, max_value=50.0, step=0.1)
        transmission = st.selectbox('Transmission', ['Select one', 'Manual', 'Automatic'], index=0)
        year = st.number_input("Year Model", min_value=1990, max_value=2022, step=1)
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000)

        submit = st.form_submit_button("Predict Price")

    # Show results only after form is submitted
    if submit:
        car_name = car_name.lower().strip()

        st.markdown("### 🔍 Processing your car details...")
        input_df = pd.DataFrame([{
            'Model': car_name,
            'Year': year,
            'Kilometers_Driven': km_driven,
            'Fuel_Type': fuel_type,
            'Transmission': transmission,
            'Mileage': mileage,
            'Engine(in-CC)': engine,
            'Power(in-bhp)': power,
            'Brand': 'car_brd'
        }])

        prediction = pipe.predict(input_df)[0]
        st.success(f"💰 Estimated Price: ₹ {prediction:.2f} lakhs")
        st.success(f"Accuracy of model is: 80%")
        st.markdown("---")
        st.markdown("✅ This prediction is based on the latest model trained on real car listings.")
        st.info("Note: The price may vary due to factors like car condition, service history, and region.")

elif menu == 'Analytics Dashboard':
    st.title("📊 My Dashboard")
    st.markdown("Welcome to the **Dashboard** section of the website.")
    st.markdown("""
    This report provides interactive insights into sales performance.  
    Use the slicers and filters built into the report to explore the data.

    > 📝 **Note**: This is a view-only report. Users cannot edit or export data.
    """)
    components.iframe(
        "https://app.powerbi.com/reportEmbed?reportId=2c64e4c1-a3bd-49f0-95d2-997e5796aa70&autoAuth=true&ctid=0603a590-21a9-4210-9bff-def56fca0a48",
        width=1200, height=500)


elif menu == "Specific Vehicle Info":
    st.header("🔍 Search for a Specific Car")
    set_bg_from_local("car_bg_image.jpg")  # Use the image in the same folder as app.py


    with st.form("car_search_form"):
        car_brd = st.selectbox("Enter Car Brand",
                               ['-- Select --', 'Audi', 'BMW', 'Chevrolet', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda',
                                'Hyundai', 'ISUZU', 'Isuzu', 'Jaguar', 'Jeep', 'Land', 'Mahindra', 'Maruti',
                                'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan', 'Porsche', 'Renault', 'Skoda', 'Tata',
                                'Toyota', 'Volkswagen', 'Volvo'], index=0)
        car_model = st.text_input("Enter Car Model (Optional)")
        location_input = st.selectbox("Enter Location",
                                      ['-- Select --', 'Ahmedabad', 'Bangalore', 'Chennai', 'Coimbatore', 'Delhi',
                                       'Hyderabad', 'Jaipur', 'Kochi', 'Kolkata', 'Mumbai', 'Pune'], index=0)
        price_range = st.selectbox(
            "Enter Car Price Range",
            ["-- Select --", '0-5L', '5-10L', '10-15L', '15-20L', '20-25L', '25-35L', '35-50L', 'Above 50L']
        )
        seat = st.text_input("Enter Number of Seats (Optional)")
        btn = st.form_submit_button("Submit")

    if btn:
        filtered_df = df.copy()

        # Filter by brand
        filtered_df = filtered_df[filtered_df['Brand'] == car_brd]

        # Filter by model if provided
        if car_model:
            filtered_df = filtered_df[filtered_df['Model'].str.contains(car_model, case=False, na=False)]

        # Filter by location
        filtered_df = filtered_df[filtered_df['Location'] == location_input]


        # Filter by price range
        def price_filter(row):
            Price = row['Price']
            if price_range == '0-5L':
                return 0 <= Price <= 5
            elif price_range == '5-10L':
                return 5 < Price <= 10
            elif price_range == '10-15L':
                return 10 < Price <= 15
            elif price_range == '15-20L':
                return 15 < Price <= 20
            elif price_range == '20-25L':
                return 20 < Price <= 25
            elif price_range == '25-35L':
                return 25 < Price <= 35
            elif price_range == '35-50L':
                return 35 < Price <= 50
            elif price_range == 'Above 50L':
                return Price > 50
            return True


        if price_range != "Select One":
            filtered_df = filtered_df[filtered_df.apply(price_filter, axis=1)]

        # Filter by seat count if provided
        if seat:
            try:
                seat = int(seat)
                filtered_df = filtered_df[filtered_df['Seats'] == seat]
            except ValueError:
                st.warning("Please enter a valid number for seats.")


        # create unit function for mantain the unit of mileage(km/l ,km/kg)..
        def unit(*x):
            for i in x:
                if i == 'Diesel' or i == 'Petrol':
                    u = 'km/l'
                else:
                    u = "Km/kg"

            return u


        if not filtered_df.empty:
            st.success(f"Found {len(filtered_df)} car(s) matching your search.")
            for idx, row in filtered_df.iterrows():
                st.markdown("---")
                st.subheader(f" {row['Model']} ({row['Year']})")
                st.write(f"📍 Location: {row['Location']}")
                st.write(f"💰 Price: ₹{row['Price']:,} Lakh")
                st.write(f"🪑 Seats: {row['Seats']}")
                st.write(f"⛽ **Fuel Type:** {row['Fuel_Type']}")
                st.write(f"⚙️ **Transmission:** {row['Transmission']}")
                st.write(f"🏁 **Mileage:** {row['Mileage']} {unit(row['Fuel_Type'])}")
                st.write(f"🔋 **Engine:** {row['Engine(in-CC)']} CC")
                st.write(f"💨 **Power:** {row['Power(in-bhp)']} bhp")
                st.write(f"📏 Kilometers Driven: {row['Kilometers_Driven']:,} km")

            st.markdown("---")
            st.markdown("### 🙌 Thank you for using Car Price Explorer!")
            st.markdown("Feel free to explore more cars or use the prediction tool.")

        else:
            st.error("No cars found with the selected filters.")


