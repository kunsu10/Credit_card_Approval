import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression




@st.cache_data
def load_and_train_model():
    """Generates synthetic data and trains a simple Linear Regression model."""
    
   
    np.random.seed(42)
    
   
    sqft = np.random.randint(1000, 5000, 100)
    
    
    bedrooms = np.random.randint(2, 6, 100)
    
    
    noise = np.random.normal(0, 50000, 100)
    prices = 100 * sqft + 50000 * bedrooms + 100000 + noise
    
    
    data = pd.DataFrame({'SquareFootage': sqft, 'Bedrooms': bedrooms, 'Price': prices})
    
    
    model = LinearRegression()
    X = data[['SquareFootage', 'Bedrooms']]
    y = data['Price']
    model.fit(X, y)
    
    return model, data


model, data = load_and_train_model()



def main():
    st.set_page_config(page_title="House Price Predictor", layout="wide")
    
    st.title(" House Price Prediction App")
    st.markdown("Use the controls below to predict the price of a house using a simple ML model.")

    
    with st.sidebar:
        st.header("House Details")
        
       
        sqft_input = st.slider(
            "Square Footage (sqft)",
            min_value=1000,
            max_value=5000,
            value=2500,
            step=100
        )
        
       
        bedrooms_input = st.selectbox(
            "Number of Bedrooms",
            options=[2, 3, 4, 5],
            index=1 
        )
        
       
        predict_button = st.button("Predict Price")

  
    st.subheader("Prediction Result")
    
    if predict_button:
        
        new_data = pd.DataFrame({
            'SquareFootage': [sqft_input],
            'Bedrooms': [bedrooms_input]
        })
        
        predicted_price = model.predict(new_data)[0]
        
        
        st.success(f"**The Estimated House Price is: \${predicted_price:,.2f}**")
        st.balloons() 

        
        st.markdown("---")
        st.subheader("Feature vs. Price Visualization")

       
        plot_data = data.copy()
        
        
        plot_data['Type'] = 'Training Data'
        prediction_point = pd.DataFrame({
            'SquareFootage': [sqft_input], 
            'Price': [predicted_price], 
            'Type': ['Your Prediction']
        })
        plot_data = pd.concat([plot_data, prediction_point], ignore_index=True)

        # Plotly Scatter Plot
        fig = px.scatter(
            plot_data, 
            x='SquareFootage', 
            y='Price', 
            color='Type', 
            size='Bedrooms', 
            color_discrete_map={'Training Data': 'blue', 'Your Prediction': 'red'},
            title="Price vs. Square Footage (Size indicates # of Bedrooms)",
            height=500
        )
        
      
        fig.update_traces(
            marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Enter the house features in the sidebar and click 'Predict Price' to see the result.")
        st.subheader("About the Model")
        st.write("This app uses a simple **Linear Regression Model** trained on synthetic data. The training data scatter plot is shown below.")
        
     
        fig_initial = px.scatter(
            data, 
            x='SquareFootage', 
            y='Price', 
            color='Bedrooms',
            title="Training Data Distribution",
            height=500
        )
        st.plotly_chart(fig_initial, use_container_width=True)

if __name__ == "__main__":
    main()