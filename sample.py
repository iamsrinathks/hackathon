import pandas as pd
import numpy as np
import streamlit as st
import openai
from sklearn.cluster import KMeans
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load sample dataset (bank statements)
def load_financial_data(filepath):
    return pd.read_csv(filepath)

# Data Analysis & Financial Modeling: Categorize expenses and detect spending patterns
def analyze_spending(data):
    categories = ['Rent', 'Groceries', 'Entertainment', 'Savings', 'Investments', 'Utilities']
    data['Category'] = np.random.choice(categories, size=len(data))  # Mock categorization
    summary = data.groupby('Category')['Amount'].sum().reset_index()
    return summary

# Smart Investing Recommendations using Clustering (K-Means)
def recommend_investment(data, risk_tolerance):
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['Amount']])
    if risk_tolerance == 'low':
        return "Consider bonds and ETFs with steady returns."
    elif risk_tolerance == 'medium':
        return "Balanced mix of stocks and index funds."
    else:
        return "High-growth stocks and crypto for aggressive investors."

# Conversational AI for financial advice
# def chatbot_response(user_query):
#     openai.api_key = "your-openai-api-key"
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": user_query}]
#     )
#     return response['choices'][0]['message']['content']


# Conversational AI for financial advice
# def chatbot_response(user_query, monthly_income, monthly_expense, financial_goal, years_to_achieve):
#     openai.api_key = "your-openai-api-key"  # Replace with your actual OpenAI API key
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "You are a financial advisor."},
#             {"role": "user", "content": f"My monthly income is ${monthly_income}, my monthly expense is ${monthly_expense}, and my financial goal is ${financial_goal} in {years_to_achieve} years."},
#             {"role": "user", "content": user_query}
#         ]
#     )
#     return response['choices'][0]['message']['content']


# Conversational AI for financial advice
def chatbot_response(user_query, monthly_income, monthly_expense, financial_goal, years_to_achieve):
    # Replace with your actual Gemini API key
    genai.configure(api_key="")
    
    # Initialize the Gemini model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Create the prompt for Gemini
    prompt = (
        f"You are a financial advisor. My monthly income is ${monthly_income}, "
        f"my monthly expense is ${monthly_expense}, and my financial goal is "
        f"${financial_goal} in {years_to_achieve} months. {user_query}"
    )
    
    # Generate the response using Gemini
    response = model.generate_content(prompt)
    
    # Return the generated text
    return response.text

# Gamified Savings Challenges
def savings_challenge(goal, monthly_savings):
    years_to_achieve = goal / monthly_savings
    years_to_achieve = years_to_achieve / 12
    return f"Save ${monthly_savings} per month, and you'll reach ${goal} in {round(years_to_achieve, 1)} years!"



def main():
    st.title("Personal Finance Dashboard")

    # User input fields
    monthly_income = st.number_input("Enter your monthly income ($):", min_value=0.0, step=100.0)
    age = st.number_input("Enter your age:", min_value=0, step=1)
    monthly_expense = st.number_input("Enter your monthly expenses ($):", min_value=0.0, step=100.0)
    financial_goal = st.number_input("Enter your financial goal ($):", min_value=0.0, step=100.0)
    years_to_achieve = st.number_input("Enter the number of years to achieve your goal:", min_value=0, step=1)
    # File uploader for financial data
    uploaded_file = st.file_uploader("Upload your financial data CSV file", type="csv")

    if uploaded_file is not None:
        data = load_financial_data(uploaded_file)
        spending_summary = analyze_spending(data)
        
        st.subheader("Spending Summary")
        st.dataframe(spending_summary)
        
        risk_tolerance = st.selectbox("Select your risk tolerance level:", ["low", "medium", "high"])
        investment_recommendation = recommend_investment(spending_summary, risk_tolerance)
        
        st.subheader("Investment Recommendation")
        st.write(investment_recommendation)
        
        if monthly_income > 0 and monthly_expense > 0:
            monthly_savings = monthly_income - monthly_expense
            savings_plan = savings_challenge(financial_goal, monthly_savings)
            
            st.subheader("Savings Challenge")
            st.write(savings_plan)
        else:
            st.error("Please enter valid monthly income and expenses.")

    st.subheader("Chat with Financial Advisor")
    user_query = st.text_input("Ask a question about your financial goals:")
    if user_query:
        response = chatbot_response(user_query, monthly_income, monthly_expense, financial_goal, years_to_achieve)
        st.write(response)

# Run the Streamlit app
if __name__ == "__main__":
    main()

# # Example usage
# data = load_financial_data("/Users/Desktop/Personal_Finance_Dataset.csv")
# spending_summary = analyze_spending(data)
# print(spending_summary)
# print(recommend_investment(spending_summary, "medium"))
# # print(chatbot_response("How can I save more money?"))
# print(savings_challenge(5000, 200))
