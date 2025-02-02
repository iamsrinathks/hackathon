import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
import google.generativeai as genai
from dotenv import load_dotenv
import os
import requests

load_dotenv()


BODHI_LLM_GATEWAY_BASE_URL = os.getenv('BODHI_LLM_GATEWAY_BASE_URL')
BODHI_CURATE_BASE_URL = os.getenv('BODHI_LLM_GATEWAY_BASE_URL')
AUTHORIZATION_TOKEN = os.getenv('AUTHORIZATION_TOKEN')

# Load sample dataset (bank statements)
def load_financial_data(file):
    data = pd.read_csv(file)
    # Ensure mandatory columns are present
    required_columns = ['Date', 'Transaction Description', 'Category', 'Amount', 'Type']
    if not all(column in data.columns for column in required_columns):
        raise ValueError(f"Dataset must contain the following columns: {required_columns}")
    return data


# Data Analysis & Financial Modeling: Categorize expenses and detect spending patterns monthly
def analyze_spending(data):
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure date format
    data['Month'] = data['Date'].dt.to_period('M')  # Extract month
    # No need to mock categorization as 'Category' is already provided
    summary = data.groupby(['Month', 'Category'])['Amount'].sum().reset_index()
    return summary


def generate_spending_quiz(data, num_questions=3):
    # Ensure the data has the required columns
    if not all(col in data.columns for col in ['Date', 'Category', 'Amount']):
        raise ValueError("Dataset must contain 'Date', 'Category', and 'Amount' columns.")
    
    # Format the date and extract month
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.to_period('M')
    
    # Find top spending categories for the latest month in the dataset
    latest_month = data['Month'].max()  # Get the latest month
    top_spending = data[data['Month'] == latest_month]
    top_categories = top_spending.groupby('Category')['Amount'].sum().nlargest(4).reset_index()
    
    # Extract the top 4 categories
    options = top_categories['Category'].tolist()
    
    # Configure the Gemini API
    # genai.configure(api_key="")  # Replace with your actual API key
    # model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Generate quiz questions dynamically
    quiz_questions = []
    for i in range(num_questions):
        # Create a prompt for Gemini to generate a quiz question
        prompt = (
            f"Create a quiz question based on the following spending categories: {', '.join(options)}. "
            "The question should ask where the user spent the most in a specific month and provide four options. "
            "The options should be randomized, and the correct answer should be one of the top categories."
            "The tone should be funny and engaging."
        )

        payload = {
            "model": "gpt-4o",  # Assuming the model name; replace with the correct one if different
            "messages": [{"role": "user", "content": prompt}]
        }

        # Make a POST request to the ASK Bodhi API
        response = requests.post(
            f"{BODHI_LLM_GATEWAY_BASE_URL}/api/openai/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {AUTHORIZATION_TOKEN}"}

        )
        
        # Generate the response using Gemini
        # response = model.generate_content(prompt)
        
        response_data = response.json()
        question_text = response_data['choices'][0]['message']['content']

        # Append the generated question to the list
        quiz_questions.append({
            "question": question_text,
            "correct_answer": np.random.choice(options),  # Randomly choose a correct answer
            "options": options
        })
    
    return quiz_questions

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
#     prompt = (
#         "You are a friendly financial advisor.",
#         f"User query: {user_query}"
#     )

#     payload = {
#         "model": "gpt-4o",  # Assuming the model name; replace with the correct one if different
#         "messages": [{"role": "user", "content": prompt}]
#     }

#     # Make a POST request to the ASK Bodhi API
#     response = requests.post(
#         f"{BODHI_LLM_GATEWAY_BASE_URL}/api/openai/chat/completions",
#         json=payload,
#         headers={"Authorization": f"Bearer {AUTHORIZATION_TOKEN}"}
#     )

#     print(response.text)

#     # Parse the response
#     response_data = response.json()
#     response_text = response_data['choices'][0]['message']['content']
    
#     # Return the generated text
#     return response_text

# Gamified Savings Challenges
# def savings_challenge(goal, monthly_savings):
#     years_to_achieve = goal / monthly_savings
#     years_to_achieve = years_to_achieve / 12
#     return f"Save ${monthly_savings} per month, and you'll reach ${goal} in {round(years_to_achieve, 1)} years!"


# Financial Modeling: Calculate Savings
def calculate_savings(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.to_period('M')
    
    # Identify income from the dataset (assuming entries with Type 'Income')
    income_data = data[data['Type'].str.lower() == 'income']
    
    if income_data.empty:
        return "No income data found in the dataset."
    
    # Get the income for the latest month
    latest_month = data['Month'].max()
    income = income_data[income_data['Month'] == latest_month]['Amount'].sum()
    
    # Calculate total expenses for the latest month (assuming entries with Type 'Expense')
    expense_data = data[(data['Type'].str.lower() == 'expense') & (data['Month'] == latest_month)]
    total_expenses = expense_data['Amount'].sum()
    
    # Calculate savings as the difference
    savings = income - total_expenses
    
    return savings

def main():
    st.title("Personal Finance Dashboard")
    
    # Step 1: Ask for basic user information
    name = st.text_input("What's your name? (Or should we just call you 'Legend'?)")
    dob = st.date_input("What's your birthdate? We promise we won't tell anyone!")
    email = st.text_input("Email address? We won't spam you, promise!")
    
    if st.button("Submit your info (and let the fun begin!)"):
        st.session_state.submitted = True
    
    # Step 2: Ask for financial goal if user details are submitted
    if st.session_state.get("submitted", False):
        goal = st.radio("Alright, what's your main financial goal? Choose wisely!", 
                        ["Budgeting", "Investing", "Saving", "Planning"])
        
        st.write(f"Awesome! Your goal is: {goal}. Now we're getting somewhere!")
        
        if st.button("Submit your financial goal (We can feel the excitement!)"):
            st.session_state.goal_selected = goal
        
        # Step 3: Ask for risk tolerance and life circumstances
        if st.session_state.get("goal_selected", False):
            risk_tolerance = st.selectbox("What's your risk tolerance? No judgment here, we promise!", ["low", "medium", "high"])
            st.write(f"Risk Tolerance: {risk_tolerance} - You do you!")
            
            life_circumstances = st.multiselect(
                "What's going on in your life? (Pick all that apply, we won't judge!)",
                ["Student debt", "First job", "Marriage or planning a family", "Home buying plans"]
            )
            
            st.write(f"Your life circumstances: {', '.join(life_circumstances)}. You're living life to the fullest!")
            
            if st.button("Submit life situation and risk (You're so close now!)"):
                st.session_state.life_selected = life_circumstances
                st.session_state.risk_tolerance = risk_tolerance
                
        # Step 4: Ask for bank statement upload
        if st.session_state.get("life_selected", False):
            uploaded_file = st.file_uploader("Last one, I promise! Upload your bank statement here.", type="csv")
            
            if uploaded_file is not None:
                data = load_financial_data(uploaded_file)
                spending_summary = analyze_spending(data)
                
                st.subheader("Monthly Spending Summary (Brace yourself!)")
                st.dataframe(spending_summary)

                # Calculate savings from the dataset
                savings = calculate_savings(data)

                st.subheader("Monthly Savings Calculation")
                st.write(f"Your total savings for the latest month is: ${savings}")
                
                # Generate quiz questions based on the spending data
                quiz_questions = generate_spending_quiz(data, num_questions=3)
                
                # Initialize the session state to keep track of current question index
                if 'current_question' not in st.session_state:
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                
                # Display the current question and options
                question = quiz_questions[st.session_state.current_question]
                st.subheader(f"Question {st.session_state.current_question + 1}: {question['question']}")
                options = question['options']
                answer = st.radio("Choose your answer:", options)
                
                if st.button("Submit Answer"):
                    if answer == question['correct_answer']:
                        st.session_state.score += 1
                        st.success("Correct! You're a financial genius! 💡")
                    else:
                        st.error(f"Oops! The correct answer was {question['correct_answer']}. Better luck next time!")
                    
                    # Move to next question
                    if st.session_state.current_question < len(quiz_questions) - 1:
                        st.session_state.current_question += 1
                    else:
                        st.write(f"Quiz completed! Your score: {st.session_state.score}/{len(quiz_questions)}")
                        st.session_state.current_question = 0
                        st.session_state.score = 0
                    
                # Generate investment recommendation based on risk tolerance
                investment_recommendation = recommend_investment(spending_summary, st.session_state.risk_tolerance)
                
                st.subheader("Investment Recommendation (Based on your risk level)")
                st.write(investment_recommendation)
                
                # if monthly_income > 0:
                #     monthly_savings = monthly_income - monthly_expense
                #     savings_plan = savings_challenge(financial_goal, monthly_savings)
                    
                #     st.subheader("Savings Challenge (Who said saving can't be fun?)")
                #     st.write(savings_plan)
                # else:
                #     st.error("Please enter a valid monthly income. I promise, this is the last time I ask.")
    
    # st.subheader("Chat with your Financial Advisor")
    # user_query = st.text_input("Got a question about your finances? Ask away!")
    # if user_query:
    #     response = chatbot_response(user_query)
    #     st.write(response)

# Run the Streamlit app
if __name__ == "__main__":
    main()
