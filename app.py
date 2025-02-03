import pandas as pd
import numpy as np
import streamlit as st
import os, requests
import google.generativeai as genai
import json
import matplotlib.pyplot as plt
import io


from sklearn.cluster import KMeans
from dotenv import load_dotenv
from datetime import date



load_dotenv()


BODHI_LLM_GATEWAY_BASE_URL = os.getenv('BODHI_LLM_GATEWAY_BASE_URL')
BODHI_CURATE_BASE_URL = os.getenv('BODHI_LLM_GATEWAY_BASE_URL')
AUTHORIZATION_TOKEN = os.getenv('AUTHORIZATION_TOKEN')
BODHI_OPTIMIZE_BASE_URL= os.getenv('BODHI_OPTIMIZE_BASE_URL')

# Load sample dataset (bank statements)
def load_financial_data(file):
    data = pd.read_csv(file)
    # Ensure mandatory columns are present
    required_columns = ['Date', 'Transaction Description', 'Category', 'Amount', 'Type']
    if not all(column in data.columns for column in required_columns):
        raise ValueError(f"Dataset must contain the following columns: {required_columns}")
    return data


def analyze_spending(data):
    # Print the first few rows of the dataframe
    print("First few rows of the dataframe:")
    print(data.head())
    print("\nData type of 'Date' column:", data['Date'].dtype)
    print("\nUnique values in 'Date' column:")
    print(data['Date'].unique()[:10])  # Print first 10 unique values
    
    # Try parsing dates without specifying format
    try:
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
        print("\nDate parsing successful!")
    except ValueError as e:
        print(f"\nError parsing dates: {str(e)}")
        return None  # Return None if parsing fails

    data['Month'] = data['Date'].dt.to_period('M')  # Extract month
    summary = data.groupby(['Month', 'Category'])['Amount'].sum().reset_index()
    return summary


def generate_spending_quiz(data, num_questions=3):
    # Ensure the data has the required columns
    if not all(col in data.columns for col in ['Date', 'Category', 'Amount']):
        raise ValueError("Dataset must contain 'Date', 'Category', and 'Amount' columns.")
    
    # Format the date and extract month
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Month'] = data['Date'].dt.to_period('M')
    
    # Find top spending categories for the latest month in the dataset
    latest_month = data['Month'].max()  # Get the latest month
    top_spending = data[data['Month'] == latest_month]
    top_categories = top_spending.groupby('Category')['Amount'].sum().nlargest(4).reset_index()
    
    # Extract the top 4 categories
    options = top_categories['Category'].tolist()
    
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
    if data is None:
        return "Unable to generate investment recommendations due to invalid data."
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['Amount']])
    if risk_tolerance == 'low':
        return "Consider bonds and ETFs with steady returns."
    elif risk_tolerance == 'medium':
        return "Balanced mix of stocks and index funds."
    else:
        return "High-growth stocks and crypto for aggressive investors."

# Conversational AI for financial advice
def chatbot_response(user_query):
    prompt = (
        "You are a friendly financial advisor.",
        f"User query: {user_query}"
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

    # Parse the response
    response_data = response.json()
    response_text = response_data['choices'][0]['message']['content']
    
    # Return the generated text
    return response_text

def calculate_savings(data):
    # Parse dates using the correct format
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Month'] = data['Date'].dt.to_period('M')

    # Rest of the function remains the same
    income_data = data[data['Type'].str.lower() == 'income']
    
    if income_data.empty:
        return "No income data found in the dataset."
    
    latest_month = data['Month'].max()
    print ("latest_month", latest_month)
    income = income_data[income_data['Month'] == latest_month]['Amount'].sum()
    print ("income", income)
    
    expense_data = data[(data['Type'].str.lower() == 'expense') & (data['Month'] == latest_month)]
    print("expense_data",expense_data)
    total_expenses = expense_data['Amount'].sum()
    print("total_expenses of a month", total_expenses)
    
    savings = income + total_expenses

    return savings

def calculate_age(born):
    today = date.today()
    print ("Age:",today.year - born.year - ((today.month, today.day) < (born.month, born.day)))
    return (today.year - born.year - ((today.month, today.day) < (born.month, born.day)))


def analyze_budget(data):
    """
    Analyze spending habits and provide budgeting recommendations using AI.
    """
    if data is None:
        return "Unable to analyze budget due to invalid data."
    
    # Calculate total income and expenses
    total_income = data[data['Type'].str.lower() == 'income']['Amount'].sum()
    total_expenses = data[data['Type'].str.lower() == 'expense']['Amount'].sum()
    
    # Identify top spending categories
    top_categories = data[data['Type'].str.lower() == 'expense'].groupby('Category')['Amount'].sum().nlargest(3).reset_index()
    
    # Create a prompt for AI
    prompt = (
        "You are a friendly financial advisor. Analyze the following financial data and provide budgeting recommendations:\n"
        f"- Total Income: £{total_income:.2f}\n"
        f"- Total Expenses: £{total_expenses:.2f}\n"
        f"- Top Spending Categories: {', '.join(top_categories['Category'].tolist())}\n"
        "Provide actionable advice to improve budgeting and savings."
    )
    url = "https://bodhi-optimise-bot-hackathon.api.sandbox.psbodhi.live/chat"
    payload = { "message": prompt}
    headers = {
        "authorization": f"Bearer {AUTHORIZATION_TOKEN}",
        "content-type": "application/json"
    }
    response = requests.post( url,json=payload,headers= headers)
    # Parse the response and filter for the "summariser" role
    response_data = response.json()
    pretty_data = json.dumps(response_data, indent=4)
    print(pretty_data)
    summ_res = []
    for msg in response_data["responses"]:
        if msg["name"] == "summariser":
            summ_res.append(msg["content"])
    if summ_res:
        print("Summariser response:")
        print(summ_res[-1])
        return (summ_res[-1])
    else:
        print("No summariser response found.")


def generate_investment_plan(data, risk_tolerance):
    """
    Generate a personalized investment plan using AI.
    """
    if data is None:
        return "Unable to generate investment plan due to invalid data."
    
    # Calculate total savings
    total_savings = calculate_savings(data)
    
    # Create a prompt for AI
    prompt = (
        "You are a friendly financial advisor. Generate a personalized investment plan based on the following details:\n"
        f"- Risk Tolerance: {risk_tolerance}\n"
        f"- Total Savings: £{total_savings:.2f}\n"
        "Provide a detailed investment plan with asset allocation and advice."
    )

   
    url = "https://bodhi-optimise-bot-hackathon.api.sandbox.psbodhi.live/chat"
    payload = { "message": prompt}
    headers = {
        "authorization": f"Bearer {AUTHORIZATION_TOKEN}",
        "content-type": "application/json"
    }
    response = requests.post( url,json=payload,headers= headers)
    # Parse the response and filter for the "summariser" role
    response_data = response.json()
    pretty_data = json.dumps(response_data, indent=4)
    print(pretty_data)
    summ_res = []
    for msg in response_data["responses"]:
        if msg["name"] == "summariser":
            summ_res.append(msg["content"])
    if summ_res:
        print("Summariser response:")
        print(summ_res[-1])
        return (summ_res[-1])
    else:
        print("No summariser response found.")

def create_savings_plan(data, goal_amount, timeframe_months):
    """
    Create a savings plan using AI.
    """
    if data is None:
        return "Unable to create savings plan due to invalid data."
    
    # Calculate current savings rate
    total_income = data[data['Type'].str.lower() == 'income']['Amount'].sum()
    total_expenses = data[data['Type'].str.lower() == 'expense']['Amount'].sum()
    current_savings = total_income - total_expenses

    prompt = (
        "You are a friendly financial advisor. Create a savings plan based on the following details:\n"
        f"- Goal Amount: £{goal_amount:.2f}\n"
        f"- Timeframe: {timeframe_months} months\n"
        f"- Current Savings: £{current_savings:.2f}\n"
        "Provide actionable advice to achieve the savings goal."
    )
    url = "https://bodhi-optimise-bot-hackathon.api.sandbox.psbodhi.live/chat"
    payload = { "message": prompt}
    headers = {
        "authorization": f"Bearer {AUTHORIZATION_TOKEN}",
        "content-type": "application/json"
    }
    response = requests.post( url,json=payload,headers= headers)
    # Parse the response and filter for the "summariser" role
    response_data = response.json()
    pretty_data = json.dumps(response_data, indent=4)
    print(pretty_data)
    summ_res = []
    for msg in response_data["responses"]:
        if msg["name"] == "summariser":
            summ_res.append(msg["content"])
    if summ_res:
        print("Summariser response:")
        print(summ_res[-1])
        return (summ_res[-1])
    else:
        print("No summariser response found.")

def generate_financial_plan(data, risk_tolerance, goal_amount, timeframe_months):
    """
    Generate a comprehensive financial plan using AI.
    """
    if data is None:
        return "Unable to generate financial plan due to invalid data."
    
    # Calculate total savings
    total_savings = calculate_savings(data)
    
    # Create a prompt for AI
    prompt = (
        "You are a friendly financial advisor. Generate a comprehensive financial plan based on the following details:\n"
        f"- Risk Tolerance: {risk_tolerance}\n"
        f"- Goal Amount: £{goal_amount:.2f}\n"
        f"- Timeframe: {timeframe_months} months\n"
        f"- Total Savings: £{total_savings:.2f}\n"
        "Provide a detailed plan covering budgeting, saving, and investing strategies."
    )
    url = "https://bodhi-optimise-bot-hackathon.api.sandbox.psbodhi.live/chat"
    payload = { "message": prompt}
    headers = {
        "authorization": f"Bearer {AUTHORIZATION_TOKEN}",
        "content-type": "application/json"
    }
    response = requests.post( url,json=payload,headers= headers)
    # Parse the response and filter for the "summariser" role
    response_data = response.json()
    pretty_data = json.dumps(response_data, indent=4)
    print(pretty_data)
    summ_res = []
    for msg in response_data["responses"]:
        if msg["name"] == "summariser":
            summ_res.append(msg["content"])
    if summ_res:
        print("Summariser response:")
        print(summ_res[-1])
        return (summ_res[-1])
    else:
        print("No summariser response found.")


def plot_monthly_expenses(data):
    # Ensure the data is properly formatted
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Month'] = data['Date'].dt.to_period('M')

    # Filter for expenses
    expense_data = data[data['Type'].str.lower() == 'expense']
    
    # Group by month and category to get total expenses
    monthly_expenses = expense_data.groupby(['Month', 'Category'])['Amount'].sum().unstack(fill_value=0)
    
    # Plotting the data
    monthly_expenses.plot(kind='bar', stacked=True, figsize=(10, 6), cmap='Set3')
    plt.title("Monthly Expenses by Category")
    plt.ylabel("Amount (£)")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_savings(data):
    # Ensure the data is properly formatted
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Month'] = data['Date'].dt.to_period('M')

    # Calculate income and expenses for each month
    income_data = data[data['Type'].str.lower() == 'income']
    expense_data = data[data['Type'].str.lower() == 'expense']
    
    monthly_income = income_data.groupby('Month')['Amount'].sum()
    monthly_expenses = expense_data.groupby('Month')['Amount'].sum()
    
    # Calculate savings: Income - Expenses
    savings = monthly_income - monthly_expenses
    
    # Plotting savings
    savings.plot(kind='line', marker='o', figsize=(10, 6), color='green')
    plt.title("Monthly Savings")
    plt.ylabel("Savings (£)")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_expenses_by_category(data, month):
    # Filter for expenses in the specified month
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Month'] = data['Date'].dt.to_period('M')
    expense_data = data[data['Type'].str.lower() == 'expense']
    monthly_expenses = expense_data[expense_data['Month'] == month]
    
    # Group by category and sum the amounts
    category_expenses = monthly_expenses.groupby('Category')['Amount'].sum()
    
    # Plotting the data as a pie chart
    plt.figure(figsize=(7, 7))
    category_expenses.plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.Paired.colors, startangle=90)
    plt.title(f"Expense Breakdown for {month}")
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

def plot_income_vs_expenses(data):
    # Ensure the data is properly formatted
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Month'] = data['Date'].dt.to_period('M')

    # Group income and expense by month
    income_data = data[data['Type'].str.lower() == 'income']
    expense_data = data[data['Type'].str.lower() == 'expense']
    
    monthly_income = income_data.groupby('Month')['Amount'].sum()
    monthly_expenses = expense_data.groupby('Month')['Amount'].sum()

    # Plotting income vs expenses
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_income.index.astype(str), monthly_income, label='Income', color='blue', marker='o')
    plt.plot(monthly_expenses.index.astype(str), monthly_expenses, label='Expenses', color='red', marker='o')
    plt.title("Monthly Income vs Expenses")
    plt.ylabel("Amount (£)")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_to_streamlit(fig):
    # Convert the plot to a PNG image and display in Streamlit
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    st.image(buf)

st.markdown("""
    <style>
        .chat-container {
            position: fixed;
            bottom: 10px;
            right: 10px;
            width: 300px;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #ddd;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            z-index: 1000;
        }
        .chat-container input {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

def plot_monthly_expenses(df):
    if df is None or df.empty:
        st.error("No data available for expenses.")
        return None  # Ensure we don't return an invalid figure
    
    df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime
    df['Month'] = df['Date'].dt.strftime('%Y-%m')  # Extract month as "YYYY-MM"
    
    expense_data = df[df['Type'].str.lower() == 'expense']  # Filter only expenses

    if expense_data.empty:
        st.warning("No expense data available to plot.")
        return None
    
    monthly_expenses = expense_data.groupby(['Month', 'Category'])['Amount'].sum().unstack(fill_value=0)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_expenses.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title("Monthly Expenses Breakdown")
    ax.set_ylabel("Amount Spent")
    ax.set_xlabel("Month")
    ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig 

def main():
    st.title("Wealth Wise")
    min_date = date(1984, 1, 1)
    max_date = date.today()
    dataset = ""
    
    # Step 1: Ask for basic user information
    name = st.text_input("What's your name? (Or should we just call you 'Legend'?)")
    
    dob = st.date_input("What's your birthdate? We promise we won't tell anyone!",min_value= min_date, max_value = max_date)
    age = calculate_age(dob)
    if age < 18:
        st.error("You should be studying, not working on financial planning!")
    elif age > 40:
        st.info("Hmm! You missed the train! It's always never late to start financial planning :P ")
    
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
        
        # Step 3: Ask for additional inputs based on the goal
        if st.session_state.get("goal_selected", False):
            if goal == "Saving":
                goal_amount = st.number_input("Enter your savings goal amount (£):", min_value=0)
                timeframe_months = st.number_input("Enter the timeframe (in months):", min_value=1)
                st.session_state.goal_amount = goal_amount
                st.session_state.timeframe_months = timeframe_months
            
            risk_tolerance = st.selectbox("What's your risk tolerance? No judgment here, we promise!", ["low", "medium", "high"])
            st.session_state.risk_tolerance = risk_tolerance
            
            if st.button("Submit additional details (You're almost there!)"):
                st.session_state.details_submitted = True
                
        # Step 4: Generate and display results based on the goal
        if st.session_state.get("details_submitted", False):
            uploaded_file = st.file_uploader("Upload your bank statement here.", type="csv")
            
            if uploaded_file is not None:
                data = load_financial_data(uploaded_file)
                dataset = data
                if goal == "Budgeting":
                    budget_analysis = analyze_budget(data)
                    st.subheader("Budgeting Analysis")
                    st.write(budget_analysis)
                
                elif goal == "Investing":
                    investment_plan = generate_investment_plan(data, st.session_state.risk_tolerance)
                    st.subheader("Investment Plan")
                    st.write(investment_plan)
                
                elif goal == "Saving":
                    savings_plan = create_savings_plan(data, st.session_state.goal_amount, st.session_state.timeframe_months)
                    st.subheader("Savings Plan")
                    st.write(savings_plan)
                
                elif goal == "Planning":
                    financial_plan = generate_financial_plan(data, st.session_state.risk_tolerance, st.session_state.goal_amount, st.session_state.timeframe_months)
                    st.subheader("Comprehensive Financial Plan")
                    st.write(financial_plan)

                # **Spending Quiz Integration**
                st.subheader("Test Your Financial Wisdom! 🎯")
                quiz_questions = generate_spending_quiz(data, num_questions=3)
                
                if 'current_question' not in st.session_state:
                    st.session_state.current_question = 0
                    st.session_state.score = 0

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
            
            # st.subheader("Monthly Expenses Breakdown")
            # fig1 = plot_monthly_expenses(dataset)  
            # if fig1: 
            #     plot_to_streamlit(fig1)
            # else:
            #     st.warning("Could not generate the expenses chart.")        


    # Chat section always visible at the corner
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Savings Section
    # st.subheader("Monthly Savings Trend")
    # fig2 = plot_savings(calculate_savings(data))
    # plot_to_streamlit(fig2)

    # # Income vs Expenses Section
    # st.subheader("Income vs Expenses Over Time")
    # fig3 = plot_income_vs_expenses(data)
    # plot_to_streamlit(fig3)
    st.subheader("Chat with your Financial Advisor")
    user_query = st.text_input("Got a question about your finances? Ask away!")
    if user_query:
        response = chatbot_response(user_query)
        st.write(response)
    st.markdown('</div>', unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
