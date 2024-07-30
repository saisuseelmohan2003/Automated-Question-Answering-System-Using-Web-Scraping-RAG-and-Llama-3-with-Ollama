import pandas as pd
import os
import random
import mysql.connector
import requests
from bs4 import BeautifulSoup
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  # Import OllamaLLM from LangChain

# Project name and greeting
print("Automated Question Answering System Using Web Scraping, RAG and Llama 3 with Ollama")
print("Welcome to the Automated Question Answering System!")

# Set the OpenAI API key and temperature
openai_api_key = ''
os.environ["OPENAI_API_KEY"] = openai_api_key
temperature = 0  # Set temperature to 0

# Define the prompt templates
question_prompt = PromptTemplate(
    input_variables=["url", "category"],
    template="Generate a question that {category} for the product at the following URL: {url}"
)

# Initialize the OpenAI model with temperature setting
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)

# Define the chains using the pipe operator
question_chain = question_prompt | llm

def extract_text(content):
    """
    Helper function to extract the main text from the response content.
    Handles both single and double quotes.
    """
    if "content='" in content:
        start = content.find("content='") + len("content='")
        end = content.rfind("' response_metadata")
    elif 'content="' in content:
        start = content.find('content="') + len('content="')
        end = content.rfind('" response_metadata')
    else:
        return content
    
    if start != -1 and end != -1:
        return content[start:end]
    return content

def generate_question(url, category):
    try:
        response = question_chain.invoke({"url": url, "category": category})
        # Extract and return only the text content
        if isinstance(response, dict) and 'content' in response:
            return extract_text(response['content'])
        elif hasattr(response, 'text'):
            return extract_text(response.text)
        return extract_text(str(response))
    except Exception as e:
        print(f"Error generating question for URL {url}: {e}")
        return "Error generating question"

# Categories for questions
categories = [
    "is a valid question according to the product: Ask about title or features or key features or specifications or description or product info of the product only",
    "contains foul language",
    "contains seller related question",
    "is product irrelevant",
    "contains illegal content",
    "reveals personal identity",
    "is in non English script",
    "is miscellaneous",
    "contains prohibited content",
    "is a hallucinating question for AI model"
]

# Ensure categories are evenly distributed
def categorize_questions(num_urls):
    category_counts = {category: 0 for category in categories}
    category_limit = max(1, num_urls // len(categories))
    
    categorized_questions = []
    
    for _ in range(num_urls):
        while True:
            category = random.choice(categories)
            if category_counts[category] < category_limit:
                category_counts[category] += 1
                categorized_questions.append(category)
                break
                
    # Adjust for any remaining categories if num_urls is not perfectly divisible
    remaining_categories = num_urls - len(categorized_questions)
    if remaining_categories > 0:
        additional_categories = random.choices(categories, k=remaining_categories)
        categorized_questions.extend(additional_categories)
    
    return categorized_questions

# Read the Excel file
file_path = ""

df = pd.read_excel(file_path)

# Ensure the DataFrame columns have string type
df['questions'] = df['questions'].astype(str)

# Update DataFrame with generated questions
categorized_questions = categorize_questions(len(df))

for i, url in enumerate(df['urls']):
    category = categorized_questions[i]
    question = generate_question(url, category)
    df.at[i, 'questions'] = question  # Ensure value is a string
    df.at[i, 'category'] = category

# Save the updated DataFrame back to Excel
output_file_path = ""
df.to_excel(output_file_path, index=False)

print("Successfully completed")

# MySQL connection details
MYSQL_HOST = ''
MYSQL_USER = ''
MYSQL_PASSWORD = ''
MYSQL_DB = ''
MYSQL_PORT = 

# Insert data into MySQL
def insert_into_mysql():
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            port=MYSQL_PORT
        )
        cursor = conn.cursor()

        for i, row in df.iterrows():
            insert_query = """
            INSERT INTO mytable (urls, questions, category, question_status, rejection_reason, answers)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (row['urls'], row['questions'], row['category'], None, None, None))
        
        conn.commit()
        print("Data inserted into MySQL successfully")
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("MySQL connection closed")

insert_into_mysql()

# Initialize the Ollama model
model = OllamaLLM(model="llama3", temperature=0)

# Web scraping function
def scrape_product_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extracting product details
        title_elem = soup.find('div', {'class': 'product-header-name jm-mb-xs jm-body-m-bold', 'id': 'pdp_product_name'})
        features_elems = soup.select('ul.product-key-features-list li.product-key-features-list-item')
        features = [feature.text.strip() for feature in features_elems] if features_elems else ["Features not found"]

        key_features_elem = soup.find('section', {'class': 'product-key-features'})
        if key_features_elem:
            key_features_list = key_features_elem.find('ul', {'class': 'product-key-features-list'})
            if key_features_list:
                key_features_elems = key_features_list.find_all('li')
                key_features = [feature.text.strip() for feature in key_features_elems]
            else:
                key_features = ["Key Features not found"]
        else:
            key_features = ["Key Features section not found"]

        # Extracting specifications
        specifications = {}
        specifications_section = soup.find('section', {'class': 'product-specifications'})
        if specifications_section:
            tables = specifications_section.find_all('table', {'class': 'product-specifications-table'})
            for table in tables:
                rows = table.find_all('tr', {'class': 'product-specifications-table-item'})
                for row in rows:
                    header = row.find('th', {'class': 'product-specifications-table-item-header'})
                    data = row.find('td', {'class': 'product-specifications-table-item-data'})
                    if header and data:
                        specifications[header.text.strip()] = data.text.strip()

        description_elem = soup.find('div', {'id': 'pdp_description'})
        product_info_elem = soup.find('div', {'id': 'pdp_product_information'})

        title = title_elem.text.strip() if title_elem else "Title not found"
        description = description_elem.text.strip() if description_elem else "Description not found"
        product_info = product_info_elem.text.strip() if product_info_elem else "Product Information not found"

        return {
            'Title': title,
            'Features': features,
            'Key Features': key_features,
            'Description': description,
            'Product Information': product_info,
            'Specifications': specifications
        }
    except Exception as e:
        print(f"Error scraping URL {url}: {e}")
        return None

# Function to check if the question meets rejection criteria and relevance
def check_question(question, product_data):
    prompt = f"""
    Consider yourself as a Customer Question Answering System. Your task is to approve or reject customer questions based on the given rejection criteria. If a question is rejected, provide the reason for rejection.

    ### Rejection Criteria:
    1. **Foul language**: Offensive, derogatory, or harmful language towards individuals, religions, society, national sentiments, causing hurt, anger, or resentment.
    2. **Prohibited or Illegal content**: Content promoting or encouraging illegal activities or products (e.g., drugs, weapons, stolen goods, fraud, extortion, money laundering).
    3. **Personal identity**: Contains contact details (phone number, email, payment details, address) or refers to orders, shipments, or uses any type of pronouns.
    4. **Non-English script**: Hinglish (Hindi sentence typed in English) or any other non-English script that is not fully understandable.
    5. **Seller-related info**: Mentions sellers, vendors, merchants, apps, experiences, specific brands (e.g., Jiomart, Flipkart, Amazon), return/replacement/refund policies, customer service, support, warranty, installation, setup, insurance, delivery, or shipping.

    ### Context:
    **Product Title**: {product_data['Title']}
    **Product Description**: {product_data['Description']}
    **Product Information**: {product_data['Product Information']}
    **Key Features**: {', '.join(product_data['Key Features'])}
    **Specifications**: {', '.join([f"{key}: {value}" for key, value in product_data['Specifications'].items()])}
    **Features**: {', '.join(product_data['Features'])}

    ### Rules:
    1. First, check if the question meets any of the rejection criteria.
    2. If the question meets any rejection criteria, reject the question even if it is relevant to the product context.
    3. If the question does not meet any rejection criteria and is relevant to the product context, approve the question.

    ### Customer Question:
    **Question**: {question}

    ### Output:
    Only print in this format:
    "Rejection Reason: [Reason(s) to reject the question, comma-separated]"
    """

    # Example Section (optional):
    # To further clarify the rejection criteria, you can add examples for each rejection reason. This will help ensure the system understands each criterion clearly.
    """
    Examples:
    1. "Is this product good?" - Approve
    2. "What is the battery life?" - Approve
    3. "Give me your phone number." - Rejection Reason: Personal identity
    4. "Is this product available on Flipkart?" - Rejection Reason: Seller-related info
    5. "Yeh product kaisa hai?" - Rejection Reason: Non-English script
    6.What is your favorite fruit to enjoy on a hot summer day?  - Rejection Reason: Personal identity
    7.What is the best way to use grapefruit in a cocktail to enhance its flavor and potency? - Rejection Reason: Prohibited or Illegal content
    8.Is this 925 silver set available for purchase from multiple sellers or only from Reliance Jewels on the JioMart platform? - Rejection Reason: Seller-related info

    """
    
    try:
        response = model.invoke(prompt)
        rejection_response = response.strip()
        if "Rejection Reason:" in rejection_response:
            rejection_reason = rejection_response.split("Rejection Reason:")[1].strip()
            return rejection_reason if rejection_reason else None
    except Exception as e:
        print(f"Error checking rejection criteria for question '{question}': {e}")
        return "Error checking rejection criteria"

# Function to generate answer using Llama 3 with Ollama
def generate_answer(question, product_data):
    context = f"""
    Title: {product_data['Title']}
    Features: {', '.join(product_data['Features'])}
    Key Features: {', '.join(product_data['Key Features'])}
    Description: {product_data['Description']}
    Product Information: {product_data['Product Information']}
    Specifications: {product_data['Specifications']}
    """
    try:
        # Prepare the input for Ollama
        input_text = f"""
        Context: {context}
        Question: {question}
        Consider yourself as a Customer Question Answering System. Answer the question using the context only in a polite and concise manner. If the question is out of context, respond with exactly "out of context and question is approved by mistake." and nothing else.
        """
        # Use Ollama's library to generate the response
        response = model.invoke(input_text)
        answer = response.strip()

        return answer
    except Exception as e:
        print(f"Error generating answer for question '{question}': {e}")
        return None
    
def process_questions():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        port=MYSQL_PORT
    )
    cursor = conn.cursor()

    select_query = "SELECT * FROM mytable WHERE question_status IS NULL"
    cursor.execute(select_query)
    rows = cursor.fetchall()

    for row in rows:
        url = row[0]  # Assuming the URL is in the 1st column
        question = row[1]  # Assuming the question is in the 2nd column
        print(f"Processing question: {question}")

        product_data = scrape_product_data(url)
        if product_data:
            question_result = check_question(question, product_data)
            if question_result==None:
                answer = generate_answer(question, product_data)
                update_query = """
                UPDATE mytable 
                SET answers = %s, question_status = %s, rejection_reason = %s 
                WHERE urls = %s AND questions = %s
                """
                cursor.execute(update_query, (answer, 'Approved', None, url, question))
                print("Answer column is updated")
            elif "None" not in question_result:
                update_query = """
                UPDATE mytable 
                SET question_status = %s, rejection_reason = %s 
                WHERE urls = %s AND questions = %s
                """
                cursor.execute(update_query, ('Rejected',question_result, url, question))
            else:
                answer = generate_answer(question, product_data)
                update_query = """
                UPDATE mytable 
                SET answers = %s, question_status = %s, rejection_reason = %s 
                WHERE urls = %s AND questions = %s
                """
                cursor.execute(update_query, (answer, 'Approved', None, url, question))
                print("Answer column is updated")

    conn.commit()
    cursor.close()
    conn.close()

process_questions()


# Convert table to JSON
def convert_to_json():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        port=MYSQL_PORT
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM mytable")
    rows = cursor.fetchall()

    json_data = json.dumps(rows, indent=4)

    with open('#write your output json path', 'w') as json_file:
        json_file.write(json_data)
    print("JSON file of the table is created successfully")

    cursor.close()
    conn.close()

convert_to_json()
