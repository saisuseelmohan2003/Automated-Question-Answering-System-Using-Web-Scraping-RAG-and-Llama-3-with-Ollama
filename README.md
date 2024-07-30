# Automated Question Answering System Using Web Scraping, RAG, and Llama 3 with Ollama 🤖📚

## Overview 🚀

This project is an **Automated Question Answering System** that uses web scraping, retrieval-augmented generation (RAG), and the Llama 3 model from Ollama to generate, categorize, validate, and answer customer questions about products. 

## Features 💡

- **Web Scraping**: Extracts detailed product information from various websites. 🌐
- **Question Generation**: Uses AI to create relevant questions about products. 🤔
- **Validation**: Ensures questions meet specific criteria before answering. ✅
- **Answer Generation**: Provides accurate and context-specific answers using the Ollama Llama 3 model. 🦙📚
- **Data Handling**: Manages data efficiently with Pandas and MySQL. 💾

## How It Works 🛠️

1. **Data Collection** 📥:
   - The system starts by reading URLs from an Excel file.
   - For each URL, it uses BeautifulSoup to scrape detailed product information such as title, features, key features, description, product information, and specifications.

2. **Question Generation** 🤔:
   - The project uses the OpenAI GPT-3.5 model to generate relevant questions based on the product information.
   - Questions are categorized into predefined categories to ensure a diverse set of inquiries.

3. **Question Validation** ✅:
   - The generated questions are checked against a set of criteria to identify any issues such as foul language, prohibited content, personal identity information, non-English script, and seller-related info.
   - If a question meets any rejection criteria, it is marked as rejected with the reason specified.

4. **Answer Generation** 🦙📚:
   - For valid questions, the system uses the Ollama Llama 3 model to generate precise and contextually accurate answers based on the scraped product information.
   - The answer is then stored along with the question in the MySQL database.

5. **Data Storage** 💾:
   - All processed data, including URLs, questions, categories, validation statuses, rejection reasons, and answers, are stored in a MySQL database.
   - The final results are exported to an updated Excel file and a JSON file for easy access and further analysis.

## Technologies Used 🛠️

- **Python**: For scripting and automation.
- **Pandas**: For data manipulation and analysis.
- **MySQL**: For database management.
- **BeautifulSoup**: For web scraping.
- **LangChain**: For integrating AI models.
- **OpenAI GPT-3.5**: For question generation.
- **Ollama Llama 3**: For answer generation.

## How To Run 🧑‍💻


1. **Set up MySQL database**:
    - Create a database named `project`.
    - Create a table `mytable` with columns `urls`, `questions`, `category`, `question_status`, `rejection_reason`, and `answers`.

2. **Configure environment variables**:
    
    OPENAI_API_KEY='your-openai-api-key'
   

## Usage 📚

1. **Prepare the input file**: 
    - Place the Excel file with URLs in the designated path ('choose a path for input file').

2. **Run the main script**   

3. **Check the outputs**:
    - Updated Excel file: `choose a path for updating the input excel file`
    - JSON file: `choose a path for output json file`


*Happy Coding!* 👨‍💻👩‍💻
