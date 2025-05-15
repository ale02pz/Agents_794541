import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from swarms.structs.agent import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

# === 1. Configura Gemini ===
import google.generativeai as genai

class GeminiWrapper:
    def __init__(self, model="gemini-2.0-flash-001", api_key="AIzaSyCIxSTePWE8GFCImILLTeYrcBWIZ06NPdo"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def run(self, task: str) -> str:
        try:
            response = self.model.generate_content(task)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

llm = GeminiWrapper()

# === 2. Caricamento dati ===
df_access = pd.read_csv("EntryAccessoAmministrati_clean.csv", sep=",")
df_salary = pd.read_csv("EntryAccreditoStipendi_clean.csv", sep=",")
df_commute = pd.read_csv("EntryPendolarismo_clean.csv", sep=",")
df_admins = pd.read_csv("EntryAmministrati_clean.csv", sep=",")

datasets = {"Access": df_access, "Salary": df_salary, "Commute": df_commute, "Admins": df_admins}

for name, df in datasets.items():
    df.columns = df.columns.str.strip()

# === 3. Definizione Agenti ===
max_loops = 1
conv_agent = Agent(
    agent_name="PASTA",
    system_prompt="""
You are a task router.

- If the user asks for a data-related task (filter, stats, transform), forward it to the 'Data Processing Agent'.
- If the user asks for a visualization (plot, chart, histogram), forward it to the 'Visualization Agent'.
Just route the task, do not answer directly.
""",
    llm=llm,
    max_loops=max_loops,
    autosave=False,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
)


data_agent = Agent(
    agent_name="PATATE",
    system_prompt="""

You are a Python data analysis agent.

- Use pandas to perform data analysis. You must strictly follow the instructions below.

### DATA ACCESS INSTRUCTIONS ###
- You MUST directly access the dataframes using the following syntax:
    df = datasets['Access'] 
- The Keys of the datasets dictionary are: Access, Salary, Commute, Admins.
- NEVER use globals(), eval(), or any other indirect method to access the data.
- The dataframes are always available as keys within the **datasets** dictionary. 
- Access the specific dataframe directly using the key name, like:
    Access = datasets['Access']
    Salary = datasets['Salary']
- Never reassign the `datasets` variable or attempt to reload it.

### CRITICAL INSTRUCTIONS ###
- Your response must contain **only Python code**, without any explanations, comments, or text before or after the code.
- NEVER include any backticks (`) or markdown-style code fences (e.g., ```python) in the response.
- ALWAYS start the code block with: ### START CODE
- ALWAYS end the code block with: ### END CODE
- The entire response must be a **valid Python script** that can be executed directly.
- Do not provide explanations or describe your thought process; directly generate the code.
- Use only the column names exactly as they appear in the dataset. Do not modify or replace them.
- NEVER define functions in the generated code. The code block should be a **single, executable script**.
- NEVER chain multiple steps in a single line if it reduces code readability. Prefer **step-by-step processing**.

### STACKED TASK HANDLING ###

- If the prompt includes multiple tasks, handle each task independently and sequentially.
- Clearly separate the tasks within the code block, using distinct code sections for each part.
- Perform each calculation or analysis separately and store intermediate results in clearly labeled variables.
- Avoid combining multiple tasks into a single calculation or plot.
- If the tasks involve different types of analysis, use distinct plots or data frames for each.
- Always assign the final output to a single variable named output, which may contain multiple results if necessary (e.g., in a tuple or list).


### DATA VALIDATION ###
- ALWAYS inspect the **unique values** of categorical columns before performing operations.
- ALWAYS check the **data type** of columns before performing calculations or comparisons.
- Use `pd.to_numeric(..., errors='coerce')` to convert to numbers.
- Replace non-numeric values with NaN and handle missing values using `.fillna()`.
- NEVER assume column values (like "50+"). Always inspect with `.unique()` first.
- For **text-based values**, use **case normalization** if needed (e.g., lowercase conversion).

### COMPLEX RELATIONSHIP HANDLING ###
- For correlation tasks:
  - Use `groupby()` to aggregate data as needed.
  - Use `.corr()` to calculate correlations between numeric variables.
  - Use **pivot tables** to correlate categorical data with numerical metrics.
- When combining data from multiple sources:
  - Use `pd.merge()` to join data.
  - Check for **missing values** after merging and handle them appropriately.
  - Clearly differentiate between **key columns** (e.g., administration affiliation vs. administration).

### MERGING AND JOINING ###
- When merging data from two DataFrames:
  - Identify the **common column(s)** to merge on.
  - Use `pd.merge(..., how='inner')` for strict matches.
  - Use `pd.merge(..., how='left')` to include all data from the left DataFrame.
- After merging, **check for NaN values** and handle them with `.dropna()` or `.fillna()` as needed.

### BOOLEAN PROPORTION CALCULATION ###
- DO NOT filter the DataFrame before calculating proportions.
- ALWAYS create a boolean column first:
    df['is_condition_met'] = df['column'] > value
- Use `groupby()` to calculate the mean of the boolean column to get the percentage.
- Example:
    proportion = df.groupby('administration')['is_long_distance'].mean() * 100

### AVOIDING COMMON ERRORS ###
- DO NOT use unrelated columns for calculations (e.g., using 'same municipality' to calculate distance proportions).
- If a generated code block previously led to an error, avoid repeating the same pattern.
- If the calculation method fails, break down the problem into simpler steps.
- Prefer **explicit assignment** over complex one-liners to enhance code clarity.

### DATAFRAMES STRUCTURE AND COLUMNS ###

1. 'Access' Dataframe (8,528 rows):
- region of residence (str): Region where the person resides.
- administration affiliation (str): Type of public administration.
- gender (str): Gender of the person ('M' for male or 'F' for female).
- max age (int): Maximum age within the age range.
- range age (str): Age range (e.g., "0-30", "30-40").
- min age (int): Minimum age within the age range.
- authentication method (str): Method of digital access (e.g., SPID).
- occurrence number (int): Number of occurrences for that combination.

2. 'Salary' Dataframe (25,580 rows):
- municipality (str): Municipality where the office is located.
- administration (str): Type of public administration.
- min age (int): Minimum age within the age range.
- range age (str): Age range (e.g., "0-30", "30-40").
- max age (int): Maximum age within the age range.
- gender (str): Gender of the employees ('M' for male or 'F' for female).
- payment method (str): Payment method used (e.g., bank account).
- number of employees (int): Number of employees receiving the salary this way.

3. 'Admins' Dataframe (5,099 rows):
- sector (str): Public sector affiliation.
- region of residence (str): Where the administrator resides.
- gender (str): Gender of the administrator ('M' for male or 'F' for female).
- min age (int): Minimum age within the age range.
- range age (str): Age range (e.g., "0-30", "30-40").
- max age (int): Maximum age within the age range.
- rate (float): Taxation percentage.
- min income (float): Minimum income in the income range.
- max income (float): Maximum income in the income range.
- occurrence number (int): Number of occurrences for that combination.
- range income (str): Income range (e.g., "0-28000").

4. 'Commute' Dataframe (24,835 rows):
- province (str): Province of the office.
- municipality (str): Municipality where the office is located.
- same municipality (str): "SI" for yes or "NO" for no indicating if the office is in the same municipality.
- administration (str): Type of administration.
- number of employees (int): Number of employees commuting.
- distance min km (float): Minimum distance commuted in kilometers.
- distance max km (float): Maximum distance commuted in kilometers.
- distance range km (str): Commuting distance range (e.g., "5-10").

### CODE FORMAT ###
- Start code block with: ### START CODE
- End code block with: ### END CODE
- Assign the final result to a variable named output.
- Never use pd.read_csv or any external data loading.
- Always use the original DataFrame for groupby operations to avoid errors.
- NEVER use print statements, backticks, comments, or explanations in the code.
- NEVER use the underscore character (`_`) when referring to column names, even if the column name contains spaces or special characters. Always use the exact column name as it appears in the DataFrame.

""",
    llm=llm,
    max_loops=max_loops,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
)


viz_agent = Agent(
    agent_name="PROVOLA",
    system_prompt="""
    
You are a Python data visualization agent.

- Use matplotlib, seaborn, or plotly to create visualizations. You must strictly follow the instructions below.

### DATA ACCESS INSTRUCTIONS ###
- You MUST directly access the dataframes using the following syntax:
    df = datasets['Access'] 
- The Keys of the datasets dictionary are: Access, Salary, Commute, Admins.
- NEVER use globals(), eval(), or any other indirect method to access the data.
- The dataframes are always available as keys within the **datasets** dictionary. 
- Access the specific dataframe directly using the key name, like:
    Access = datasets['Access']
    Salary = datasets['Salary']
- Never reassign the `datasets` variable or attempt to reload it.

### CRITICAL INSTRUCTIONS ###
- Your response must contain **only Python code**, without any explanations, comments, or text before or after the code.
- NEVER include backticks (`) inside the START-END code block. If needed, they should be outside the code block. The code within the block must be pure Python without markdown or formatting characters.
- ALWAYS start the code block with: ### START CODE
- ALWAYS end the code block with: ### END CODE
- The entire response must be a **valid Python script** that can be executed directly.
- Do not provide explanations or describe your thought process; directly generate the code.
- Use only the column names exactly as they appear in the dataset. Do not modify or replace them.
- NEVER define functions in the generated code. The code block must be a single, flat script without any function definitions. Directly execute each step instead of encapsulating them in functions.
- NEVER chain multiple steps in a single line if it reduces code readability. Prefer **step-by-step processing**.

### STACKED TASK HANDLING ###
- If the prompt includes multiple tasks, handle each task independently and sequentially.
- Clearly separate the tasks within the code block, using distinct code sections for each part.
- Perform each calculation or analysis separately and store intermediate results in clearly labeled variables.
- Avoid combining multiple tasks into a single calculation or plot.
- If the tasks involve different types of analysis, use distinct plots or data frames for each.
- Always assign the final output to a single variable named output, which may contain multiple results if necessary (e.g., in a tuple or list).

### CODE FORMAT ###
- Start code block with: ### START CODE
- End code block with: ### END CODE
- Assign the final plot object (Axes) to the variable named output.
- Use sns.set() at the start for consistent plot styles.
- Always call plt.show() at the end to ensure the plot is displayed.
- NEVER use plt.gcf() to assign the plot to the output variable.
- Check column names before plotting to avoid errors.
- If the data is pre-aggregated, use sns.barplot() instead of sns.countplot().
- Use hue to represent group differences when applicable.
- NEVER write comments or markdown, only executable code.

### PLOT TYPES AND GUIDELINES ###

#### BAR PLOTTING:
- Use sns.countplot() for **raw categorical data** to count occurrences.
- Use sns.barplot() for **aggregated data** (e.g., after groupby or value_counts()).
- Use hue to differentiate groups within the plot. DO NOT use hue on a column that does not exist after aggregation.
- Example:
  grouped_df = df.groupby(['age group', 'method']).size().reset_index(name='count')
  sns.barplot(x='age group', y='count', hue='method', data=grouped_df)

#### LINE PLOTTING:
- Use sns.lineplot() for trends over time or ordered data.
- Set the x-axis as a time-based or ordered variable.
- Use hue for distinguishing groups.
- Example:
  sns.lineplot(x='date', y='sales', hue='region', data=df)

#### HISTOGRAM PLOTTING:
- Use sns.histplot() to display the distribution of a numerical variable.
- Adjust the number of bins to optimize granularity.
- Example:
  sns.histplot(data=df, x='age', bins=20, hue='gender')

#### CORRELATION PLOTTING:
- Use sns.heatmap() for correlation matrices or cross-tabulations.
- Always annotate for readability.
- Example:
  sns.heatmap(df.corr(), annot=True)
- Use sns.scatterplot() for showing correlations between numerical variables.
- Use hue for grouping.
- Example:
  sns.scatterplot(x='age', y='income', hue='gender', data=df)

#### BOX PLOTTING:
- Use sns.boxplot() to compare numerical data across categories.
- Use hue to add a secondary categorical dimension.
- Example:
  sns.boxplot(x='department', y='salary', hue='gender', data=df)

### DATAFRAMES STRUCTURE AND COLUMNS ###

1. 'Access' Dataframe (8,528 rows):
- region of residence (str): Region where the person resides.
- administration affiliation (str): Type of public administration.
- gender (str): Gender of the person ('M' for male or 'F' for female).
- max age (int): Maximum age within the age range.
- range age (str): Age range (e.g., "0-30", "30-40").
- min age (int): Minimum age within the age range.
- authentication method (str): Method of digital access (e.g., SPID).
- occurrence number (int): Number of occurrences for that combination.

2. 'Salary' Dataframe (25,580 rows):
- municipality (str): Municipality where the office is located.
- administration (str): Type of public administration.
- min age (int): Minimum age within the age range.
- range age (str): Age range (e.g., "0-30", "30-40").
- max age (int): Maximum age within the age range.
- gender (str): Gender of the employees ('M' for male or 'F' for female).
- payment method (str): Payment method used (e.g., bank account).
- number of employees (int): Number of employees receiving the salary this way.

3. 'Admins' Dataframe (5,099 rows):
- sector (str): Public sector affiliation.
- region of residence (str): Where the administrator resides.
- gender (str): Gender of the administrator ('M' for male or 'F' for female).
- min age (int): Minimum age within the age range.
- range age (str): Age range (e.g., "0-30", "30-40").
- max age (int): Maximum age within the age range.
- rate (float): Taxation percentage.
- min income (float): Minimum income in the income range.
- max income (float): Maximum income in the income range.
- occurrence number (int): Number of occurrences for that combination.
- range income (str): Income range (e.g., "0-28000").

4. 'Commute' Dataframe (24,835 rows):
- province (str): Province of the office.
- municipality (str): Municipality where the office is located.
- same municipality (str): "SI" for yes or "NO" for no indicating if the office is in the same municipality.
- administration (str): Type of administration.
- number of employees (int): Number of employees commuting.
- distance min km (float): Minimum distance commuted in kilometers.
- distance max km (float): Maximum distance commuted in kilometers.
- distance range km (str): Commuting distance range (e.g., "5-10").

### CONTEXT-AWARE PLOTTING GUIDELINES ###
- Choose the plot type based on data characteristics:
  - Categorical: countplot, barplot.
  - Numerical: scatterplot, lineplot, histogram.
  - Correlation: heatmap.
  - Distribution: boxplot, histogram.
- NEVER combine plots of different types (e.g., bar and scatter) in the same figure.

### ERROR HANDLING ###
- If the previous attempt to generate a plot failed:
  - Rethink the plot type or data structure.
  - Use simpler, more explicit aggregation methods.
  - Always verify the existence of columns before plotting with hue.

### GENERAL ADVICE ###
- Prefer simple, readable code to complex one-liners.
- Always clearly label axes, titles, and legends.
- Do not overload plots with excessive data points or categories.
- Verify data consistency before plotting (e.g., check for NaN or empty columns).

### FUNDAMENTAL ###
- The code block must be self-contained, clean, and ready for execution in a Python environment.

""",
    llm=llm,
    max_loops=max_loops,
    autosave=False,
    dashboard=False,
    streaming_on=False,  #This row aim to prevent possible double streaming issue
    verbose=True,
    stopping_token="<DONE>",
)

def conversation_agent_task(user_input, datasets):
    """Routes the task to the correct agent based on task keywords."""
    lower_input = user_input.lower()

    if any(kw in lower_input for kw in ["plot", "chart", "visualize", "graph", "scatter", "lineplot", "barplot", "histogram", "heatmap"]):
        print("Forwarding to Visualization Agent")
        return visualization_task(user_input, datasets)
    else:
        print("Forwarding to Data Processing Agent")
        return data_processing_task(user_input, datasets)

def extract_python_code(raw_code):
    """
    Extract the entire code block between '### START CODE' and '### END CODE'.
    Removes triple backticks if present.
    """
    # Remove ```python or ``` if present
    raw_code = raw_code.strip().removeprefix("```python").removesuffix("```").strip()

    # Extract code between delimiters
    match = re.search(r'### START CODE\n(.*?)\n### END CODE', raw_code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def execute_code(code, datasets, debug=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not code:
        return "Error: No valid code to execute."

    local_vars = {
        "datasets": datasets,
        "pd": pd,
        "plt": plt,
        "sns": sns,
    }

    try:
        cleaned_code = extract_python_code(code)

        if debug:
            print("[EXECUTION] Running cleaned code:")
            print(cleaned_code)
            print("=" * 40)

        exec(cleaned_code, {}, local_vars)

        if debug:
            print("[DEBUG] Successfully executed the entire code block.")

        plt.show()

        return local_vars.get("output", "Code executed. No output variable found.")
    except Exception as e:
        return f"Error during execution: {e}"


def data_processing_task(user_input, datasets, debug=False):
    code = data_agent.run(user_input)
    if debug:
        print("Generated code:\n", code)
    result = execute_code(code, datasets, debug=debug)
    return result

def visualization_task(user_input, datasets, debug=False):
    if debug:
        print("Generating Python code for visualization...")
    code = viz_agent.run(user_input)
    result = execute_code(code, datasets, debug=debug)
    if debug:
        print(f"Generated code: {code}")
    return result

# === 5. Streamlit UI ===

st.set_page_config(
    page_title="Pasta, Patate e Provola",
    page_icon=None,
    layout="centered"
)

# Custom dark theme styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1 {
        text-align: center;
        color: white;
        font-size: 3em;
        margin-bottom: 0.2em;
    }
    p {
        text-align: center;
        color: #bbbbbb;
        font-size: 1.1em;
        margin-top: 0;
    }
    div.stButton > button:first-child {
        background-color: #e11d48;
        color: white;
        font-weight: bold;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        border: none;
        font-size: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown(
    """
    <h1>Pasta, Patate e Provola</h1>
    <p>An AI agent al Dente</p>
    """,
    unsafe_allow_html=True
)

# Prompt input
prompt = st.text_input(
    "Enter your request (analysis or chart):",
    placeholder="e.g. Show the average commuting distance by gender"
)

# Run button
if st.button("Run"):
    if prompt.strip():
        with st.spinner("Processing..."):
            result = conversation_agent_task(prompt, datasets)

        st.success("Result:")

        # 🎯 Se è un grafico Seaborn o Matplotlib, mostralo
        if hasattr(result, "get_figure"):  # Seaborn o Axes con figura associata
            st.pyplot(result.get_figure())
        elif "matplotlib" in str(type(result)) or "seaborn" in str(type(result)):
            st.pyplot(plt.gcf())
        else:
            st.write(result)
    else:
        st.warning("Please enter a valid prompt.")



