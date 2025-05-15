**Team Members**
- Alessio Attolico (794541)
- Gaetano D'Elia (805091)
- Andrea Roscini (807141)

### 1. Introduction
This project implements a multi-agent system designed to process and visualize data from various public administration datasets. The system utilizes a combination of natural language processing, data analysis, and visualization techniques to provide insightful analyses.
The core functionality revolves around using multiple specialized agents to handle tasks efficiently. These agents leverage advanced libraries such as `pandas`, `seaborn`, and `matplotlib` for data manipulation and visualization, while NLP capabilities are integrated using the `transformers` library and components from `langchain`.

The system is structured to read and analyze four key datasets related to:
- **Access Information**: User authentication methods and demographics.
- **Salary Data**: Employee demographics and salary information.
- **Commuting Patterns**: Distance and commuting patterns of public administration employees.
- **Administrative Data**: Income and demographic distribution among administrators.

The main goal of this multi-agent system is to facilitate automated data analysis and visualization through conversational interactions, making it a valuable tool for data-driven decision-making.


### 2. Methods
We developed a modular multi-agent system using the `swarms` framework, where each agent executes a specific set of tasks in coordination with others. The system leverages structured prompts and modular behaviors to simulate intelligent distributed decision-making.
-  **Architecture**: Built using `swarms`, each agent is defined with memory, tools (e.g., search or code generation), and a swarm strategy.

- **Coordination**: Agents are connected via a `ClusterFlow` (from `clusterops`) that orchestrates task decomposition, agent selection & routing, output aggregation and response generation

-  **Environment**: The project simulates a decentralized processing scenario where agents complete subtasks (e.g., data analysis, visualization).

To ensure efficient inference, we utilized **Gemini** with an API key to execute local language models, leveraging the capabilities of **Gemini 2.0-flash-001** for robust natural language processing.
Our multi-agent system is designed to efficiently manage tasks by directly interacting with the language model, ensuring fast and accurate responses without relying on additional libraries for workflow management. Here we can see a visualization of the workflow of our multiagent system:

![Workflow Agents](https://github.com/ale02pz/Agents_794541/blob/main/images/workflow%20agents.png?raw=true)

### 3. Evaluation
In our opinion, the most effective way to evaluate the performance of the agents was to test them on a variety of tasks of differing complexity. By assessing their ability to handle basic, intermediate, and advanced tasks, we could systematically gauge their strengths and limitations. In this section, we present a list of some of the tasks used during the evaluation. All of these tasks were successfully completed, demonstrating the agents' ability to correctly interpret the input and generate accurate outputs.
This approach allowed us to thoroughly test the agents' capabilities in both data processing and visualization, ensuring they perform consistently across diverse scenarios.
Lists of the used prompts:

list_of_tasks_easy = ['Calculate the total number of digital accesses grouped by authentication method.', "Plot the distribution of genders among digital access users",
"Find the most common payment method for each municipality", "Plot the gender distribution of employees across the top 3 municipalities with the most employees.",
"Calculate the average tax rate for each sector.", "Plot the distribution of age groups within the public sector.", "Plot the number of employees commuting within the same municipality",
"Plot the percentage of the age range in POTENZA in the salary dataframe as a pie plot","Identify the most common authentication method for users aged 25-40.", 
"List all regions where the majority of users are younger than 30. If it is empty print 0", "Find the proportion of employees commuting less than 10 km in each province."]

list_of_tasks_intermediate = ["Compare the distribution of authentication methods between males and females.", "Plot the average number of employees per gender for each payment method.", "Determine the income range with the highest number of employees within each region.", "Identify the top 5 administrations where employees with longer commuting distances (more than 20 km) tend to have higher incomes", "Calculate the average distance commuted by employees earning more than 50,000 in different provinces.",
"Plot a heatmap to visualize the relationship between income level and commuting distance across regions."]

list_of_hard_tasks = ["Calculates the percentage distribution of access methods to the NoiPA portal among users aged 18-30 compared to those over 50, broken down by region of residence", "Identifies the most used payment method for each age group and generates a graph showing whether there are correlations between gender and payment method preference",
"Analyzes commuting data to identify which administrations have the highest percentage of employees who travel more than 20 miles to work",
"Compares the gender distribution of staff among the five municipalities with the largest number of employees, highlighting any significant differences in representation by age group",
"Determines if there is a correlation between the method of portal access (from EntryAccessAdministration) and the average commuting distance (from EntryPendularity) for each administration.", "Identify the relationship between income level and the preferred payment method, grouped by region.", "Identify the top 5 regions where employees aged under 30 are more likely to use digital payment methods compared to employees over 50.", "Create a dashboard with subplots: a bar plot for payment methods by age group; a line chart showing the trend of commuting distances; a heatmap of the distribution of authentication methods by region."]

### 4. Result
Our multi-agent system successfully demonstrates the ability to process data analysis tasks and generate accurate visualizations. The system leverages a combination of conversational agents and specialized data processing agents to perform tasks efficiently. Following there are the executions of some of the tasks listed above.

![Data Processing Task1](https://github.com/ale02pz/Agents_794541/blob/main/images/WhatsApp%20Image%202025-05-15%20at%2018.34.31.jpeg?raw=true)

![Data Processing Task2](https://github.com/ale02pz/Agents_794541/blob/main/images/Screenshot%202025-05-15%20alle%2003.42.27.png?raw=true)

![Visualization prompt1]

![Visualization Task1](https://github.com/ale02pz/Agents_794541/blob/main/images/WhatsApp%20Image%202025-05-15%20at%2018.37.55.jpeg?raw=true)

![Visualization Task2]

### 5. Conclusion
This project demonstrates the effectiveness of a modular multi-agent system for handling complex data tasks through specialized agents. By leveraging Gemini with an API key for robust natural language processing and implementing a structured ClusterFlow for task routing, we successfully simulated intelligent task delegation without the need for traditional reinforcement learning. Our design allows for flexible, interpretable, and reproducible agent-based pipelines that can handle data processing, conversational logic, and visualization in parallel—culminating in a coherent, aggregated output.
While our multi-agent system demonstrates robust performance in routing tasks, data processing, and generating visualizations, there are still some questions that remain partially addressed. One of the primary limitations lies in the system's ability to handle ambiguous natural language inputs, where the conversational agent occasionally misinterprets vague or multi-faceted requests. Additionally, the system's visualization agent sometimes struggles when tasked with comparing categorical variables that lack clear numerical associations.

--------------------------------------------------------------
### HOW TO RUN THE CODE
1. Ensure you have the version Python 3.11.9, if not install it.
2. We strongly suggest to create a dedicated virtual environment and install the packages contained in `requirements.txt`.
3. Once you have installed everything, run the notebook.

### HOW TO USE THE STREAMLIT VERSION
1. Install the streamlit package -> `pip install streamlit`
2. Ensure that app.py is in the same directory of the dataset
3. In the terminal activate the venv -> .\.”venv_name\Scripts\Activate.ps1
4. At the end run this in the terminal -> streamlit run app.py
5. HAVE FUN!

---------------------------------------------------------------
