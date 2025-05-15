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

!(blob:https://web.whatsapp.com/f8d3d526-6a5e-4c39-bba2-c195269d5071)
### 4. Result
Our multi-agent system successfully demonstrates the ability to process data analysis tasks and generate accurate visualizations. The system leverages a combination of conversational agents and specialized data processing agents to perform tasks efficiently.
The example that we provided is this:


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
**At the end of the notebook, we added three lists of prompts divided by difficulty (easy, intermediate and hard) that we tested on Agents.**
