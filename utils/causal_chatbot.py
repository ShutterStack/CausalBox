# utils/causal_chatbot.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.preprocessor import summarize_dataframe_for_chatbot
from utils.graph_utils import get_graph_summary_for_chatbot
import pandas as pd

load_dotenv()

# Configure Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY environment variable not set.")
    raise ValueError("GROQ_API_KEY is required.")

# Debug: Print API key details
print(f"Loaded GROQ_API_KEY: {GROQ_API_KEY[:5]}...{GROQ_API_KEY[-5:]}")
print(f"API Key Length: {len(GROQ_API_KEY)}")

# Initialize the Groq model with LangChain
try:
    model = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=GROQ_API_KEY
    )
except Exception as e:
    print(f"Error configuring Groq API: {e}")
    model = None

def assess_causal_compatibility(data_json: list) -> str:
    """
    Assesses the dataset's compatibility for causal inference analysis.
    
    Args:
        data_json: List of dictionaries representing the dataset.
    
    Returns:
        String describing the dataset's suitability for causal analysis.
    """
    if not data_json:
        return "No dataset provided for compatibility assessment."
    
    try:
        df = pd.DataFrame(data_json)
        num_rows, num_cols = df.shape
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        missing_values = df.isnull().sum().sum()

        assessment = [
            f"Dataset has {num_rows} rows and {num_cols} columns.",
            f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols) if len(numeric_cols) > 0 else 'None'}.",
            f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols) if len(categorical_cols) > 0 else 'None'}.",
            f"Missing values: {missing_values}."
        ]

        # Causal compatibility insights
        if num_cols < 3:
            assessment.append("Warning: Dataset has fewer than 3 columns, which may limit causal analysis (e.g., no room for treatment, outcome, and confounders).")
        if len(numeric_cols) == 0:
            assessment.append("Warning: No numeric columns detected. Causal inference often requires numeric variables for treatment or outcome.")
        if missing_values > 0:
            assessment.append("Note: Missing values detected. Preprocessing (e.g., imputation) may be needed for accurate causal analysis.")
        if len(numeric_cols) >= 2 and num_rows > 100:
            assessment.append("Positive: Dataset has multiple numeric columns and sufficient rows, suitable for causal inference with proper preprocessing.")
        else:
            assessment.append("Note: Ensure at least two numeric columns (e.g., treatment and outcome) and sufficient data points for robust causal analysis.")

        return "\n".join(assessment)
    except Exception as e:
        print(f"Error in assess_causal_compatibility: {e}")
        return "Unable to assess dataset compatibility due to processing error."

# Define tools using LangChain's @tool decorator
@tool
def get_dataset_info() -> dict:
    """
    Provides summary information and causal compatibility assessment for the currently loaded dataset.
    The dataset is provided by the backend session context.
    
    Returns:
        Dictionary containing the dataset summary and compatibility assessment.
    """
    return {"summary": "Dataset will be provided by session context"}

@tool
def get_causal_graph_info() -> dict:
    """
    Provides summary information about the currently discovered causal graph.
    The graph data is provided by the backend session context.
    
    Returns:
        Dictionary containing the graph summary.
    """
    return {"summary": "Graph data will be provided by session context"}

# Bind tools to the model
tools = [get_dataset_info, get_causal_graph_info]
if model:
    model_with_tools = model.bind_tools(tools)

def get_chatbot_response(user_message: str, session_context: dict) -> str:
    """
    Gets a response from the Groq chatbot, handling tool calls.

    Args:
        user_message: The message from the user.
        session_context: Dictionary containing current session data
                        (e.g., processed_data, causal_graph_adj, causal_graph_nodes).

    Returns:
        The chatbot's response message.
    """
    if model is None:
        return "Chatbot is not configured correctly. Please check Groq API key."

    try:
        # Create a prompt template to guide the model's behavior
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are CausalBox Assistant, an AI that helps users analyze datasets and causal graphs.
            Use the provided tools to access dataset or graph information. Do NOT generate or guess parameters for tool calls; the backend will provide all necessary data (e.g., dataset or graph details).
            For dataset queries (e.g., "read the dataset", "dataset compatibility"), call `get_dataset_info` without arguments.
            For graph queries (e.g., "describe the causal graph"), call `get_causal_graph_info` without arguments.
            For other questions (e.g., "what is a confounder?"), respond directly with clear, accurate explanations.
            
            When you receive tool results, provide a comprehensive analysis and explanation to help the user understand their data and causal analysis possibilities.
            
            Examples:
            - User: "Tell me about the dataset" -> Call `get_dataset_info`.
            - User: "Check dataset compatibility for causal analysis" -> Call `get_dataset_info`.
            - User: "Describe the causal graph" -> Call `get_causal_graph_info`.
            - User: "What is a confounder?" -> Respond: "A confounder is a variable that influences both the treatment and outcome, causing a spurious association."
            """),
            ("human", "{user_message}")
        ])

        # Chain the prompt with the model
        chain = prompt | model_with_tools

        # Log the user message and session context
        print(f"Processing user message: {user_message}")
        print(f"Session context keys: {list(session_context.keys())}")

        # Invoke the chain with the user message
        response = chain.invoke({"user_message": user_message})
        print(f"Model response: {response}")

        # Handle tool calls if present
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            function_name = tool_call["name"]
            function_args = tool_call["args"]

            print(f"Chatbot calling tool: {function_name} with args: {function_args}")

            # Map session context to tool arguments
            tool_output = {}
            if function_name == "get_dataset_info":
                data_json = session_context.get("processed_data", [])
                if not isinstance(data_json, list) or not data_json:
                    print(f"Invalid or empty data_json: {data_json}")
                    return "Error: No valid dataset available."
                tool_output = get_dataset_info.invoke({})
                tool_output["summary"] = summarize_dataframe_for_chatbot(data_json)
                tool_output["causal_compatibility"] = assess_causal_compatibility(data_json)
            elif function_name == "get_causal_graph_info":
                graph_adj = session_context.get("causal_graph_adj", [])
                nodes = session_context.get("causal_graph_nodes", [])
                if not graph_adj or not nodes:
                    print("No causal graph data available")
                    return "Error: No causal graph available."
                tool_output = get_causal_graph_info.invoke({})
                tool_output["summary"] = get_graph_summary_for_chatbot(graph_adj, nodes)
            else:
                print(f"Unknown tool: {function_name}")
                return f"Error: Unknown tool {function_name}."

            print(f"Tool output: {tool_output}")

            # Create the tool output text
            output_text = tool_output["summary"]
            if tool_output.get("causal_compatibility"):
                output_text += "\n\nCausal Compatibility Assessment:\n" + tool_output["causal_compatibility"]

            # Create messages for the final response - FIXED VERSION
            messages = [
                HumanMessage(content=user_message),
                AIMessage(content="", tool_calls=[tool_call]),
                ToolMessage(content=output_text, tool_call_id=tool_call["id"])
            ]

            # Create a follow-up prompt to ensure the model provides a comprehensive response
            follow_up_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are CausalBox Assistant. Based on the tool results, provide a comprehensive, helpful response to the user's question. 
                Explain the dataset characteristics, causal compatibility, and provide actionable insights for causal analysis.
                Be specific about what the data shows and what causal analysis approaches would be suitable.
                Always provide a complete response, not just acknowledgment."""),
                ("human", "{original_question}"),
                ("assistant", "I'll analyze the dataset information for you."),
                ("human", "Here's the dataset analysis: {tool_results}\n\nPlease provide a comprehensive explanation of this data and its suitability for causal analysis.")
            ])

            # Get final response from the model with explicit prompting
            print("Invoking model with tool response messages")
            try:
                final_chain = follow_up_prompt | model
                final_response = final_chain.invoke({
                    "original_question": user_message,
                    "tool_results": output_text
                })
                print(f"Final response content: {final_response.content}")
                
                if final_response.content and final_response.content.strip():
                    return final_response.content
                else:
                    # Fallback response if model still returns empty
                    return create_fallback_response(output_text, user_message)
                    
            except Exception as e:
                print(f"Error in final response generation: {e}")
                return create_fallback_response(output_text, user_message)
                
        else:
            print("No tool calls, returning direct response")
            if response.content and response.content.strip():
                return response.content
            else:
                return "I'm ready to help you with causal analysis. Please ask me about your dataset, causal graphs, or any causal inference concepts you'd like to understand."

    except Exception as e:
        print(f"Error communicating with Groq: {e}")
        return f"Sorry, I'm having trouble processing your request: {str(e)}"

def create_fallback_response(tool_output: str, user_message: str) -> str:
    """
    Creates a fallback response when the model returns empty content.
    """
    response_parts = ["Based on your dataset analysis:\n"]
    
    if "Dataset Summary:" in tool_output:
        response_parts.append("📊 **Dataset Overview:**")
        summary_part = tool_output.split("Dataset Summary:")[1].split("Causal Compatibility Assessment:")[0]
        response_parts.append(summary_part.strip())
        response_parts.append("")
    
    if "Causal Compatibility Assessment:" in tool_output:
        response_parts.append("🔍 **Causal Analysis Compatibility:**")
        compatibility_part = tool_output.split("Causal Compatibility Assessment:")[1]
        response_parts.append(compatibility_part.strip())
        response_parts.append("")
    
    # Add specific insights based on the data
    if "FinalExamScore" in tool_output:
        response_parts.append("💡 **Key Insights for Causal Analysis:**")
        response_parts.append("- Your dataset appears to be education-related with variables like FinalExamScore, StudyHours, and TuitionHours")
        response_parts.append("- This is excellent for causal analysis as you can explore questions like:")
        response_parts.append("  • Does increasing study hours causally improve exam scores?")
        response_parts.append("  • What's the causal effect of tutoring (TuitionHours) on performance?")
        response_parts.append("  • How does parental education influence student outcomes?")
        response_parts.append("")
        response_parts.append("🚀 **Next Steps:**")
        response_parts.append("- Consider identifying your treatment variable (e.g., TuitionHours)")
        response_parts.append("- Define your outcome variable (likely FinalExamScore)")
        response_parts.append("- Identify potential confounders (ParentalEducation, SchoolType)")
    
    return "\n".join(response_parts)