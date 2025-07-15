# ==============================================================================
# Step 1: Imports and Environment Setup
# ==============================================================================
import os
import shutil
from typing import List, TypedDict, Dict

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field

from langchain_openai.chat_models import ChatOpenAI
from langchain.tools import Tool
from ddgs import DDGS
# Import for image display
from IPython.display import Image, display

# ==============================================================================
# Step 2: Helper Functions & Agent Definitions
# ==============================================================================

def print_directory_tree(root_dir):
    if not os.path.isdir(root_dir):
        print(f"❌ Error: Root directory '{root_dir}' not found.")
        return
    print(f"🌳 Final Project Structure for: {root_dir}")
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '', 1).count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}├── {os.path.basename(root)}/')
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{sub_indent}└── {f}')

def duckduckgo_search(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return "\n\n".join([f"Title: {r['title']}, URL: {r['href']}" for r in results]) if results else "No results found."

duck_tool = Tool.from_function(name="DuckDuckGoSearch", description="Search the web for code examples.", func=duckduckgo_search)
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

def generate_frontend_code(spec: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior frontend engineer. Generate responsive React components with TailwindCSS. Use functional components and hooks. Return ONLY code in a single block."),
        ("human", "Here is the component spec:\n{spec}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"spec": spec})

def generate_backend_code(spec: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior Python backend engineer. Generate clean, efficient code for SQLAlchemy models and FastAPI CRUD endpoints based on the spec. Use Python typing and docstrings. Return only code."),
        ("human", "Here is the component spec:\n{spec}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"spec": spec})

# ==============================================================================
# Step 3: Define the Graph State
# ==============================================================================
class AgentState(TypedDict):
    srs_path: str; diagram_paths: dict; coding_plan: str; is_plan_sufficient: bool;
    project_root: str; file_structure: List[str]; tasks: List[dict];
    current_task_index: int; current_code: str; test_results: str;
    revision_count: int; final_codebase: Dict[str, str]

# ==============================================================================
# Step 4: Define the Graph Nodes (Agent Actions)
# ==============================================================================

def planner_agent_node(state: AgentState):
    print("--- AGENT: Planner ---")
    srs_content = PyPDFLoader(state['srs_path']).load()
    if not srs_content: return {"is_plan_sufficient": False, "coding_plan": "SRS document is empty."}
    prompt = ChatPromptTemplate.from_template("Based on the SRS document, create a step-by-step coding plan. For each step, specify the file path and what needs to be coded.\n\nSRS:\n{srs}")
    chain = prompt | llm | StrOutputParser()
    plan = chain.invoke({"srs": str(srs_content)})
    print("  ✅ Plan generated.")
    return {"is_plan_sufficient": True, "coding_plan": plan}

def architect_agent_node(state: AgentState):
    print("--- AGENT: Architect ---")
    class FilePathList(BaseModel):
        paths: List[str] = Field(description="A list of all file and directory paths to create.")
    parser = JsonOutputParser(pydantic_object=FilePathList)
    prompt = ChatPromptTemplate.from_template("Parse the coding plan and extract all file and directory paths. Return as a JSON array.\n\nPLAN:\n{plan}\n\n{format_instructions}")
    chain = prompt | llm | parser
    file_list = chain.invoke({"plan": state['coding_plan'], "format_instructions": parser.get_format_instructions()})['paths']
    root_dir = "generated_project"
    if os.path.exists(root_dir): shutil.rmtree(root_dir)
    os.makedirs(root_dir)
    for path_item in file_list:
        full_path = os.path.join(root_dir, path_item.strip())
        if path_item.endswith('/') or '.' not in os.path.basename(path_item):
            os.makedirs(full_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f: pass
    print("  ✅ Project structure created.")
    return {"project_root": root_dir, "file_structure": file_list}

def task_decomposer_node(state: AgentState):
    print("--- AGENT: Task Decomposer ---")
    class Task(BaseModel):
        file_path: str; task_description: str; type: str
    class TaskList(BaseModel):
        tasks: List[Task]
    parser = JsonOutputParser(pydantic_object=TaskList)
    prompt = ChatPromptTemplate.from_template("Decompose the coding plan into tasks. Only create tasks for files, not directories. For each file, identify its path, a detailed task description, and the code type ('frontend-react' or 'backend-python').\n\nPLAN:\n{plan}\n\n{format_instructions}")
    chain = prompt | llm | parser
    tasks = chain.invoke({"plan": state['coding_plan'], "format_instructions": parser.get_format_instructions()})['tasks']
    print(f"  ✅ Plan decomposed into {len(tasks)} file-based tasks.")
    return {"tasks": tasks, "current_task_index": 0, "final_codebase": {}}

def coder_agent_node(state: AgentState):
    print("--- AGENT: Coder ---")
    task = state['tasks'][state['current_task_index']]
    print(f"  - Working on task {state['current_task_index'] + 1}/{len(state['tasks'])}: {task['file_path']}")
    if 'frontend' in task['type']: code = generate_frontend_code(task['task_description'])
    elif 'backend' in task['type']: code = generate_backend_code(task['task_description'])
    else: code = f"# Task: {task['task_description']}"
    return {"current_code": code, "revision_count": 0}

def tester_agent_node(state: AgentState):
    print("--- AGENT: Tester ---")
    task, code = state['tasks'][state['current_task_index']], state['current_code']
    ### --- FIX --- ### Improved, more structured prompt for the tester.
    prompt = ChatPromptTemplate.from_template(
        "You are a meticulous QA engineer. Review the code for file `{file_path}` based on its task. Be critical. If the code is perfect, respond with only the word 'PASSED'. Otherwise, respond with 'FAILED' followed by a bulleted list of specific, actionable changes required. Mention line numbers if possible.\n\nTASK:\n{task}\n\nCODE:\n```\n{code}\n```"
    )
    chain = prompt | llm | StrOutputParser()
    results = chain.invoke({"file_path": task['file_path'], "task": task['task_description'], "code": code})
    print(f"  - Test results for {task['file_path']}: {results.splitlines()[0]}")
    return {"test_results": results}

def reviser_agent_node(state: AgentState):
    print("--- AGENT: Reviser ---")
    task = state['tasks'][state['current_task_index']]
    print(f"  - Revising {task['file_path']} based on feedback.")
    ### --- FIX --- ### Improved prompt for the reviser to be more focused.
    prompt = ChatPromptTemplate.from_template(
        "You are a senior developer. Rewrite the code to address ALL points in the feedback. Do not add new features or commentary. Return ONLY the full, corrected code.\n\nORIGINAL CODE:\n```\n{code}\n```\n\nREQUIRED CHANGES:\n{feedback}\n\n"
    )
    chain = prompt | llm | StrOutputParser()
    revised_code = chain.invoke({"code": state['current_code'], "feedback": state['test_results']})
    return {"current_code": revised_code, "revision_count": state.get('revision_count', 0) + 1}

def integrator_agent_node(state: AgentState):
    print("--- AGENT: Integrator ---")
    task, code = state['tasks'][state['current_task_index']], state['current_code']
    file_path = os.path.join(state['project_root'], task['file_path'])
    if '.' in os.path.basename(file_path):
        with open(file_path, 'w', encoding='utf-8') as f: f.write(code)
        print(f"  ✅ Code integrated into {file_path}")
        final_codebase = state.get('final_codebase', {})
        final_codebase[task['file_path']] = code
    else:
        print(f"  - WARNING: Skipping integration for directory-like path: {task['file_path']}")
        final_codebase = state.get('final_codebase', {})
    return {"final_codebase": final_codebase, "current_task_index": state.get('current_task_index', 0) + 1}

# ==============================================================================
# Step 5: Define Conditional Edges for Graph Logic
# ==============================================================================
def should_start_coding(state: AgentState) -> str:
    print("---CONDITION: Plan Sufficient?---")
    if state.get("is_plan_sufficient"): return "proceed_to_architect"
    return "end_workflow"

def should_revise_or_integrate(state: AgentState) -> str:
    print("---CONDITION: Code Passed Tests?---")
    revision_limit = 3
    if "FAILED" in state['test_results'].strip().upper() and state.get('revision_count', 0) < revision_limit:
        return "needs_revision"
    return "integrate_as_is"

def should_continue_coding(state: AgentState) -> str:
    print("---CONDITION: More Tasks To Code?---")
    if state.get('tasks') and state['current_task_index'] < len(state['tasks']):
        return "continue_coding"
    return "end_workflow"

# ==============================================================================
# Step 6: Assemble and Visualize the Graph
# ==============================================================================
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_agent_node)
workflow.add_node("architect", architect_agent_node)
workflow.add_node("decomposer", task_decomposer_node)
workflow.add_node("coder", coder_agent_node)
workflow.add_node("tester", tester_agent_node)
workflow.add_node("reviser", reviser_agent_node)
workflow.add_node("integrator", integrator_agent_node)
workflow.add_edge(START, "planner")
workflow.add_conditional_edges("planner", should_start_coding, {"proceed_to_architect": "architect", "end_workflow": END})
workflow.add_edge("architect", "decomposer")
workflow.add_edge("decomposer", "coder")
workflow.add_edge("coder", "tester")
workflow.add_conditional_edges("tester", should_revise_or_integrate, {"needs_revision": "reviser", "integrate_as_is": "integrator"})
workflow.add_edge("reviser", "tester")
workflow.add_conditional_edges("integrator", should_continue_coding, {"continue_coding": "coder", "end_workflow": END})
app = workflow.compile()

# ==============================================================================
# Main execution block
# ==============================================================================
if __name__ == "__main__":
    print("📊 Attempting to generate and display LangGraph architecture...")
    ### --- FIX --- ### Added image display with a graceful fallback to text.
    try:
        # Try to generate and display the PNG image
        png_data = app.get_graph().draw_mermaid_png()
        display(Image(png_data))
        print("   ✅ Graph image displayed successfully.")
    except Exception as e:
        # If it fails, print the Mermaid text definition instead
        print(f"   ⚠️  Could not generate graph image ({e}). Falling back to text definition.")
        mermaid_text = app.get_graph().draw_mermaid()
        print("\n" + "="*60)
        print("📋 Mermaid Graph Definition (For Manual Visualization)")
        print("="*60)
        print(mermaid_text)
        print("="*60)
        print("➡️ To see the visual, copy the text above and paste it into https://mermaid.live")
    print("="*60 + "\n")

    srs_document_path = os.path.join("inputs", "SRS_Blog_Submission_System.pdf")
    diagram_paths = { "sequence_diagram": os.path.join("inputs", "Editor _ Mermaid Chart-2025-06-19-074902.mmd") }
    initial_state = { "srs_path": srs_document_path, "diagram_paths": diagram_paths }
    
    ### --- FIX --- ### Added a config dictionary to increase the recursion limit.
    config = {"recursion_limit": 100}

    # Invoke the graph with the increased limit
    final_state = app.invoke(initial_state, config=config)

    print("\n\n" + "="*50)
    print("✅✅✅ Agentic SDLC Workflow Complete! ✅✅✅")
    print("="*50)
    print_directory_tree(final_state.get("project_root", "generated_project"))