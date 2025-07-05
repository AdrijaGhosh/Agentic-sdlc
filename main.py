# ==============================================================================
# Step 1: Imports and Environment Setup
# ==============================================================================
import os
import json
import shutil
from typing import List, TypedDict

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field

# ==============================================================================
# Helper function to replace the 'tree' command
# ==============================================================================
def print_directory_tree(root_dir):
    """A Python function to print a directory tree."""
    print(f"🌳 Final Project Structure for: {root_dir}")
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}├── {os.path.basename(root)}/')
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{sub_indent}└── {f}')

# ==============================================================================
# Agent 1: Planner Agent
# ==============================================================================
def run_complex_planner_agent(srs_path: str, diagram_paths_dict: dict):
    print("🚀 Initializing Complex Planner Agent...")

    class PlannerState(TypedDict):
        srs_content: str
        diagrams_input_string: str
        analysis_result: str
        final_output: str

    def read_text_from_file(file_path):
        try:
            with open(file_path, 'r') as f: return f.read()
        except FileNotFoundError: return None

    def extract_text_from_pdf(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            return "\n".join(page.page_content for page in loader.load())
        except Exception: return None

    # ... The planner agent nodes are identical ...
    def load_inputs_node(state: PlannerState):
        srs_content = extract_text_from_pdf(srs_path)
        if not srs_content:
            return {"final_output": f"❌ CRITICAL ERROR: SRS document at '{srs_path}' could not be read."}
        diagram_inputs = []
        for diagram_type, path in diagram_paths_dict.items():
            if path and (content := read_text_from_file(path)):
                diagram_inputs.append(f"### {diagram_type.replace('_',' ').title()}\n```mermaid\n{content}\n```")
        diagrams_str = "\n\n".join(diagram_inputs) if diagram_inputs else "No diagrams provided."
        return {"srs_content": srs_content, "diagrams_input_string": diagrams_str}

    def analyze_sufficiency_node(state: PlannerState):
        if state.get("final_output"): return {}
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            "You are a project manager. Determine if the provided SRS and diagrams are sufficient to create a detailed coding plan. Vague documents are INSUFFICIENT. Respond with ONLY 'SUFFICIENT' or 'INSUFFICIENT' and a brief justification.\n\n**Inputs:**\n{diagrams_input_string}\n---\n{srs_content}"
        )
        result = (prompt | llm | StrOutputParser()).invoke(state)
        return {"analysis_result": result}

    def generate_plan_node(state: PlannerState):
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            "You are a master Planner Agent. Create a detailed, step-by-step coding plan from the SRS and diagrams. Group tasks by component. For each task, specify the file path and its purpose.\n\n**Inputs:**\n{diagrams_input_string}\n---\n{srs_content}"
        )
        plan = (prompt | llm | StrOutputParser()).invoke(state)
        return {"final_output": plan}

    def handle_insufficient_data_node(state: PlannerState):
        error_message = f"## Planning Halted\n**Reason:** Insufficient documents.\n**Justification:**\n> {state['analysis_result']}"
        return {"final_output": error_message}

    def decide_next_step(state: PlannerState):
        if "final_output" in state and "CRITICAL ERROR" in state["final_output"]: return END
        if "SUFFICIENT" in state["analysis_result"].upper(): return "generate_plan"
        return "handle_insufficient_data"

    graph = StateGraph(PlannerState)
    graph.add_node("load_inputs", load_inputs_node)
    graph.add_node("analyze_sufficiency", analyze_sufficiency_node)
    graph.add_node("generate_plan", generate_plan_node)
    graph.add_node("handle_insufficient_data", handle_insufficient_data_node)
    graph.add_edge(START, "load_inputs")
    graph.add_edge("load_inputs", "analyze_sufficiency")
    graph.add_conditional_edges("analyze_sufficiency", decide_next_step, {"generate_plan": "generate_plan", "handle_insufficient_data": "handle_insufficient_data"})
    graph.add_edge("generate_plan", END)
    graph.add_edge("handle_insufficient_data", END)
    app = graph.compile()

    # ==================================================================
    # VISUALIZATION BLOCK
    # ==================================================================
    print("📊 Generating Planner Agent Graph Definition...")
    try:
        # Generate the graph definition as text instead of a PNG image
        mermaid_text = app.get_graph().draw_mermaid()
        print("\n" + "="*50)
        print("📋 Mermaid Graph Definition (For Manual Visualization)")
        print("="*50)
        print(mermaid_text)
        print("="*50)
        print("➡️ To see the visual, copy the text above and paste it into a Mermaid editor like https://mermaid.live")
        print("="*50 + "\n")
    except Exception as e:
        print(f"  - ⚠️ Could not generate graph definition. Error: {e}")
    # ==================================================================

    print("\n🤖 Invoking the LangGraph Planner Agent...")
    final_state = app.invoke({})
    print("✅ Planner Agent Finished.")
    print(final_state['final_output'])
    return final_state

# ==============================================================================
# Agent 2: Architect Agent 
# ==============================================================================
class FilePathList(BaseModel):
    """A model to hold a list of file paths."""
    paths: List[str] = Field(description="A complete list of all file and directory paths that need to be created for the project.")

def create_project_structure_from_plan(plan_text: str, root_dir: str = "generated_project"):
    print(f"\n🚀 Initializing Architect Agent to build project in './{root_dir}'...")

    llm = ChatOpenAI(model="gpt-4o", temperature=0) # 4.1 nano
    json_parser = JsonOutputParser(pydantic_object=FilePathList)

    parsing_prompt = ChatPromptTemplate.from_template(
        "You are an expert file system parser. Read the provided software plan and extract a complete list of all file paths to be created. Return the list as a flat JSON array of strings.\n\n**Plan to Parse:**\n---\n{coding_plan}\n---\n\n{format_instructions}"
    )
    parsing_chain = parsing_prompt | llm | json_parser

    print("\n🔵 Phase 1: Parsing the plan to extract file structure...")
    try:
        parsed_output = parsing_chain.invoke({
            "coding_plan": plan_text,
            "format_instructions": json_parser.get_format_instructions()
        })
        file_list = parsed_output['paths']
        print(f"  ✅ Plan parsed successfully. Found {len(file_list)} files/directories to create.")
    except Exception as e:
        print(f"❌ ERROR: Failed to parse the plan. Error: {e}")
        return

    print("\n🔵 Phase 2: Building the project structure...")
    if os.path.exists(root_dir):
        print(f"  - Warning: Root directory '{root_dir}' already exists. Clearing it for a fresh build.")
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)

    for file_path in file_list:
        if ".." in file_path:
            print(f"  - Skipping potentially malicious path: {file_path}")
            continue
        full_path = os.path.join(root_dir, file_path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                pass # Create empty file
        except Exception as e:
            print(f"  ❌ Failed to create '{full_path}'. Error: {e}")

    print("\n" + "="*50)
    print_directory_tree(root_dir)

# ==============================================================================
# Main execution block (No changes needed here)
# ==============================================================================
if __name__ == "__main__":
    srs_document_path = os.path.join("inputs", "SRS_Blog_Submission_System.pdf")
    diagram_paths = {
        "sequence_diagram": os.path.join("inputs", "Editor _ Mermaid Chart-2025-06-19-074902.mmd"),
        "class_diagram": None,
        "entity_relationship_diagram": None,
        "flowchart": None
    }

    final_planner_state = run_complex_planner_agent(
        srs_path=srs_document_path,
        diagram_paths_dict=diagram_paths
    )

    planning_successful = False
    if final_planner_state and "SUFFICIENT" in final_planner_state.get("analysis_result", ""):
        coding_plan = final_planner_state.get('final_output')
        planning_successful = True
        print("\n\n✅ Planning successful. The generated plan is now available for the Architect Agent.")
    else:
        print("\n\n❌ Planning failed or was halted. The Architect Agent will not run.")

    if planning_successful and coding_plan:
        create_project_structure_from_plan(plan_text=coding_plan)
    else:
        print("\nSkipping Architect Agent because the planning phase did not produce a valid plan.")