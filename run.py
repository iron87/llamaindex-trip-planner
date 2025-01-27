import sys
import asyncio
import subprocess

from dotenv import load_dotenv

from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.llms.ollama import Ollama

from workflow import TourPlannerWorkflow


async def main():
    load_dotenv()
    llm = Ollama(model="llama3.1:8b", request_timeout=60000)
    workflow = TourPlannerWorkflow(llm=llm, verbose=False, timeout=80000.0)
    # draw_all_possible_flows(workflow, filename="workflow.html")
    query = sys.argv[1]
    result = await workflow.run(query=query)
    if ".pdf" in result:
        subprocess.run(["open", result])


if __name__ == "__main__":
    asyncio.run(main())
