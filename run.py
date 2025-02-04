import sys
import asyncio
import subprocess
import streamlit as st


from dotenv import load_dotenv

from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.llms.ollama import Ollama

from workflow import TourPlannerWorkflow


async def run_trip_planner(query):
    load_dotenv()
    llm = Ollama(model="llama3.1:8b", request_timeout=60000)
    workflow = TourPlannerWorkflow(llm=llm, verbose=False, timeout=80000.0)
    result = await workflow.run(query=query)

    return result

def main():
    st.title("Trip Planner Assistant 🧳")
    query = st.text_input("Enter your trip request:", "Plan a trip to Naples from Catania next month")
    if st.button("Plan My Trip"):
        with st.spinner("Planning your trip..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_trip_planner(query))
            st.success("Trip Plan Generated!")
            st.write(result)            
if __name__ == "__main__":
    main()