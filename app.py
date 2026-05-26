from crewai import Agent, Task, Crew, Process, LLM
import os

# Google API Key
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.4,
)

# USER INPUT

event_name = "collage Tech Fest"
budget = "$10000"
guests = 200


# AGENT 1 - EVENT PLANNER

planner_agent = Agent(
    role="Event Planner",
    goal="Create a event plan",
    backstory="Expert event organizer",
    verbose=True,
    llm=llm
)


# AGENT 2 - FOOD MANAGER

food_agent = Agent(
    role="Food Manager",
    goal="Plan food for guests",
    backstory="Expert in food management",
    verbose=True,
    llm=llm
)


# AGENT 3 - BUDGET MANAGER

budget_agent = Agent(
    role="Budget Manager",
    goal="Manage event budget",
    backstory="Expert in finance planning",
    verbose=True,
    llm=llm
)


# TASK 1 - EVENT PLAN

planning_task = Task(
    description=f"""
    plan event:
    {event_name}

    guests:
    {guests}

    give:
    - Theme
    - Venue
    - Activities
    """,

    expected_output="Simple event plan.",
    agent=planner_agent
)


# TASK 2 - FOOD PLAN

food_task = Task(
    description=f"""
    Suggest food menu for:
    {guests} guests

    give:
    - Snacks
    - Lunch
    - Drinks
    """,

    expected_output="Food plan.",
    agent=food_agent
)


# TASK 3 - BUDGET PLAN

budget_task = Task(
    description=f"""
    Create budget breakdown for:
    {budget}

    give:
    - Food cost
    - Venue cost
    - Decoration cost
    """,

    expected_output="Budget plan.",
    agent=budget_agent
)


# CREW PIPELINE

crew = Crew(
    agents=[
        planner_agent,
        food_agent,
        budget_agent
    ],

    tasks=[
        planning_task,
        food_task,
        budget_task
    ],

    process=Process.sequential,
    verbose=True
)


# RUN PIPELINE

print("\n Running Event Planning Crew...")

results = crew.kickoff()

print("\n==================")
print("Event Planning Results:")
print("\n==================")

print(results)