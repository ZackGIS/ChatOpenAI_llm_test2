import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
import openai
from pydantic import BaseModel, Field
from enum import Enum


api_key = "openai api key goes here"
openai.api_key = api_key

# input and output folder paths
inputFolderPath = r'C:\InternCSVs\GIS_Web_Services\output'
outputFolderPath = r'C:\InternCSVs\GIS_Web_Services\output\output_after_ChatOpenAI'
os.makedirs(outputFolderPath, exist_ok=True)

#redefined tagging_prompt. Tried to be more descriptive
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the following information from the passage:
1. Read the description column of each entry in the CSV file. If there is a "No description" in the description column of the CSV entry, use the tags column as reference.
2. Whatever information you have, whether from the description column or tags column, how related is this to energy on a scale from 1 to 10?
3. Select a tag from the provided list: wells, pipelines, infrastructure, imagery, weather, environmental, geology, seismic, geomatics, renewables, emissions, basemaps, bathymetry.

Passage:
{input}
"""
)


#Not certain if defining the tags as a separate class is really necessary but the llm seems to like it
class EnergyTag(str, Enum):
    wells = "wells"
    pipelines = "pipelines"
    infrastructure = "infrastructure"
    imagery = "imagery"
    weather = "weather"
    environmental = "environmental"
    geology = "geology"
    seismic = "seismic"
    geomatics = "geomatics"
    renewables = "renewables"
    emissions = "emissions"
    basemaps = "basemaps"
    bathymetry = "bathymetry"

#slightly modified Classification BaseModel. this version is not using it at the moment since with_structured_output is breaking things. Because of this, output is all over the place.
# will simplify the prompt tomorrow.
class Classification(BaseModel):
    energy: str = Field(description="What aspect of energy is this related to?")
    energy_related: int = Field(description="How related is this to energy from 1 to 10")
    tag: EnergyTag = Field(description="Classification tag")

# Create the language model and agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=None, timeout=None)
agent_csv_path = os.path.join(inputFolderPath, 'export-1.csv')   # need to get rid of this. Already have a CSV file path

# declaring an agent with create_csv_agent called on the llm. invoke() seems to work when called on the agent
agent = create_csv_agent(llm, agent_csv_path, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# tagging_chain = tagging_prompt | llm    #declaring tagging chain, no longer using this for this script

def tagging_function(description, tags):
    if description == "No description":    # we have a bunch of "No description" cells so read the tags instead if one is encountered
        text = tags
    else:
        text = description
    prompt_input = tagging_prompt.format(input=text)
    result = agent.invoke(prompt_input)  # here i'm not calling invoke() on a tagging_chain. Using the agent instead.
    return {
        'energy': result['energy'],
        'energy_related': result['energy_related'],
        'tag': result['tag']
    }


for filename in os.listdir(inputFolderPath):
    if filename.endswith('.csv'):
        csvFilePath = os.path.join(inputFolderPath, filename)
        data = pd.read_csv(csvFilePath)

        # creating the extra columns in the df to house the output from the llm
        data['energy'] = None
        data['energy_related'] = None
        data['tag'] = None

        # since im only focusing on the description or tags columns, these two lines aren't needed.
        # data['title_str'] = data['title'].astype(str).str.replace('_', ' ')
        # data['title_len'] = data['title'].apply(lambda x: len(str(x).split()))

        for index, row in data.iterrows():
            description_text = row['description']
            tags_text = row['tags']

            tagging_result = tagging_function(description_text, tags_text)
            if tagging_result:
                data.at[index, 'energy'] = tagging_result['energy']
                data.at[index, 'energy_related'] = tagging_result['energy_related']
                data.at[index, 'tag'] = tagging_result['tag']

        filtered_data = data[(data['energy_related'] >= 7) & (data['energy_related'] <= 10)]  # I'm selecting entries with 7 rating ot

        outputFilePath = os.path.join(outputFolderPath, filename[:-4] + "_filtered.csv")
        filtered_data.to_csv(outputFilePath, index=False, columns=['title', 'url', 'type', 'tags', 'description', 'thumbnailurl'])
