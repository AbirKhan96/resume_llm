import pandas as pd
from pathlib import Path
from docx import Document


import transformers
import torch
import os
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


# Load the resume texts from the documents
resume_texts = {}
domains = [fold for fold in Path('/home/aiml/hemanth/jackmack/llms/ner/Job_description').iterdir()]

for row in domains:
    document = Document(row)
    doc_text = '\n'.join([paragraph.text for paragraph in document.paragraphs])
    resume_texts[str(row)] = doc_text

# Initialize entity prefixes dictionary
entity_prefixes = {
    "Degrees": [],
    "Designation": [],
    "Skills": [],
    "Years_of_Experience": []
}

# Iterate over resume texts and collect the required information
for path, text in resume_texts.items():
    # Skills Section
    messages = [
        {"role": "system", "content": "List only the skills that are tangible and can be directly applied. Avoid listing activities, roles, or abstract concepts."},
        {"role": "user", "content": f"{text}"},
        {"role": "user", "content": "Please provide a list of tangible skills only, excluding roles, activities, and responsibilities."}
    ]
    
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        temperature=0.3,
        top_k=50,
        top_p=0.9
    )
    
    skills = outputs[0]["generated_text"][-1]['content']
    entity_prefixes['Skills'].append(skills)

    # Define prompts for each type of information
    prompts = {
        "Years_of_Experience": "Can you tell me how many years of experience is required as per text",
        "Designation": "What is the job title or designation required in the job description?",
        "Degrees": "List each specific degree mentioned in the text separately, ensuring each is in the format 'Bachelor's degree in [Field]' or 'Master's degree in [Field]'."
    }
    
    # Collect responses for each entity type
    for key, prompt in prompts.items():
        messages = [
            {"role": "system", "content": "Provide direct answers without any explanation. Format the response as distinct entries for each degree, listing each degree on a new line."},
            {"role": "user", "content": f"{text}"},
            {"role": "user", "content": prompt}
        ]
        
        output = pipeline(
            messages,
            max_new_tokens=150,
            temperature=0.1,
            top_k=50,
            top_p=0.9
        )
        
        response = output[0]["generated_text"][-1]['content']
        entity_prefixes[key].append(response)
    # break
# Create a DataFrame from the entity_prefixes dictionary
df = pd.DataFrame(entity_prefixes)

# Optionally, save the DataFrame to a CSV file
# df.to_csv('resume_analysis.csv', index=False)

# Display the DataFrame
print(df)
