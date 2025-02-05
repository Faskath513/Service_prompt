# scripts/generate_outputs.py
from models.huggingface_model import generate_service_description

def generate_descriptions():
    with open('data/example_inputs.txt', 'r') as file:
        services = [line.strip() for line in file.readlines()]

    for service in services:
        description = generate_service_description(service)
        print(f"Service: {service}\nDescription: {description}\n")