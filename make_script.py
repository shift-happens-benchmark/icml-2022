import json

with open("./tasks_config.json", "r") as f:
    tasks_config = json.load(f)

scriptstring = """
from datetime import datetime
import shifthappens.benchmark
import shifthappens.utils
"""

print(tasks_config["tasks"])
for task in tasks_config["tasks"]:
    scriptstring += tasks_config["import_lines"][task]
    scriptstring += "\n"

scriptstring += tasks_config["import_lines"][tasks_config["model"]]

out_file_location = tasks_config["out_file_location"]
relative_data_folder = tasks_config["relative_data_folder"]

scriptstring += f"""
tasks = shifthappens.benchmark.get_task_registrations()
model = {tasks_config['model']}()
results = shifthappens.benchmark.evaluate_model(
    model, "{relative_data_folder}"
)
results_string = shifthappens.utils.serialize_model_results(results)
out_file_location = "{out_file_location}"
with open(out_file_location, 'w') as outfile:
    outfile.write(results_string)
"""
with open("./run_tasks.py", "w") as run_script_file:
    run_script_file.write(scriptstring)
