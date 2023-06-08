import neptune
from pathlib import Path

""" Real project Windows environment variables setting:
setx NEPTUNE_API_TOKEN "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ=="
setx NEPTUNE_PROJECT "commander.pho/PhoDibaLongShort2023"


export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ=="
export NEPTUNE_PROJECT="commander.pho/PhoDibaLongShort2023"
"""

project = neptune.init_project(project="commander.pho/PhoDibaTestProject", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ==")


run = neptune.init_run(project='commander.pho/PhoDibaTestProject')
run["run_platform/computer"] = "Apogee"

test_file_path = Path(r'W:\Data\global_batch_result_new.pkl')
run["artifact1"].track_files(f"file://{test_file_path}") # Proper way to track a Windows Path.

run.stop()

project.stop()


# project = neptune.init_project(project="commander.pho/PhoDibaLongShort2023", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ==")
# project["general/brief"] = URL_TO_PROJECT_BRIEF
# project["general/data_analysis"].upload("data_analysis.ipynb")
# project["dataset/v0.1"].track_files("s3://datasets/images")
# project["dataset/latest"] = project["dataset/v0.1"].fetch()



# ## Start a Run:
# run = neptune.init_run(
#     capture_hardware_metrics=True,
#     capture_stderr=True,
#     capture_stdout=True,
# )


# run["namespace/field"] = "some metadata"


# run.stop()