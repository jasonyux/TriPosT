# Description: Utility functions for runners
from retry import retry

# TODO: delete this function
# @retry(Exception, tries=100, delay=2)
def chain_run_wrapper(chain, prompt_data:dict):
    return chain(prompt_data)