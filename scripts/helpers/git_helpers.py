import os


class GitHelpers:
	"""docstring for GitHelpers."""
	
	@classmethod
	def reset_local_changes(cls, repo_path):
		""" Resets local changes to the repo."""
		os.chdir(repo_path)
		# os.system("git reset --hard HEAD")
		# os.system("git clean -f -d")
		os.system("git stash")
		os.system("git stash drop")