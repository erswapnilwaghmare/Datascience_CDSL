## Github_Roadmap

| Category                | Command                                 | Description                                                   | Tips                                                                 |
|-------------------------|-----------------------------------------|---------------------------------------------------------------|----------------------------------------------------------------------|
| **Setup**               | `git config --global user.name "[name]"` | Set a name that is identifiable for credit when review version history. | Use your GitHub username or your real name.                         |
|                         | `git config --global user.email "[email address]"` | Set an email address that will be associated with each commit. | Use the email associated with your GitHub account.                  |
| **Initialize**          | `git init`                              | Initialize a local Git repository in the current directory.   | Run this in the root of your project directory.                     |
| **Clone**               | `git clone [url]`                       | Create a local copy of a remote repository.                    | Replace `[url]` with the GitHub repo URL you want to clone.         |
| **Staging**             | `git add [file]`                        | Add a file to the staging area.                                | Replace `[file]` with the file name to stage specific files.        |
|                         | `git add .`                             | Add all new and changed files to the staging area.             | Be cautious, as this adds all files to the commit.                  |
| **Committing**          | `git commit -m "[commit message]"`      | Commit changes to head (but not yet to the remote repository). | Always include a brief, descriptive commit message.                 |
| **Pushing**             | `git push origin [branch]`              | Send changes to the master branch of your remote repository.   | Replace `[branch]` with the branch you want to push.                |
| **Pulling**             | `git pull`                              | Fetch and merge changes on the remote server to your working directory. | Always `pull` before you start working to ensure you have the latest changes. |
| **Branching**           | `git branch`                            | List all the branches in your repo, and also tells you what branch you're currently in. | Use branching to work on different parts of a project simultaneously. |
|                         | `git branch [branch name]`              | Create a new branch at the current commit.                     | Replace `[branch name]` with your new branch name.                  |
|                         | `git checkout [branch name]`            | Switch to another branch and check it out into your working directory. | Be sure to commit current changes before switching branches.        |
|                         | `git merge [branch]`                    | Merge another branch into your active branch.                   | Make sure you understand the changes before merging.                |
| **Status/Log**          | `git status`                            | Check the status of changes as untracked, modified, or staged. | Use this often to check the status before committing.               |
|                         | `git log`                               | View changes.                                                   | Use to see commit history.                                          |
| **Diff**                | `git diff`                              | View differences between your working directory and the staged changes. | Use this before committing to see what changes are being committed. |
| **Remote Repositories** | `git remote add origin [server]`        | Connect your local repository to a remote server.              | Replace `[server]` with the GitHub repository URL.                  |
|                         | `git push -u origin [branch]`           | Push your changes for the first time and set the remote as upstream. | Use `-u` to set the upstream (tracking reference) on first push.    |
| **Undo Changes**        | `git checkout -- [file]`                | Undo your changes to a file.                                    | Be cautious, as this will erase changes.                            |
|                         | `git fetch origin`                      | Fetches the branches and their respective commits from the remote repository. | Does not merge changes automatically.                              |
| **Tags**                | `git tag [tag name]`                    | Mark specific points in history as important.                   | Typically used to mark release points (v1.0, v2.0, etc.).           |

Tips for Beginners:
- Commit Often: Commits should be made whenever you reach a significant point in your development where if you needed to, you could revert back to this point.
- Pull Often: Especially in collaborative settings, you want to ensure you're working with the most up-to-date code.
- Branch Wisely: Create branches for features, bug fixes, or experiments. It's easier to manage and integrate.
- Understand `.gitignore`: Use this file to exclude files from being tracked by Git.
- Learn Markdown: `README.md`, Issues, and Wikis in GitHub support Markdown for formatting. Learn the syntax to improve readability.
- Leverage GitHub Features: Use GitHub features
