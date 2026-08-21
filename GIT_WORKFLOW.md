# 🚀 GitHub Team Collaboration Guide & Repo Setup

This guide provides step-by-step instructions to connect your local folder to GitHub and best practices for working smoothly with your teammates.

---

## Part 1: Prerequisites & Initial Connection

### 1. Install Git on Linux
Since `git` is required, open your Linux terminal and install git:
```bash
sudo apt update && sudo apt install -y git
```

### 2. Connect Your Project to GitHub
Run the following commands in your terminal:

```bash
# Navigate to your project directory
cd ~/FreeTraining-OVSS

# 1. Initialize git repository
git init

# 2. Check status (make sure .gitignore is hiding venv and data files)
git status

# 3. Add remote repository URL
git remote add origin https://github.com/FasterThanAi/FreeTraining-OVSS.git

# 4. Stage and commit your initial project baseline
git add .
git commit -m "feat: initial commit of project baseline"

# 5. Set default branch to main and push to GitHub
git branch -M main
git push -u origin main
```

> **Note**: If `https://github.com/FasterThanAi/FreeTraining-OVSS.git` already contains files (like a remote README), run `git pull origin main --allow-unrelated-histories` or `git rebase origin/main` before pushing.

---

## Part 2: Inviting Teammates to GitHub

1. Go to your repository on GitHub: `https://github.com/FasterThanAi/FreeTraining-OVSS`
2. Click **Settings** -> **Collaborators** (or **Manage Access**).
3. Click **Add people** and enter your teammates' GitHub usernames or email addresses.
4. Once they accept the invitation, they can clone the repository to their machines using:
   ```bash
   git clone https://github.com/FasterThanAi/FreeTraining-OVSS.git
   ```

---

## Part 3: Best Workflow for Team Collaboration (Feature Branch Workflow)

Working directly on `main` causes code overwrites, broken builds, and merge conflicts. Follow the **Feature Branch Workflow**:

```
 main branch:   ───●─────────────────────────● (Stable code only)
                    \                       / (Merged via Pull Request)
 feature branch:     └───●───────●───────●─── (Developer working area)
```

### Daily 6-Step Teammate Workflow

#### Step 1: Pull Latest Main
Always make sure you start with the newest code from your teammates:
```bash
git checkout main
git pull origin main
```

#### Step 2: Create a Feature Branch
Create a branch named after the feature or fix you are working on:
```bash
# Example: feature/sam-segmentation or fix/dataset-loader
git checkout -b feature/your-feature-name
```

#### Step 3: Make Changes & Commit
Make your code changes, then commit with clear, descriptive messages:
```bash
git status
git add <files-you-changed>
git commit -m "feat: add SAM model feature extraction pipeline"
```

#### Step 4: Push Branch to GitHub
Push your local feature branch to GitHub so your teammates can see it:
```bash
git push -u origin feature/your-feature-name
```

#### Step 5: Open a Pull Request (PR) on GitHub
1. Open `https://github.com/FasterThanAi/FreeTraining-OVSS` in your web browser.
2. You will see a button **"Compare & pull request"**. Click it.
3. Add a short title and description explaining what you changed.
4. Assign a teammate under **Reviewers**.
5. Click **Create pull request**.

#### Step 6: Code Review & Merge
1. Your teammate reviews the code on GitHub, leaves feedback or approves it.
2. Click **"Merge pull request"** -> **"Confirm merge"**.
3. Delete the feature branch on GitHub.
4. Back on your computer:
   ```bash
   git checkout main
   git pull origin main
   ```

---

## Part 4: How to Avoid & Fix Merge Conflicts

### Rule #1: Never edit the same file on the same lines simultaneously without communicating.

### How to resolve a conflict if it happens:
If your teammate merged code into `main` while you were working:

1. Fetch and rebase your feature branch against `main`:
   ```bash
   git checkout feature/your-feature-name
   git fetch origin
   git rebase origin/main
   ```
2. Git will pause if there are conflicting lines.
3. Open the conflicted files in VS Code / your text editor. Look for:
   ```text
   <<<<<<< HEAD
   Your code
   =======
   Teammate's code
   >>>>>>> main
   ```
4. Choose the correct code, remove the marker lines (`<<<<<<<`, `=======`, `>>>>>>>`), and save the file.
5. Stage the resolved files and continue:
   ```bash
   git add <resolved-file>
   git rebase --continue
   ```
6. Push the updated branch to GitHub:
   ```bash
   git push --force-with-lease origin feature/your-feature-name
   ```

---

## Part 5: Summary Checklist for Great Teamwork

| Goal | Command / Action |
| --- | --- |
| **Start of day** | `git checkout main && git pull origin main` |
| **New feature** | `git checkout -b feature/feature-name` |
| **Save progress** | `git add . && git commit -m "descriptive message"` |
| **Share work** | `git push -u origin feature/feature-name` |
| **Integrate code** | Open Pull Request on GitHub & ask teammate to review |
| **Keep `.gitignore` clean** | Never commit virtual environments, raw datasets, or model weights! |
