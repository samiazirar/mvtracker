# Docker Registry CI/CD Setup

This project is configured to automatically build and push Docker images to the GitLab Container Registry.

## Registry
```
registry.gitlab.uni-bonn.de:5050/rpl/public_registry
```

## Image Types

### 🔒 STABLE Build
**Tags:** `stable`, `YYYY-MM-stable` (e.g., `2025-11-stable`)

- **Everything pre-installed** in the Docker image
- All `third_party` dependencies baked in at `/workspace/third_party_stable/`
- All Python packages pre-installed (PyTorch, SAM2, HaMeR, etc.)
- **No git cloning at runtime** - completely self-contained
- Best for: Production, reproducible environments, offline use

### 🔄 LATEST Build  
**Tag:** `latest`

- **Minimal base image**
- Pulls all dependencies fresh at container start
- Uses mounted workspace `third_party/` from host
- Installs Python packages at runtime
- Best for: Development, getting latest updates

## Directory Structure

```
.devcontainer/
├── stable/
│   ├── Dockerfile              # Full build with all deps
│   ├── devcontainer.json       # VS Code config for stable
│   └── post-create-stable.sh   # Minimal setup (symlinks only)
├── latest/
│   ├── Dockerfile              # Minimal base image
│   ├── devcontainer.json       # VS Code config for latest
│   └── post-create-latest.sh   # Full setup (clones repos, installs packages)
├── Dockerfile                  # Original (kept for compatibility)
├── devcontainer.json          # Original (kept for compatibility)
└── post-create.sh             # Original (kept for compatibility)
```

## Using in VS Code

### Open Stable Build
1. Open Command Palette (Ctrl+Shift+P)
2. Select "Dev Containers: Open Folder in Container..."
3. Choose `.devcontainer/stable/devcontainer.json`

### Open Latest Build
1. Open Command Palette (Ctrl+Shift+P)
2. Select "Dev Containers: Open Folder in Container..."
3. Choose `.devcontainer/latest/devcontainer.json`

## Setup Instructions

### 1. Configure CI/CD Variables

Go to your GitLab project → **Settings** → **CI/CD** → **Variables** and add:

| Variable | Value | Protected | Masked |
|----------|-------|-----------|--------|
| `CI_REGISTRY_USER` | Your GitLab username or deploy token username | ✓ | ✗ |
| `CI_REGISTRY_PASSWORD` | Your GitLab password or deploy token | ✓ | ✓ |

> **Tip**: Use a [Deploy Token](https://docs.gitlab.com/ee/user/project/deploy_tokens/) instead of personal credentials for better security.

### 2. Set Up Monthly Schedule

Go to your GitLab project → **Build** → **Pipeline schedules** → **New schedule**:

- **Description**: `Monthly Docker Build`
- **Interval Pattern**: `0 2 1 * *` (runs at 2:00 AM on the 1st of each month)
- **Cron Timezone**: Select your timezone
- **Target branch**: `main`
- **Activated**: ✓

### 3. Manual Trigger

You can also manually trigger a build:
- Go to **Build** → **Pipelines** → **Run pipeline**
- Select `main` branch
- Click **Run pipeline**

## Local Build & Push

### Build STABLE image (all deps baked in)
```bash
# Login to registry
docker login registry.gitlab.uni-bonn.de:5050

# Build with tags
DATE_TAG=$(date +%Y-%m)-stable
docker build -t registry.gitlab.uni-bonn.de:5050/rpl/public_registry:stable \
             -t registry.gitlab.uni-bonn.de:5050/rpl/public_registry:$DATE_TAG \
             -f .devcontainer/stable/Dockerfile .

# Push both tags
docker push registry.gitlab.uni-bonn.de:5050/rpl/public_registry:stable
docker push registry.gitlab.uni-bonn.de:5050/rpl/public_registry:$DATE_TAG
```

### Build LATEST image (minimal)
```bash
docker build -t registry.gitlab.uni-bonn.de:5050/rpl/public_registry:latest \
             -f .devcontainer/latest/Dockerfile .

docker push registry.gitlab.uni-bonn.de:5050/rpl/public_registry:latest
```

## Pulling the Image

```bash
# Pull stable (recommended for production)
docker pull registry.gitlab.uni-bonn.de:5050/rpl/public_registry:stable

# Pull specific month's stable build
docker pull registry.gitlab.uni-bonn.de:5050/rpl/public_registry:2025-11-stable

# Pull latest (development)
docker pull registry.gitlab.uni-bonn.de:5050/rpl/public_registry:latest
```

## Environment Variable

Inside the container, check which build you're running:
```bash
echo $MVTRACKER_BUILD  # outputs: "stable" or "latest"
```
