# CI/CD With GitHub Actions

This project separates continuous integration from container publication.
The separation prevents unverified code from reaching the container registry.

## Delivery Flow

```text
Developer push or pull request
        |
        v
CI: lint + tests + Compose validation
        |
        v
CI: build API and frontend images without publishing
        |
        v
Merge or push to main
        |
        v
Successful CI workflow
        |
        v
CD: rebuild the verified commit and publish it to GHCR
```

## Continuous Integration

`.github/workflows/ci.yml` runs on every branch push, pull request to `main`,
and manual dispatch.

### Python quality and configuration

The first job:

1. checks out the repository without persisting Git credentials;
2. installs Python 3.11 and restores the pip download cache;
3. installs application and test dependencies;
4. runs Ruff across `src` and `tests`;
5. runs the complete pytest suite;
6. validates the rendered Docker Compose configuration.

The job has read-only repository permissions. A failed command stops the job,
so Docker images are not built from code that failed a quality gate.

### Container build verification

The second job depends on the quality job and uses a matrix to build:

- the FastAPI image from `Dockerfile`;
- the Nginx frontend image from `frontend/Dockerfile`.

The builds use BuildKit's GitHub Actions cache, but `push: false` ensures that
pull requests and feature branches cannot publish images.

Concurrency cancellation stops obsolete CI runs when newer commits arrive on
the same branch.

## Continuous Delivery

`.github/workflows/cd.yml` starts only when the `CI` workflow succeeds on
`main`. It can also be started manually from `main`.

The workflow checks out the exact SHA verified by CI, logs in to GitHub
Container Registry using the short-lived `GITHUB_TOKEN`, and publishes:

```text
ghcr.io/<owner>/<repository>-api:latest
ghcr.io/<owner>/<repository>-api:<full-commit-sha>
ghcr.io/<owner>/<repository>-frontend:latest
ghcr.io/<owner>/<repository>-frontend:<full-commit-sha>
```

The full commit SHA is an immutable deployment reference. `latest` is a
convenience pointer and should not be used for controlled production rollback.

Published images include OCI metadata, an SBOM, and build provenance.

## Required GitHub Settings

In the repository settings:

1. Open **Actions > General**.
2. Ensure workflows can read repository contents.
3. Allow the `GITHUB_TOKEN` to write packages for the CD workflow.
4. Protect `main` under **Branches** or **Rulesets**.
5. Require the CI status checks before merge:
   - `Python quality and configuration`
   - `Build api image`
   - `Build frontend image`
6. Require pull requests and prevent direct pushes to `main`.

No personal registry password is required. GitHub creates `GITHUB_TOKEN`
automatically for each workflow run, and the workflow grants package-write
permission only to CD.

## Dependency Maintenance

`.github/dependabot.yml` checks GitHub Actions weekly and Python/Docker
dependencies monthly. Dependency pull requests still pass through the same CI
gates before they can be merged.

Workflow actions are pinned to immutable commit SHAs. The comments retain their
major versions for readability, while Dependabot proposes reviewed SHA updates.

## What This Does Not Deploy

CD currently publishes versioned container artifacts. It does not yet update a
staging or production server. The deployment step will consume the immutable
SHA tags after infrastructure, secrets management, backups, and environment
approval rules are defined.

## Learning Resources

- [GitHub Actions workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
- [GitHub workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [GitHub protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [Docker build GitHub Actions](https://docs.docker.com/build/ci/github-actions/)
- [Docker build cache](https://docs.docker.com/build/ci/github-actions/cache/)
