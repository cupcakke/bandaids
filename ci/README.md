CI workflow

The workflow definition lives here as github-workflow-ci.yml because the
automation account that produced this branch does not hold the GitHub
`workflows` permission and therefore cannot write under .github/workflows.

To activate it, move the file into place and push from an account that
holds that permission:

    mkdir -p .github/workflows
    git mv ci/github-workflow-ci.yml .github/workflows/ci.yml
    git commit -m "Enable CI workflow"
    git push

The workflow defines three jobs. build-and-test pins Zig 0.14.1 and Futhark
0.26.4, type-checks both Futhark sources, builds the CPU kernel library,
verifies formatting with zig fmt --check, builds the inference server and
runs zig build test-all. gpu-source-check confirms that the training entry
points the accelerator binds against are present in main.fut.
python-checks byte-compiles and lints the deployment scripts.

The GPU build itself is not exercised in CI because it needs CUDA hardware;
that path is covered by the Modal harness in scripts/modal_status_bench.py.
