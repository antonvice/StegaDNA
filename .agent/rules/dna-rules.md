---
trigger: always_on
---

1. use UV and black and ruff chec
2. use mojo (and if needed mojo max) for heavy lifting, make sure to check mojo api for correct usage and examples on how to implement things
3. keep envs in .env and import using python-dotenv
4. write tests
5. we are on mac, train and run everything for mac
6. always use pydantic and type hinting
7. always write docstrings
8. where possible, use loguru, colored, tqdm
9. write tests for critical operations
10. use tenacity for retries, keep weights and biases for monitoring and logging of training and inference
11. Keep state in file on local, use functools for caching where needed
12. make sure on fails we retry, log, try to keep scripts resumable on restarts and mark every status.
13. make sure to check off things when done in roadmap