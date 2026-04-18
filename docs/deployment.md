# Hybrid Deployment Guide

## Local mode (Mac)

- Run API locally with `uvicorn app.api.main:app --reload`
- Submit jobs with `mode=local` to process directly on your machine
- Best for quick preview and small clips

## Cloud mode

- Start stack:
  - `docker compose -f infra/docker-compose.yml up --build`
- API receives jobs and pushes task to Redis queue
- Worker executes YOLO + tracking pipeline and saves results

## Runtime routing policy

- Use local mode by default for short clips (<2 minutes)
- Use cloud mode for heavy jobs:
  - high resolution
  - long full-game footage
  - batch processing
