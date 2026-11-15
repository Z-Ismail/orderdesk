# Deploying orderdesk

Two ways to run it: the whole stack on your own machine with Docker, or a free public demo
on Render + Neon. Prices and limits below were checked in November 2026; confirm them before
relying on them.

## 1. The whole stack locally, in containers

```bash
docker compose --profile app up --build
```

Then open http://localhost:8000 and log in as `demo@orderdesk.app` / `try-orderdesk-demo`.

What happens, in order:

1. `db` starts Postgres with pgvector and waits until it's healthy.
2. `erp` starts the simulated ERP (port 8100) and waits for its health check.
3. `setup` runs once: migrations, catalogue load with real embeddings, the sample inbox through
   the pipeline, your recorded agent investigations attached, the demo login created. It exits.
4. `web` starts only after `setup` succeeded.

Stop with `docker compose --profile app down` (add `-v` to wipe the database too).

## 2. Public demo: Render (app) + Neon (database)

| Piece | Where | Cost | Catch |
| --- | --- | --- | --- |
| Postgres + pgvector | [Neon](https://neon.tech) free plan | $0 | Sleeps after 5 min idle; first query takes a few seconds |
| Web app | [Render](https://render.com) free web service, from the Dockerfile | $0 | Sleeps after 15 min idle; about a minute to wake; 512 MB RAM |
| Simulated ERP | A second Render free web service, same image | $0 | Same; its SQLite file resets on every restart (fine for a demo) |

Avoid Render's own free Postgres for this: free databases expire after 30 days.

### Steps

1. **Neon**: create a project. Copy the connection string (it looks like
   `postgresql://user:pass@ep-xxx.eu-west-2.aws.neon.tech/neondb?sslmode=require`).
   pgvector is available; the app enables it itself on first start.

2. **Render, the ERP**: New → Web Service → connect the GitHub repo → it detects the Dockerfile.
   - Name: `orderdesk-erp`, instance type: Free
   - Docker command: `uvicorn orderdesk.erp_sim.app:create_app --factory --host 0.0.0.0 --port 8000`
   - Environment: `ERP_READ_KEY` and `ERP_WRITE_KEY` set to two long random strings
     (generate with `python -c "import secrets; print(secrets.token_urlsafe(24))"`).
     Never deploy with the `dev-...` keys: the ERP is on the public internet.

3. **Render, the web app**: New → Web Service → same repo, default Docker command.
   - Name: `orderdesk`, instance type: Free, health check path: `/health`
   - Environment:

     | Variable | Value |
     | --- | --- |
     | `DATABASE_URL` | the Neon connection string |
     | `ERP_BASE_URL` | `https://orderdesk-erp.onrender.com` (your ERP's URL) |
     | `ERP_READ_KEY`, `ERP_WRITE_KEY` | the same values as the ERP service |
     | `SECURE_COOKIES` | `true` (Render serves HTTPS) |
     | `DEMO_MODE` | `true` |
     | `DEMO_PASSWORD` | a password you're happy to show publicly on the login page |

     No `ANTHROPIC_API_KEY`: the demo never calls Claude, so there is nothing to steal or run up.

4. **Seed the demo** from your Mac, pointing at the cloud services (one-off, about a minute):

   ```bash
   DATABASE_URL='<neon string>' ERP_BASE_URL='https://orderdesk-erp.onrender.com' \
   ERP_READ_KEY='<read key>' DEMO_MODE=true DEMO_PASSWORD='<demo password>' \
   uv run python -m orderdesk.demo seed
   ```

   Running this locally rather than on Render avoids loading the embedding model inside a
   512 MB free instance.

5. Open your Render URL, log in with the demo credentials, and click around.

### Known limits of the free setup

- **Cold starts**: after 15 idle minutes the first visitor waits about a minute. Upgrade the
  web service to a paid instance to keep it awake.
- **The Reset demo button** loads the embedding model inside the web app. On a 512 MB free
  instance that may run out of memory; if it does, re-run step 4 instead.
- **The ERP forgets its draft orders on restart.** The demo database still remembers approvals,
  so after an ERP restart the two disagree until you reset the demo.

## Production checklist (beyond the demo)

- `DEMO_MODE` off. Real users via `python -m orderdesk.auth add-user` (or a provider like Clerk).
- `SECURE_COOKIES=true`, long random ERP keys, an Anthropic key with a workspace spend limit.
- Database backups (Neon has point-in-time restore on paid plans).
- The pipeline and agent run as scheduled jobs (e.g. a Render cron job every few minutes).
