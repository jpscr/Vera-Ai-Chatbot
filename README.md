# Vera Bot — Magicpin AI Challenge

**Submitter:** Jayant Pant  
**Team:** Individual  
**Contact:** jayantpant_23ee116@dtu.ac.in  
**Live API:** https://vera-ai-chatbot-production.up.railway.app

## Overview

Vera Bot is a deterministic, context-aware message composer with HTTP endpoints for context ingestion, scheduled ticks, customer replies, health checks, and metadata.

## Approach

### Signal selection
For each trigger, the system selects a single dominant signal (anchor) from the available merchant state. The selection is based on **weight × magnitude**, helping the message focus on the most relevant signal instead of combining several potentially conflicting signals.

### Deterministic composition
Message composition uses a deterministic template pipeline rather than an LLM call. This makes outputs reproducible and auditable, avoids model timeout risk in the composition path, and allows message facts to be traced through a provenance-tracking facts layer.

### Reply handling
Customer replies are handled by a rule-based intent router. This keeps intent handling predictable and tied to explicit rules.

## Trade-offs and limitations

A deterministic template system is easier to audit and reproduce, but can be less flexible or nuanced than a generative system. Message specificity is also limited by the context supplied to the bot. For example, real pricing and order-level data would improve the accuracy of customer-facing totals and other personalized details.

## API

- `POST /v1/context` — push context
- `POST /v1/tick` — run a tick
- `POST /v1/reply` — handle a customer reply
- `GET /v1/healthz` — health check
- `GET /v1/metadata` — submission metadata

## Deployment

The service is deployed on Railway at the live API URL listed above. The health endpoint has returned `status: ok`; the fresh deployment reported zero loaded context records.
