# Issue Contract — `openai-reasoning-kv-sentinel`

## Pain
Long reasoning fills KV cache with low-value tokens; anchors must survive prune under pressure.

## Claim
Shannon-entropy prune keeps anchors/high-entropy/tail; soft-cap under max_cache_tokens; schema tool gate rejects bad types.

## Proof
```bash
python3 job-app/helix/proofs/proof_openai_kv.py
```

## Done when
Proof exits 0. Architecture (strand/integrity/helix) is **not** a substitute for this proof.

## Anti-claim
Not claiming OpenAI production deployment.
